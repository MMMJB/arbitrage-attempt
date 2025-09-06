import argparse
import csv
from datetime import datetime
from typing import Union
import numpy as np


def hawkes_lambda(
    t: float, t_history: list[float], c_history: list[int], mu, alpha, beta
):
    t_history_as_array = np.asarray(t_history)
    c_history_as_array = np.asarray(c_history)

    lambda_t = mu.copy()
    past_mask = t_history_as_array < t
    t_past = t_history_as_array[past_mask]
    c_past = c_history_as_array[past_mask]

    if len(t_past) > 0:
        for k in range(2):
            for j in range(2):
                type_j_events = t_past[c_past == j]
                if len(type_j_events) > 0:
                    dt = t - type_j_events
                    excitation = alpha[k, j] * np.sum(np.exp(-beta[k, j] * dt))
                    lambda_t[k] += excitation
    return lambda_t


def predict_delta_spread(
    current_time: float,
    event_timestamps: list[float],
    event_types: list[int],  # 0 = widen, 1 = narrow
    projected_seconds: float,
) -> float:
    current_lambda = hawkes_lambda(
        current_time,
        event_timestamps,
        event_types,
        mu=np.array([0.58841823, 0.70208766]),
        alpha=np.array(
            [[2.33355352e-05, 5.91376586e00], [1.17077446e01, 6.74719942e-01]]
        ),
        beta=np.array([[2.37373713e04, 7.99569438e00], [1.61963724e01, 4.47351193e01]]),
    )

    lambda_widen, lambda_narrow = current_lambda[0], current_lambda[1]
    lambda_total = lambda_widen + lambda_narrow

    if lambda_total > 0:
        prob_any_event = 1 - np.exp(-lambda_total * projected_seconds)
        prob_widen = (lambda_widen / lambda_total) * prob_any_event
        prob_narrow = (lambda_narrow / lambda_total) * prob_any_event

        # Expected change
        expected_delta_spread = (
            prob_widen * 1.6562525075872022e-05 + prob_narrow * -1.6240756328973555e-05
        )
    else:
        expected_delta_spread = 0.0

    return expected_delta_spread


def simulate_prediction_accuracy(
    instrument: str,
    period: str,
    exchange_latency: float,
    window_size: int,
    flipped: bool,
    naive: bool,
):
    sliding_timestamps: list[float] = []
    sliding_timestamp_types: list[int] = []  # 0 = widen, 1 = narrow
    # pair of timestamp, predicted spread
    predictions_awaiting_matches: list[tuple[float, float]] = []
    total_predictions = 0
    total_prediction_error = 0

    last_spread: Union[float, None] = None

    path = f"data/{instrument if not flipped else '-'.join(instrument.split('-')[::-1])}_{period}.csv"
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            bid = float(row[2])
            ask = float(row[3])

            if flipped:
                bid, ask = 1 / ask, 1 / bid

            spread = ask - bid
            timestamp = get_date(row[1]).timestamp()

            for pred_timestamp, pred_spread in predictions_awaiting_matches[:]:
                if timestamp <= pred_timestamp:
                    # 1 second hasn't elapsed
                    continue

                # the actual spread is the last recorded spread (it would've traded at the last spread)
                if last_spread is not None and last_spread != 0:
                    error = abs((last_spread - pred_spread) / last_spread)
                    total_prediction_error += error
                    total_predictions += 1

                predictions_awaiting_matches.remove((pred_timestamp, pred_spread))

            if last_spread is not None:
                # only keep the newest window_size elements
                sliding_timestamps = sliding_timestamps[-(window_size - 1) :]
                sliding_timestamp_types = sliding_timestamp_types[-(window_size - 1) :]

                sliding_timestamps.append(timestamp)
                if spread > last_spread:
                    sliding_timestamp_types.append(0)
                elif spread < last_spread:
                    sliding_timestamp_types.append(1)
                else:
                    sliding_timestamps.pop()  # no change, don't add

                prediction_timestamp = timestamp + (exchange_latency / 1000)

                # make prediction
                if not naive:
                    predicted_delta_spread = predict_delta_spread(
                        current_time=timestamp,
                        event_timestamps=sliding_timestamps,
                        event_types=sliding_timestamp_types,
                        projected_seconds=exchange_latency / 1000,
                    )
                    predicted_spread = spread + predicted_delta_spread
                else:
                    predicted_spread = spread
                # add it to pending predictions
                predictions_awaiting_matches.append(
                    (prediction_timestamp, predicted_spread)
                )

            last_spread = spread

            average_error = (
                total_prediction_error / total_predictions
                if total_predictions > 0
                else 0
            )
            print(
                f"Running average prediction error: {(average_error * 100):.2f}%",
                end="\r",
            )


def instrument_is_flipped(instrument: str, period: str) -> bool:
    try:
        with open(f"data/{instrument}_{period}.csv", "r"):
            return False
    except FileNotFoundError:
        currency1, currency2 = instrument.split("-")
        reversed_exch = f"{currency2}-{currency1}"
        try:
            with open(f"data/{reversed_exch}_{period}.csv", "r"):
                print(
                    f"Using reversed instrument rate for {instrument} as {reversed_exch}."
                )
                return True
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Neither {instrument} nor {reversed_exch} data files exist."
            )


def get_date(date_str: str) -> datetime:
    # dates are stored in yyyyMMdd hh:mm:ss
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[9:11])
    minute = int(date_str[12:14])
    second = float(date_str[15:])
    microsecond = int((second % 1) * 1_000_000)
    return datetime(year, month, day, hour, minute, int(second), microsecond)


def main():
    parser = argparse.ArgumentParser(description="Simulate prediction accuracy")
    parser.add_argument("instrument", type=str, help="Instrument identifier")
    parser.add_argument("period", type=str, help="Time period for prediction")
    parser.add_argument("exchange_latency", type=float, help="Exchange latency in ms")
    parser.add_argument(
        "window_size", type=int, help="Window size for sliding timestamps"
    )
    parser.add_argument(
        "--naive", action="store_true", help="Use naive prediction (no Hawkes)"
    )
    args = parser.parse_args()

    flipped = instrument_is_flipped(args.instrument, args.period)
    simulate_prediction_accuracy(
        args.instrument,
        args.period,
        args.exchange_latency,
        args.window_size,
        flipped,
        args.naive,
    )


if __name__ == "__main__":
    main()
