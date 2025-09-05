import argparse
import csv
from datetime import datetime
from typing import Union
import numpy as np


def lambda_widen(recent_widen_count: int, recent_narrow_count: int, hour_of_day: int):
    return np.exp(
        0.2353
        + 0.0905 * recent_widen_count
        + 0.0275 * recent_narrow_count
        + 0.0254 * hour_of_day
    )


def lambda_narrow(recent_widen_count: int, recent_narrow_count: int, hour_of_day: int):
    return np.exp(
        0.1345
        + 0.0834 * recent_widen_count
        + 0.0363 * recent_narrow_count
        + 0.0220 * hour_of_day
    )


def predict_spread(
    current_spread: float,
    recent_widen_count: int,
    recent_narrow_count: int,
    hour_of_day: int,
    projected_seconds: float,
) -> float:
    lambda_w = lambda_widen(recent_widen_count, recent_narrow_count, hour_of_day)
    lambda_n = lambda_narrow(recent_widen_count, recent_narrow_count, hour_of_day)

    p_widen = (1 - np.exp(-lambda_w * projected_seconds)) * np.exp(
        -lambda_n * projected_seconds
    )
    p_narrow = (1 - np.exp(-lambda_n * projected_seconds)) * np.exp(
        -lambda_w * projected_seconds
    )

    expected_value_delta_spread = (
        p_widen * 1.3823143652012289e-05 - p_narrow * 1.4469175071553994e-05
    )

    return current_spread + expected_value_delta_spread


def simulate_prediction_accuracy(
    instrument: str, period: str, exchange_latency: float, flipped: bool
):
    sliding_upward_timestamps: list[float] = []
    sliding_downward_timestamps: list[float] = []
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
                sliding_upward_timestamps = [
                    t for t in sliding_upward_timestamps if timestamp - t <= 1
                ]
                sliding_downward_timestamps = [
                    t for t in sliding_downward_timestamps if timestamp - t <= 1
                ]

                if spread > last_spread:
                    sliding_upward_timestamps.append(timestamp)
                elif spread < last_spread:
                    sliding_downward_timestamps.append(timestamp)

                num_upward = len(sliding_upward_timestamps)
                num_downward = len(sliding_downward_timestamps)
                prediction_timestamp = timestamp + (exchange_latency / 1000)

                # make prediction
                predicted_spread = predict_spread(
                    current_spread=spread,
                    recent_narrow_count=num_downward,
                    recent_widen_count=num_upward,
                    hour_of_day=_,
                    projected_seconds=exchange_latency / 1000,
                )
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
    args = parser.parse_args()

    flipped = instrument_is_flipped(args.instrument, args.period)
    simulate_prediction_accuracy(
        args.instrument, args.period, args.exchange_latency, flipped
    )


if __name__ == "__main__":
    main()
