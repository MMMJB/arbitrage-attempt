import argparse
from datetime import datetime
import csv
from typing import Union


file_suffix = "_june25"


def process(instrument: str, flipped: bool):
    file = open(
        f"data/{instrument if flipped else '-'.join(instrument.split('-')[::-1])}{file_suffix}.csv",
        "r",
    )

    try:
        reader = csv.reader(file)

        last_spread: Union[float, None] = None
        instrument_timestamps_going_up: list[tuple[float, float, float]] = []
        instrument_timestamps_going_down: list[tuple[float, float, float]] = []

        for line in reader:
            bid = float(line[2])
            ask = float(line[3])

            if flipped:
                bid, ask = 1 / ask, 1 / bid

            spread = ask - bid
            timestamp = get_date(line[1]).timestamp()

            if last_spread and spread < last_spread:
                instrument_timestamps_going_down.append(
                    (timestamp, spread - last_spread, spread)
                )
            elif last_spread and spread > last_spread:
                instrument_timestamps_going_up.append(
                    (timestamp, spread - last_spread, spread)
                )

            last_spread = spread

        with open(f"data/{instrument}_upwards{file_suffix}.txt", "w") as f:
            f.write(
                "\n".join(
                    [f"{t[0]} {t[1]} {t[2]}" for t in instrument_timestamps_going_up]
                )
            )

        with open(f"data/{instrument}_downwards{file_suffix}.txt", "w") as f:
            f.write(
                "\n".join(
                    [f"{t[0]} {t[1]} {t[2]}" for t in instrument_timestamps_going_down]
                )
            )

        average_spread_change_up = sum(
            [t[1] for t in instrument_timestamps_going_up]
        ) / len(instrument_timestamps_going_up)
        average_spread_change_down = sum(
            [t[1] for t in instrument_timestamps_going_down]
        ) / len(instrument_timestamps_going_down)

        print(
            f"Average spread change upwards for {instrument}: {average_spread_change_up}"
        )
        print(
            f"Average spread change downwards for {instrument}: {average_spread_change_down}"
        )

    finally:
        file.close()


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


def get_flipped_instruments(instruments: list[str]) -> list[bool]:
    flipped = []
    for exch in instruments:
        try:
            with open(f"data/{exch}{file_suffix}.csv", "r") as f:
                flipped.append(True)
        except FileNotFoundError:
            currency1, currency2 = exch.split("-")
            reversed_exch = f"{currency2}-{currency1}"
            try:
                with open(f"data/{reversed_exch}{file_suffix}.csv", "r") as f:
                    flipped.append(False)
                    print(
                        f"Using reversed instrument rate for {exch} as {reversed_exch}."
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Neither {exch} nor {reversed_exch} data files exist."
                )

    return flipped


def main():
    parser = argparse.ArgumentParser(
        description="Get hawkes constants for instrument feed"
    )
    parser.add_argument("instrument", type=str, help="instrument identifiers")
    args = parser.parse_args()

    flipped_instruments = get_flipped_instruments([args.instrument])
    process(
        instrument=args.instrument,
        flipped=flipped_instruments[0],
    )


if __name__ == "__main__":
    main()
