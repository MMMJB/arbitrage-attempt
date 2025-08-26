import argparse
import csv
from datetime import datetime, timedelta
from typing import Union, Tuple
import line_profiler


@line_profiler.profile
def calculate_percent_profit(
    ask_prices: list[float], bid: int
) -> Tuple[float, Union[int, None]]:
    num_exchanges = len(ask_prices)
    max_profit_percentage = 0
    # represent chain of currencies as integer x
    # positive: chain = exch_x -> exch_x+1 -> ... -> exch_x+(n-1)
    # negative: chain = exch_-x -> exch_-x-1 -> ... -> exch_-x-(n-1)
    max_profit_chain: Union[int, None] = None

    for i in range(num_exchanges):
        total_conversion_rate = 1
        for j in range(num_exchanges):
            comparison_conversion = ask_prices[(i + j) % num_exchanges]
            total_conversion_rate *= comparison_conversion

        # forwards_profit_percentage = total_conversion_rate
        # ! assume that 1 / atob is approximately btoa
        backwards_profit_percentage = 1 / total_conversion_rate

        if total_conversion_rate > max_profit_percentage:
            max_profit_percentage = total_conversion_rate
            max_profit_chain = i + 1

        if backwards_profit_percentage > max_profit_percentage:
            max_profit_percentage = backwards_profit_percentage
            max_profit_chain = -1 * (i + 1)

    return max_profit_percentage * 100 - 100, max_profit_chain


def get_chain_from_id(exchanges: list[str], id: Union[int, None]) -> str:
    num_exchanges = len(exchanges)
    exchanges_pool = [exchange.split("-")[0] for exchange in exchanges]
    exchanges_list = []

    if id is None:
        return "unknown"

    for i in range(num_exchanges):
        if id > 0:
            exchanges_list.append(exchanges_pool[(id + i) % num_exchanges])
        else:
            exchanges_list.append(exchanges_pool[(-id - i) % num_exchanges])

    return " -> ".join(exchanges_list)


def get_flipped_exchanges(exchanges: list[str]) -> list[bool]:
    flipped = []
    for exch in exchanges:
        try:
            with open(f"data/{exch}.csv", "r") as f:
                flipped.append(True)
        except FileNotFoundError:
            currency1, currency2 = exch.split("-")
            reversed_exch = f"{currency2}-{currency1}"
            try:
                with open(f"data/{reversed_exch}.csv", "r") as f:
                    flipped.append(False)
                    print(
                        f"Using reversed exchange rate for {exch} as {reversed_exch}."
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Neither {exch} nor {reversed_exch} data files exist."
                )

    return flipped


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


@line_profiler.profile
def process(
    exchanges: list[str],
    flipped_exchanges: list[bool],
    fixed_bid: int,
    profit_threshold: float,
):
    files = [
        open(f"data/{exch if flipped else '-'.join(exch.split('-')[::-1])}.csv", "r")
        for exch, flipped in zip(exchanges, flipped_exchanges)
    ]
    try:
        readers = [csv.reader(f) for f in files]

        first_lines = [next(reader) for reader in readers]

        first_timestamps = [
            get_date(line[1]) for line in first_lines if line is not None
        ]

        min_timestamp = min(first_timestamps)
        max_timestamp = min_timestamp + timedelta(hours=1)

        current_ask_prices: list[Union[float, None]] = [None] * len(exchanges)
        next_ask_price_timestamps: list[datetime] = first_timestamps.copy()
        current_timestamp = min_timestamp
        ready_to_trade = False
        total_percent_profit = 0
        total_trades = 0

        while current_timestamp <= max_timestamp:
            next_timestamp: Union[datetime, None] = None
            next_timestamp_index: Union[int, None] = None
            for i, ts in enumerate(next_ask_price_timestamps):
                if ts is not None and (next_timestamp is None or ts < next_timestamp):
                    next_timestamp = ts
                    next_timestamp_index = i

            if next_timestamp is None or next_timestamp_index is None:
                break

            try:
                row = next(readers[next_timestamp_index])
                ts = get_date(row[1])

                if flipped_exchanges[next_timestamp_index]:
                    current_ask_prices[next_timestamp_index] = float(row[3])
                else:
                    current_ask_prices[next_timestamp_index] = (
                        1 / float(row[3]) if float(row[3]) != 0 else None
                    )

                next_ask_price_timestamps[next_timestamp_index] = ts
            except StopIteration:
                print(
                    f"No more data to read for exchange {exchanges[next_timestamp_index]}."
                )

            if ready_to_trade:
                [best_percent_profit, best_percent_profit_chain] = (
                    calculate_percent_profit(current_ask_prices, fixed_bid)
                )
                if best_percent_profit > profit_threshold:
                    total_percent_profit += best_percent_profit
                    total_trades += 1
                    # print(
                    #     f"Arbitrage opportunity found on {get_chain_from_id(exchanges, best_percent_profit_chain)} for {best_percent_profit}% profit"
                    # )

                # sanity check:
                # product = 1
                # for ask_price in current_ask_prices:
                #     product *= ask_price if ask_price is not None else 1
                # print(product)  # should be ~1
            else:
                ready_to_trade = all(price is not None for price in current_ask_prices)

            current_timestamp = next_timestamp

        print(f"Total arbitrage opportunities: {total_trades}")
        if total_trades > 0:
            print(
                f"Average profit per trade: {total_percent_profit / total_trades * 100:.2f}%"
            )
    finally:
        for f in files:
            f.close()


def main():
    parser = argparse.ArgumentParser(
        description="Detect arbitrage opportunities across n exchange rates"
    )
    parser.add_argument("exchanges", type=str, nargs="+", help="Exchange identifiers")
    parser.add_argument("--fixed_bid", type=int, default=100, help="Fixed buy amount")
    parser.add_argument(
        "--profit_threshold",
        type=float,
        default=0.01,
        help="Minimum profit %% before executing",
    )
    args = parser.parse_args()

    flipped_exchanges = get_flipped_exchanges(args.exchanges)
    process(
        exchanges=args.exchanges,
        flipped_exchanges=flipped_exchanges,
        fixed_bid=args.fixed_bid,
        profit_threshold=args.profit_threshold,
    )


if __name__ == "__main__":
    main()
