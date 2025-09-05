import csv
import argparse
from collections import deque
import threading
from typing import Any, Union, Tuple
import time
import signal
import sys
from datetime import datetime


file_suffix = "_april25"


def calculate_ideal_percent_profit(
    instruments: list[str],
    mid_prices: dict[str, float],
) -> Tuple[float, int]:
    num_instruments = len(mid_prices)
    max_profit_percentage = 0
    # represent chain of currencies as integer x
    # positive: chain = exch_x -> exch_x+1 -> ... -> exch_x+(n-1)
    # negative: chain = exch_-x -> exch_-x-1 -> ... -> exch_-x-(n-1)
    max_profit_chain = 0

    for i in range(num_instruments):
        total_conversion_rate = 1
        for j in range(num_instruments):
            target_instrument = instruments[(i + j) % num_instruments]
            comparison_conversion = mid_prices[target_instrument]
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

    return max_profit_percentage, max_profit_chain


def simulate_trade(
    instrument_pair: str,
    amount: float,
    inverse: bool,
    instrument_data: dict[str, deque[dict]],
    instrument_pools: dict[str, float],
):
    latest_entry = instrument_data[instrument_pair][0]
    latest_ask = float(latest_entry[1])
    latest_bid = float(latest_entry[2])

    instrument_a, instrument_b = instrument_pair.split("-")

    # determine source and target currencies depending on direction
    if not inverse:
        # converting A -> B: selling A, so use bid (you receive bid * A units of B)
        source = instrument_a
        target = instrument_b
        amount_of_a = amount
        amount_of_b = amount_of_a * latest_bid
    else:
        # converting B -> A: selling B to get A, use bid on the reversed pair,
        # which in terms of A-B uses ask (you pay ask per A), so amount_A = amount_B / ask
        source = instrument_b
        target = instrument_a
        amount_of_a = amount  # amount of source currency being spent
        amount_of_b = amount_of_a / latest_ask

    if instrument_pools.get(source, 0) < amount_of_a:
        # not enough money to make trade
        return

    print(f"Trading {amount_of_a} {source} -> {amount_of_b} {target}")

    instrument_pools[source] -= amount_of_a
    instrument_pools[target] = instrument_pools.get(target, 0) + amount_of_b

    print(instrument_pools)


def process_md_feed_data(
    instruments: list[str],
    instrument_data: dict[str, deque[dict]],
    instrument_pools: dict[str, float],
    stop_event: threading.Event,
):
    num_instruments = len(instruments)
    ready_to_trade = False
    bid_amount = 1

    while not stop_event.is_set():
        mid_prices: dict[str, float] = {}
        for instrument in instrument_data:
            data_feed = instrument_data[instrument]

            if len(data_feed) == 0:
                continue

            latest_data_point = data_feed[0]
            mid_prices[instrument] = (
                float(latest_data_point[1]) + float(latest_data_point[2])
            ) / 2

        if not ready_to_trade:
            ready_to_trade = all(instrument in mid_prices for instrument in instruments)
            continue

        max_ideal_percent_profit, max_profit_chain_id = calculate_ideal_percent_profit(
            instruments, mid_prices
        )
        max_ideal_percent_profit -= 1

        if max_ideal_percent_profit > 0.0:
            # # ! assume trading is instant
            if max_profit_chain_id > 0:
                # going forwards
                # use x currency a to buy x * conversion_rate b
                for i in range(num_instruments):
                    target_instrument_pair = instruments[
                        (max_profit_chain_id + i) % num_instruments
                    ]

                    simulate_trade(
                        instrument_pair=target_instrument_pair,
                        amount=bid_amount,
                        inverse=False,
                        instrument_data=instrument_data,
                        instrument_pools=instrument_pools,
                    )
            else:
                # going backwards
                # use x currency b to buy x * (1 / conversion_rate) a
                for i in range(num_instruments):
                    target_instrument_pair = instruments[
                        (-max_profit_chain_id - i) % num_instruments
                    ]

                    simulate_trade(
                        instrument_pair=target_instrument_pair,
                        amount=bid_amount,
                        inverse=True,
                        instrument_data=instrument_data,
                        instrument_pools=instrument_pools,
                    )


def get_chain_from_id(instruments: list[str], id: Union[int, None]) -> list[str]:
    num_instruments = len(instruments)
    instruments_list = []

    if id is None or id == 0:
        return instruments_list

    for i in range(num_instruments):
        if id > 0:
            instruments_list.append(instruments[(id + i) % num_instruments])
        else:
            instruments_list.append(instruments[(-id - i) % num_instruments])

    return instruments_list


def process_md_feed(
    feed: Any,
    instrument: str,
    instrument_data: dict[str, deque[dict]],
    flipped: bool,
    stop_event: threading.Event,
):
    print(f"Starting market data feed for instrument {instrument} (flipped={flipped})")

    first_row = next(feed)
    start_time = get_date(first_row[1]).timestamp()
    # keep the first row too (apply flip if needed)
    if flipped:
        date = first_row[1]
        ask = 1.0 / float(first_row[3])  # invert: ask_AB = 1 / bid_BA
        bid = 1.0 / float(first_row[2])  # invert: bid_AB = 1 / ask_BA
        instrument_data[instrument].appendleft(
            [date, str(ask), str(bid)] + first_row[4:]
        )
    else:
        instrument_data[instrument].appendleft(first_row[1:])

    while not stop_event.is_set():
        try:
            new_data = next(feed)
            timestamp = get_date(new_data[1]).timestamp()

            delta = max(0.0, timestamp - start_time)
            time.sleep(delta)

            if flipped:
                date = new_data[1]
                ask = 1.0 / float(new_data[3])
                bid = 1.0 / float(new_data[2])
                instrument_data[instrument].appendleft(
                    [date, str(ask), str(bid)] + new_data[4:]
                )
            else:
                instrument_data[instrument].appendleft(new_data[1:])

            start_time = timestamp
        except StopIteration:
            print(f"No more data to read for instrument {instrument}, aborting...")
            stop_event.set()


def process(
    instruments: list[str],
    flipped_instruments: dict[str, bool],
    pool_size: int,
    stop_event: threading.Event,
):
    num_instruments = len(instruments)
    instrument_names = [instrument.split("-")[0] for instrument in instruments]

    # keep track of the # of tokens for each currency (initialized with pool_size tokens each)
    instrument_pools = dict(zip(instrument_names, [pool_size] * num_instruments))
    # keep a circular buffer of 100 pricepoints for risk analysis
    instrument_data = {instr: deque(maxlen=100) for instr in instruments}

    files = []
    for instrument in instruments:
        flipped = flipped_instruments[instrument]
        path = f"data/{instrument if not flipped else '-'.join(instrument.split('-')[::-1])}{file_suffix}.csv"
        files.append(open(path, "r"))

    readers = [csv.reader(f) for f in files]

    # create market data feeds on separate threads
    threads = []
    for i, instrument in enumerate(instruments):
        thread = threading.Thread(
            target=process_md_feed,
            args=(
                readers[i],
                instrument,
                instrument_data,
                flipped_instruments[instrument],
                stop_event,
            ),
        )
        thread.start()
        threads.append(thread)

    central_thread = threading.Thread(
        target=process_md_feed_data,
        args=(instruments, instrument_data, instrument_pools, stop_event),
    )
    central_thread.start()
    threads.append(central_thread)

    def signal_handler(sig, frame):
        for thread in threads:
            stop_event.set()
            thread.join()

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


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


def get_flipped_instruments(instruments: list[str]) -> dict[str, bool]:
    flipped = {}
    for exch in instruments:
        try:
            with open(f"data/{exch}{file_suffix}.csv", "r"):
                flipped[exch] = False
        except FileNotFoundError:
            currency1, currency2 = exch.split("-")
            reversed_exch = f"{currency2}-{currency1}"
            try:
                with open(f"data/{reversed_exch}{file_suffix}.csv", "r"):
                    flipped[exch] = True
                    print(
                        f"Using reversed instrument rate for {exch} as {reversed_exch}."
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Neither {exch} nor {reversed_exch} data files exist."
                )

    return flipped


def main():
    parser = argparse.ArgumentParser(description="Simulate trades for instrument feeds")
    parser.add_argument(
        "instruments", type=str, nargs="+", help="Instrument identifiers"
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=100,
        help="Starting number of tokens for each instrument",
    )
    parser.add_argument("--latency", type=int, default=50, help="Order latency")
    parser.add_argument(
        "--latency_variation", type=int, default=50, help="Order latency variation (+-)"
    )
    args = parser.parse_args()

    flipped_instruments = get_flipped_instruments(args.instruments)
    stop_event = threading.Event()
    process(
        instruments=args.instruments,
        flipped_instruments=flipped_instruments,
        pool_size=args.pool_size,
        stop_event=stop_event,
    )


if __name__ == "__main__":
    main()
