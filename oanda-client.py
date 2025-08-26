import requests
import ujson
from dotenv import load_dotenv
import os
import argparse
import ciso8601
from datetime import datetime, timezone
from typing import Tuple, Union


def calculate_percent_profit(ask_prices: list[float]) -> Tuple[float, Union[int, None]]:
    num_instruments = len(ask_prices)
    max_profit_percentage = 0
    # represent chain of currencies as integer x
    # positive: chain = exch_x -> exch_x+1 -> ... -> exch_x+(n-1)
    # negative: chain = exch_-x -> exch_-x-1 -> ... -> exch_-x-(n-1)
    max_profit_chain: Union[int, None] = None

    for i in range(num_instruments):
        total_conversion_rate = 1
        for j in range(num_instruments):
            comparison_conversion = ask_prices[(i + j) % num_instruments]
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


def get_chain_from_id(
    instruments: list[str], flipped_instruments: dict[str, bool], id: Union[int, None]
) -> str:
    num_instruments = len(instruments)
    instruments_pool = [
        instrument.split("_")[int(flipped_instruments[instrument])]
        for instrument in instruments
    ]
    instruments_list = []

    if id is None:
        return "unknown"

    for i in range(num_instruments):
        if id > 0:
            instruments_list.append(instruments_pool[(id + i) % num_instruments])
        else:
            instruments_list.append(instruments_pool[(-id - i) % num_instruments])

    return " -> ".join(instruments_list)


def connect_to_stream(
    account_id: str,
    token: str,
    instruments: list[str],
    flipped_instruments: dict[str, bool],
) -> None:
    url = f"https://stream-fxpractice.oanda.com/v3/accounts/{account_id}/pricing/stream"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"instruments": ",".join(instruments)}

    num_messages = 0
    total_latency = 0
    num_trades = 0
    total_percent_profit: list[float] = [0] * len(instruments)
    ask_prices: dict[str, float] = {}
    seen_instruments = set[str]()

    try:
        with requests.get(url, headers=headers, params=params, stream=True) as r:
            for line in r.iter_lines():
                received_at = datetime.now(tz=timezone.utc)

                if not line:
                    continue

                try:
                    msg = ujson.loads(line.decode("utf-8"))
                except ValueError as e:
                    print("Failed to parse: ", line, e)
                    continue

                try:
                    if msg["type"] != "PRICE":
                        # heartbeat messages
                        continue
                except KeyError as e:
                    print("Unrecognized message format: ", line)
                    break

                # response format
                # {
                #   "type": "PRICE" | "HEARTBEAT",
                #   "time": string,                     <--- ISO timestamp
                #   "bids": {
                #       "price": string                 <--- float
                #       "liquidity": number             <--- int
                #   }[],
                #   "asks": {
                #       "price": string                 <--- float
                #       "liquidity": number             <--- int
                #   }[],
                #   "closeoutBid": string,              <--- float
                #   "closeoutAsk": string,              <--- float
                #   "status": ?,
                #   "tradeable": boolean,
                #   "instrument": string
                # }

                instrument = msg["instrument"]

                new_ask_price = max(float(ask["price"]) for ask in msg["asks"])
                if flipped_instruments[instrument]:
                    new_ask_price = 1 / new_ask_price

                ask_prices[instrument] = new_ask_price

                if not instrument in seen_instruments:
                    seen_instruments.add(instrument)
                    continue

                exch_timestamp = ciso8601.parse_datetime(msg["time"])
                latency = (
                    received_at.timestamp() - exch_timestamp.timestamp()
                ) * 1000  # ms

                total_latency += latency
                num_messages += 1

                average_latency = total_latency / num_messages

                best_percent_profit, chain = calculate_percent_profit(
                    list(ask_prices.values())
                )

                print(
                    f"Message {num_messages}: Latency = {latency:.2f} ms (average {average_latency:.2f} ms)"
                )
                print(
                    f"Best percent profit: {best_percent_profit:.6f} on chain {get_chain_from_id(instruments, flipped_instruments, chain)}"
                )

                if num_messages == 1000:
                    r.close()
    except AttributeError as e:
        print(e)
        pass

    average_latency = total_latency / num_messages
    print(f"Average latency {average_latency:.2f}ms")


def get_account_id(token: str) -> str:
    url = "https://api-fxpractice.oanda.com/v3/accounts"
    headers = {"Authorization": f"Bearer {token}"}

    res = requests.get(url, headers=headers)

    # response format
    # {"accounts":[{"tags":[],"id":"101-001-36196314-001"}]}
    return res.json()["accounts"][0]["id"]


def get_valid_instruments(account_id: str, token: str) -> list[str]:
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/instruments"
    headers = {"Authorization": f"Bearer {token}"}

    res = requests.get(url, headers=headers)

    # response format
    # {"instruments":[{"name": "EUR_USD"},...]}}
    return [instrument["name"] for instrument in res.json()["instruments"]]


def main():
    load_dotenv()

    api_token = os.getenv("OANDA_API_KEY")

    if not api_token:
        raise ValueError(
            "No API token found in environment; make sure you have a .env file containing OANDA_API_KEY in root"
        )

    parser = argparse.ArgumentParser(
        description="Process and detect arbitrage opportunities in instrument data."
    )
    parser.add_argument(
        "instruments",
        type=str,
        nargs="+",
        help="The instrument identifiers (ex. EUR_USD)",
    )
    args = parser.parse_args()

    account_id = get_account_id(api_token)
    valid_instruments = get_valid_instruments(account_id, api_token)

    flipped_instruments: dict[str, bool] = {}
    actual_instruments: list[str] = []
    for instrument in args.instruments:
        if instrument in valid_instruments:
            flipped_instruments[instrument] = False
            actual_instruments.append(instrument)
        else:
            start, end = instrument.split("_")
            reversed_instrument = f"{end}_{start}"

            if reversed_instrument not in valid_instruments:
                raise ValueError(f"No instrument {instrument} or {reversed_instrument}")

            print(f"Switching {instrument} -> {reversed_instrument}")

            flipped_instruments[reversed_instrument] = True
            actual_instruments.append(reversed_instrument)

    connect_to_stream(account_id, api_token, actual_instruments, flipped_instruments)


if __name__ == "__main__":
    main()
