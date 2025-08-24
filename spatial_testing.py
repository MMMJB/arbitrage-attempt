import csv
from datetime import datetime, timedelta
import random
import argparse
from typing import Union


def check_for_profits(new_ask: Union[float, None], lagging_ask: Union[float, None], fixed_bid: int):
  if new_ask is None or lagging_ask is None:
    return None
  
  # buy at lagging price and sell at new price:
  b_value = fixed_bid * lagging_ask
  b_revenue = b_value / new_ask
  btoa_profit = b_revenue - fixed_bid
  btoa_percent_profit = btoa_profit / fixed_bid

  # buy at new price and sell at lagging price
  a_value = fixed_bid * new_ask
  a_revenue = a_value / lagging_ask
  atob_profit = a_revenue - fixed_bid
  atob_percent_profit = atob_profit / fixed_bid

  return max(atob_percent_profit, btoa_percent_profit)


def process(exch: str, fixed_bid: int, profit_threshold: float):
  with open(f"data/{exch}.csv", "r") as new_file, open(f"data/{exch}_lagging.csv", "r") as lagging_file:
    lines = new_file.readlines()
    new_file.seek(0)
    lagging_file.seek(0)

    last_timestamp = datetime.strptime(lines[-1].split(",")[1], "%Y%m%d %H:%M:%S.%f")
    first_timestamp = datetime.strptime(lines[0].split(",")[1], "%Y%m%d %H:%M:%S.%f")
    range_ms = int((last_timestamp - first_timestamp).total_seconds() * 1000)

    new_reader = csv.reader(new_file)
    lagging_reader = csv.reader(lagging_file)

    current_new_ask_price: Union[float, None] = None
    current_lagging_ask_price: Union[float, None] = None
    next_new_ask_price_timestamp: Union[datetime, None] = None
    lagging_new_ask_price_timestamp: Union[datetime, None] = None
    current_timestamp = first_timestamp

    total_percent_profit = 0
    total_comparisons = 0

    for t in range(range_ms):
      if next_new_ask_price_timestamp is None or current_timestamp >= next_new_ask_price_timestamp:
        try:
          new_row = next(new_reader)

          # Parse timestamps
          new_ts = datetime.strptime(new_row[1], "%Y%m%d %H:%M:%S.%f")

          # Update current prices and timestamps
          current_new_ask_price = float(new_row[3])
          next_new_ask_price_timestamp = new_ts
        except StopIteration:
          # No more data to read
          print("No more new data to read.")
          break
      elif lagging_new_ask_price_timestamp is None or current_timestamp >= lagging_new_ask_price_timestamp:
        try:
          lagging_row = next(lagging_reader)

          # Parse timestamp
          lagging_ts = datetime.strptime(lagging_row[1], "%Y%m%d %H:%M:%S.%f")

          # Update current lagging price and timestamp
          current_lagging_ask_price = float(lagging_row[3])
          lagging_new_ask_price_timestamp = lagging_ts
        except StopIteration:
          # No more data to read
          print("No more lagging data to read.")
          break

      profits = check_for_profits(current_new_ask_price, current_lagging_ask_price, fixed_bid)
      if profits is not None:
        total_percent_profit += profits
        total_comparisons += 1

      # Increment current timestamp by 1 ms
      current_timestamp += timedelta(milliseconds=1)

  average_percent_profit = (total_percent_profit / total_comparisons) if total_comparisons > 0 else 0
  print(f"Average Percent Profit: {average_percent_profit*100}%")


def preprocess(exch: str):
  input_file = f"data/{exch}.csv"
  output_file = f"data/{exch}_lagging.csv"

  try:
    with open(output_file, "r"):
      print(f"Preprocessed file {output_file} already exists. Skipping preprocessing.")
      return
  except FileNotFoundError:
    pass

  with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
      if not row:
        continue  # skip empty lines

      # Format: [pair, timestamp, bid, ask]
      pair = row[0]
      timestamp_str = row[1]
      bid = row[2]
      ask = row[3]

      # Parse timestamp (format: YYYYMMDD HH:MM:SS.sss)
      ts = datetime.strptime(timestamp_str, "%Y%m%d %H:%M:%S.%f")

      # Generate random offset between 50–150 ms
      offset_ms = random.randint(50, 150)
      new_ts = ts - timedelta(milliseconds=offset_ms)

      # Reformat timestamp back to original style
      new_ts_str = new_ts.strftime("%Y%m%d %H:%M:%S.%f")[:-3]

      writer.writerow([pair, new_ts_str, bid, ask])


def main():
  parser = argparse.ArgumentParser(description="Process and detect arbitrage opportunities in exchange data.")
  parser.add_argument("exchange", type=str, help="The exchange identifier (e.g., 'binance')")
  parser.add_argument("--fixed_bid", type=int, default=1000, help="Fixed bid amount for arbitrage calculation")
  parser.add_argument("--profit_threshold", type=float, default=0.01, help="Profit threshold for detecting arbitrage opportunities")
  args = parser.parse_args()

  preprocess(args.exchange)
  process(args.exchange, args.fixed_bid, args.profit_threshold)


if __name__ == "__main__":
    main()