import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse
from pathlib import Path
import time
from tqdm import tqdm


def load_events_from_file(filepath):
    """Load events from file and return timestamps."""
    print(f"Loading events from {filepath}...")
    start_time = time.time()

    events = []
    try:
        with open(filepath, "r") as f:
            for line in tqdm(f, desc=f"Loading {filepath.name}"):
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 1:
                        # Parse unix timestamp (float)
                        unix_timestamp = float(parts[0])
                        timestamp = pd.to_datetime(unix_timestamp, unit="s")
                        events.append(timestamp)

    except FileNotFoundError:
        print(f"Warning: File {filepath} not found. Using empty event list.")
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")

    elapsed = time.time() - start_time
    print(f"  Completed loading {len(events):,} events in {elapsed:.1f}s")
    return events


# First, create time bins and count events
def create_time_series_data(widen_events, narrow_events, bin_size_seconds=60):
    print("Creating time series data...")
    start_time = time.time()

    # Handle case where one or both event lists are empty
    if not widen_events and not narrow_events:
        print("Warning: No events found in either file.")
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    # Determine time range
    print("  Determining time range...")
    all_events = list(widen_events) + list(narrow_events)
    start_time_data = min(all_events)
    end_time_data = max(all_events)
    print(f"  Time range: {start_time_data} to {end_time_data}")

    # Create time bins using seconds
    print("  Creating time bins...")
    time_bins = pd.date_range(
        start=start_time_data, end=end_time_data, freq=f"{bin_size_seconds}s"
    )
    print(f"  Created {len(time_bins)} time bins")

    # Convert timestamps to pandas Series for binning
    print("  Binning events...")
    widen_series = pd.Series(widen_events)
    narrow_series = pd.Series(narrow_events)

    # Use pd.cut to bin the events and then count them
    print("  Processing widen events...")
    widen_binned = pd.cut(
        widen_series, bins=time_bins, right=False, include_lowest=True
    )
    print("  Processing narrow events...")
    narrow_binned = pd.cut(
        narrow_series, bins=time_bins, right=False, include_lowest=True
    )

    # Count events in each bin
    print("  Counting events in bins...")
    widen_counts = (
        widen_binned.value_counts()
        .reindex(widen_binned.cat.categories, fill_value=0)
        .sort_index()
        .values
    )
    narrow_counts = (
        narrow_binned.value_counts()
        .reindex(narrow_binned.cat.categories, fill_value=0)
        .sort_index()
        .values
    )

    elapsed = time.time() - start_time
    print(f"  Completed time series creation in {elapsed:.1f}s")
    return widen_counts, narrow_counts, time_bins[:-1]  # exclude last bin edge


def main():
    script_start_time = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Poisson models for spread widening/narrowing events"
    )
    parser.add_argument("instrument", help="Instrument name")
    parser.add_argument("period", help="Time period")
    parser.add_argument(
        "--bin-size", type=int, default=60, help="Bin size in seconds (default: 60)"
    )

    args = parser.parse_args()

    # Construct file paths
    data_dir = Path("data")
    widen_file = data_dir / f"{args.instrument}_downwards_{args.period}.txt"
    narrow_file = data_dir / f"{args.instrument}_upwards_{args.period}.txt"

    print(f"Starting analysis for instrument: {args.instrument}, period: {args.period}")
    print(f"Widen events file: {widen_file}")
    print(f"Narrow events file: {narrow_file}")
    print()

    # Load events from files
    widen_events = load_events_from_file(widen_file)
    narrow_events = load_events_from_file(narrow_file)

    print(
        f"\nLoaded {len(widen_events):,} widen events and {len(narrow_events):,} narrow events"
    )

    if not widen_events and not narrow_events:
        print("No events loaded. Exiting.")
        return

    # Get your count data
    widen_counts, narrow_counts, time_index = create_time_series_data(
        widen_events, narrow_events, bin_size_seconds=args.bin_size
    )

    if len(time_index) == 0:
        print("No time bins created. Exiting.")
        return

    # Step 2: Create features DataFrame
    print("\nCreating features DataFrame...")
    features_start_time = time.time()
    features = pd.DataFrame(
        {
            "recent_widen_count": np.roll(widen_counts, 1),  # lag 1
            "recent_narrow_count": np.roll(narrow_counts, 1),  # lag 1
            # Add more features as you have them:
            # 'volume': your_volume_data,
            # 'volatility': your_volatility_data,
            # etc.
        },
        index=time_index,
    )

    # Handle first observation (no lag available)
    features.iloc[0, :2] = 0  # or use forward fill

    features_elapsed = time.time() - features_start_time
    print(f"Features DataFrame created in {features_elapsed:.1f}s")

    # Step 3: Now you can run your models
    print("\nFitting models...")
    model_start_time = time.time()

    X = sm.add_constant(features)  # your covariates

    # Fit separate models
    print("  Fitting widening model...")
    widen_model_start = time.time()
    model_widen = sm.GLM(widen_counts, X, family=sm.families.Poisson()).fit()
    widen_model_elapsed = time.time() - widen_model_start
    print(f"    Completed in {widen_model_elapsed:.1f}s")

    print("  Fitting narrowing model...")
    narrow_model_start = time.time()
    model_narrow = sm.GLM(narrow_counts, X, family=sm.families.Poisson()).fit()
    narrow_model_elapsed = time.time() - narrow_model_start
    print(f"    Completed in {narrow_model_elapsed:.1f}s")

    model_elapsed = time.time() - model_start_time
    print(f"Both models fitted in {model_elapsed:.1f}s")

    print(f"\nWidening model summary for {args.instrument} ({args.period}):")
    print(model_widen.summary())
    print(f"\nNarrowing model summary for {args.instrument} ({args.period}):")
    print(model_narrow.summary())

    total_elapsed = time.time() - script_start_time
    print(f"\n" + "=" * 50)
    print(f"Total execution time: {total_elapsed:.1f}s")
    print(f"Script completed successfully!")


if __name__ == "__main__":
    main()
