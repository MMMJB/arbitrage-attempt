import numpy as np
import pandas as pd
from datetime import datetime
import time

def get_date(date_str: str) -> float:
    """Your existing date parsing function"""
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[9:11])
    minute = int(date_str[12:14])
    second = float(date_str[15:])
    microsecond = int((second % 1) * 1_000_000)
    time = datetime(year, month, day, hour, minute, int(second), microsecond)
    return time.timestamp()

def prepare_forecast_data(csv_file, lookback_events=1000, eps=0.0):
    """
    Prepare your CSV data for real-time spread forecasting
    
    Parameters:
    -----------
    csv_file : str
        Path to your GBP-USD CSV file
    lookback_events : int
        Number of recent events to use for context (default 1000)
    eps : float
        Threshold to filter noise (same as your original code)
    
    Returns:
    --------
    dict with processed data ready for forecasting
    """
    # Load and process data (adapted from your existing code)
    df = pd.read_csv(csv_file, header=None, 
                     names=["instrument", "timestamp", "bid", "ask"],     
                     dtype={"instrument": str, "timestamp": str, "bid": float, "ask": float})
    
    df["timestamp"] = df["timestamp"].apply(get_date)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Convert to relative time
    t0 = df['timestamp'].iloc[0] 
    df['t_sec'] = df['timestamp'] - t0
    
    # Compute spreads and changes
    df['spread'] = df['ask'] - df['bid']
    df['diff'] = df['spread'].diff()
    
    # Filter significant events
    events = df.loc[df['diff'].abs() > eps, ['t_sec', 'diff']].copy()
    events['c'] = (events['diff'] < 0).astype(int)  # 0=widen, 1=narrow
    
    # Take most recent events for context
    recent_events = events.tail(lookback_events).copy().reset_index(drop=True)
    
    # Calculate historical average spread changes
    widen_changes = recent_events.loc[recent_events['c'] == 0, 'diff'].values
    narrow_changes = recent_events.loc[recent_events['c'] == 1, 'diff'].values
    
    avg_widen_size = np.mean(widen_changes) if len(widen_changes) > 0 else 0.0001
    avg_narrow_size = np.mean(narrow_changes) if len(narrow_changes) > 0 else -0.0001
    
    # Get current state
    current_time = recent_events['t_sec'].iloc[-1] + 0.001  # Just after last event
    current_spread = df.loc[df['t_sec'] <= recent_events['t_sec'].iloc[-1], 'spread'].iloc[-1]
    
    return {
        'event_times': recent_events['t_sec'].values,
        'event_types': recent_events['c'].values, 
        'event_changes': recent_events['diff'].values,
        'current_time': current_time,
        'current_spread': current_spread,
        'avg_widen_size': avg_widen_size,
        'avg_narrow_size': avg_narrow_size,
        'total_events': len(recent_events),
        'time_span_hours': (recent_events['t_sec'].iloc[-1] - recent_events['t_sec'].iloc[0]) / 3600
    }

def hawkes_lambda(t, t_history, c_history, mu, alpha, beta):
    """Hawkes intensity function"""
    t_history = np.asarray(t_history)
    c_history = np.asarray(c_history)
    
    lambda_t = mu.copy()
    past_mask = t_history < t
    t_past = t_history[past_mask]
    c_past = c_history[past_mask]
    
    if len(t_past) > 0:
        for k in range(2):
            for j in range(2):
                type_j_events = t_past[c_past == j]
                if len(type_j_events) > 0:
                    dt = t - type_j_events
                    excitation = alpha[k, j] * np.sum(np.exp(-beta[k, j] * dt))
                    lambda_t[k] += excitation
    return lambda_t

def forecast_300ms(data_dict, mu, alpha, beta):
    """
    Generate 300ms spread forecast from your data
    
    Parameters:
    -----------
    data_dict : dict
        Output from prepare_forecast_data()
    mu, alpha, beta : arrays
        Your fitted Hawkes parameters (from 500k fit)
    
    Returns:
    --------
    dict with forecast results
    """
    # Get current Hawkes intensities
    current_lambda = hawkes_lambda(
        data_dict['current_time'],
        data_dict['event_times'], 
        data_dict['event_types'],
        mu, alpha, beta
    )
    
    lambda_widen, lambda_narrow = current_lambda[0], current_lambda[1]
    lambda_total = lambda_widen + lambda_narrow
    time_horizon = 0.3  # 300ms
    
    # Calculate probabilities
    if lambda_total > 0:
        prob_any_event = 1 - np.exp(-lambda_total * time_horizon)
        prob_widen = (lambda_widen / lambda_total) * prob_any_event
        prob_narrow = (lambda_narrow / lambda_total) * prob_any_event
        prob_no_event = 1 - prob_any_event
        
        # Expected change
        expected_delta_spread = (prob_widen * data_dict['avg_widen_size'] + 
                               prob_narrow * data_dict['avg_narrow_size'])
    else:
        prob_widen = prob_narrow = prob_any_event = expected_delta_spread = 0.0
        prob_no_event = 1.0
    
    return {
        'current_spread': data_dict['current_spread'],
        'expected_delta_spread': expected_delta_spread,
        'forecast_spread': data_dict['current_spread'] + expected_delta_spread,
        'lambda_widen': lambda_widen,
        'lambda_narrow': lambda_narrow,
        'prob_widen': prob_widen,
        'prob_narrow': prob_narrow,
        'prob_no_event': prob_no_event,
        'avg_widen_size': data_dict['avg_widen_size'],
        'avg_narrow_size': data_dict['avg_narrow_size'],
        'context_events': data_dict['total_events'],
        'time_span_hours': data_dict['time_span_hours']
    }

# Complete pipeline example
if __name__ == "__main__":
    # Your fitted parameters (500k data - the good ones!)
    mu_fitted = np.array([0.58841823, 0.70208766])
    alpha_fitted = np.array([[2.33355352e-05, 5.91376586e+00],
                            [1.17077446e+01, 6.74719942e-01]])
    beta_fitted = np.array([[2.37373713e+04, 7.99569438e+00],
                           [1.61963724e+01, 4.47351193e+01]])
    
    # Step 1: Load and prepare your data
    print("Loading and preparing data...")
    data = prepare_forecast_data(
        "data/GBP-USD_june25.csv",  # Your CSV file path
        lookback_events=1000,        # Use last 1000 events for context
        eps=0.0                     # Same noise filter as your original
    )
    
    print(f"Data prepared:")
    print(f"- {data['total_events']} events over {data['time_span_hours']:.2f} hours")
    print(f"- Current spread: {data['current_spread']:.6f}")
    print(f"- Average widen: {data['avg_widen_size']:+.6f}")
    print(f"- Average narrow: {data['avg_narrow_size']:+.6f}")
    print()
    
    # Step 2: Generate forecast
    print("Generating 300ms forecast...")
    forecast = forecast_300ms(data, mu_fitted, alpha_fitted, beta_fitted)
    
    print("Results:")
    print(f"Current spread: {forecast['current_spread']:.6f}")
    print(f"Expected Δ in 300ms: {forecast['expected_delta_spread']:+.7f}")
    print(f"Forecast spread: {forecast['forecast_spread']:.6f}")
    print()
    
    print("Market State:")
    print(f"λ_widen: {forecast['lambda_widen']:.4f}")
    print(f"λ_narrow: {forecast['lambda_narrow']:.4f}")
    print(f"Activity level: {forecast['lambda_widen'] + forecast['lambda_narrow']:.4f}")
    print()
    
    print("Probabilities (300ms):")
    print(f"Widen: {forecast['prob_widen']:.1%}")
    print(f"Narrow: {forecast['prob_narrow']:.1%}")
    print(f"No change: {forecast['prob_no_event']:.1%}")
    print()
    
    # Trading interpretation
    if abs(forecast['expected_delta_spread']) > 1e-6:
        direction = "WIDEN" if forecast['expected_delta_spread'] > 0 else "NARROW"
        magnitude_bp = abs(forecast['expected_delta_spread']) * 10000
        dominant_prob = max(forecast['prob_widen'], forecast['prob_narrow'])
        
        print(f"Signal: Expect spread to {direction}")
        print(f"Magnitude: {magnitude_bp:.3f} basis points")
        print(f"Confidence: {dominant_prob:.1%}")
    else:
        print("Signal: No clear direction expected")

def real_time_forecast_function(csv_file, fitted_params):
    """
    Streamlined function for production use
    
    Returns just the essential forecast number
    """
    mu, alpha, beta = fitted_params
    
    data = prepare_forecast_data(csv_file, lookback_events=500)
    forecast = forecast_300ms(data, mu, alpha, beta)
    
    return {
        'expected_change_300ms': forecast['expected_delta_spread'],
        'current_spread': forecast['current_spread'],
        'forecast_spread': forecast['forecast_spread'],
        'confidence': max(forecast['prob_widen'], forecast['prob_narrow'])
    }

# Example of how to use in a loop or real-time system
def trading_system_example():
    """
    Example of how you might use this in a trading system
    """
    # Your parameters
    params = (mu_fitted, alpha_fitted, beta_fitted)
    
    while True:  # In reality, you'd have proper exit conditions
        try:
            # Get forecast
            result = real_time_forecast_function("data/GBP-USD_june25.csv", params)
            
            # Make trading decision
            expected_change = result['expected_change_300ms']
            confidence = result['confidence']
            
            if abs(expected_change) > 2e-5 and confidence > 0.6:  # Thresholds
                direction = "BUY" if expected_change < 0 else "SELL"  # Buy when spread narrows
                print(f"SIGNAL: {direction} - Expected Δ: {expected_change:+.6f}, Confidence: {confidence:.1%}")
            else:
                print(f"HOLD - Weak signal: {expected_change:+.6f}")
                
            time.sleep(0.1)  # Check every 100ms
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)