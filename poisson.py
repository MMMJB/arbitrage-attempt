import numpy as np

def lambda_widen(recent_widen_count, recent_narrow_count, time_of_day):
    """
    Intensity function for widening events (events per second)
    """
    return np.exp(0.2353 + 
                  0.0905 * recent_widen_count + 
                  0.0275 * recent_narrow_count + 
                  0.0254 * time_of_day)

def lambda_narrow(recent_widen_count, recent_narrow_count, time_of_day):
    """
    Intensity function for narrowing events (events per second)
    """
    return np.exp(0.1345 + 
                  0.0834 * recent_widen_count + 
                  0.0363 * recent_narrow_count + 
                  0.0220 * time_of_day)

# Example usage:
hour = 14  # 2 PM
recent_w = 2  # 2 widening events in last period
recent_n = 1  # 1 narrowing event in last period

λ_w = lambda_widen(recent_w, recent_n, hour)
λ_n = lambda_narrow(recent_w, recent_n, hour)

print(f"Widening intensity: {λ_w:.4f} events/second")
print(f"Narrowing intensity: {λ_n:.4f} events/second")

def spread_change_probabilities(recent_widen, recent_narrow, hour, dt=0.1):
    """
    Calculate probabilities for next dt seconds and print all values
    """
    
    # Get intensities
    λ_w = lambda_widen(recent_widen, recent_narrow, hour)
    λ_n = lambda_narrow(recent_widen, recent_narrow, hour)
    
    
    # For small dt, use linear approximation
    p_widen = (1-np.exp(-λ_w * dt))*np.exp(-λ_n * dt)
    p_narrow = (1-np.exp(-λ_n * dt))*np.exp(-λ_w * dt)
    p_widen_and_narrow = (1 - np.exp(-λ_w * dt)) * (1 - np.exp(-λ_n * dt))
    p_any_change = 1 - np.exp(-λ_w * dt) * np.exp(-λ_n * dt)
    p_no_change = np.exp(-λ_w * dt) * np.exp(-λ_n * dt)
    
    
    

    # Summary
    expected_value_delta_spread = p_widen*1.3823143652012289e-05 - p_narrow*1.4469175071553994e-05
    print(f"  • Expected value change in spread = {expected_value_delta_spread:.8f}")
    return {
        'lambda_widen': λ_w,
        'lambda_narrow': λ_n,
        'p_no_change': p_no_change,
        'p_widen': p_widen,
        'p_narrow': p_narrow,
        'p_any_change': p_any_change,
       
    }

# Example usage:
spread_change_probabilities(recent_widen=12, recent_narrow=0, hour=14, dt=0.3)

