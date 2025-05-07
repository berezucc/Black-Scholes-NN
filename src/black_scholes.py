import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, q=0.0, option_type='call'):
    """
    Calculate Black-Scholes option price (call or put), with dividend yield q.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annual)
    sigma : float
        Volatility (annual)
    q : float
        Dividend yield (annual)
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float
        Option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

# Example usage
if __name__ == "__main__":
    # Example parameters
    S = 100  # Current stock price
    K = 120  # Strike price
    T = 0.5  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.3  # Volatility (20%)
    q = 0.01  # Dividend yield (1%)
        
    # Calculate call and put prices
    call_price = black_scholes_price(S, K, T, r, sigma, q, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, q, 'put')
    
    print(f"Call option price: {call_price:.4f}")
    print(f"Put option price: {put_price:.4f}") 