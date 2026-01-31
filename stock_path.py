import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    pref = input_preference()
    if pref == "manual":
        S0, mu, sigma, T, n = manual_input()
    else:
        ticker, S0, T, n = market_input()
        mu, sigma = calculate_mu_and_sigma(ticker, T)
    return stock_path(S0, mu, sigma, T, n)


def stock_path(S0, mu, sigma, T, n, seed=None):
    S0, mu, sigma, T, n = S0, mu, sigma, T, n
    N = max(1,int(T * 252))
    dt = T/N

    rng = np.random.default_rng(seed)
    dW = rng.standard_normal(size=(N, n))

    increments = (mu - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * dW
    logS = np.empty((N+1, n))
    logS[0, :] = math.log(S0)
    logS[1:, :] = logS[0, :] + np.cumsum(increments, axis=0)
    S = np.exp(logS)
    ST = S[-1, :]

    q01, qo5, q95, q99 = np.quantile(ST, [0.01, 0.05, 0.95, 0.99])
    CI95 = np.quantile(ST, [0.025, 0.975])
    CI99 = np.quantile(ST, [0.005, 0.995])

    ST_mean = np.mean(ST)
    ES = S0*math.exp(mu*T)
    print(f"\nCurrent stock price is {S0}")
    print("-"*60)
    print(f"Average terminal simulated stock price is {round(ST_mean, 5)}")
    print("-"*60)
    print(f"Expected terminal stock price is: {round(ES, 5)}")

    print(f"\nConfidence intervals for S_T (Monte Carlo percentiles):")
    print(f"  95% interval [2.5%, 97.5%]: [{CI95[0]:.4f}, {CI95[1]:.4f}]")
    print(f"  99% interval [0.5%, 99.5%]: [{CI99[0]:.4f}, {CI99[1]:.4f}]")

    plt.figure()
    plt.hist(ST, bins=60, alpha=0.75)

    plt.axvline(q01, linestyle="--", linewidth=2, label="1%")
    plt.axvline(qo5, linestyle="--", linewidth=2, label="5%")
    plt.axvline(q95, linestyle="--", linewidth=2, label="95%")
    plt.axvline(q99, linestyle="--", linewidth=2, label="99%")

    plt.title("Terminal stock price distribution with quantiles")
    plt.xlabel("S_T")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    t = np.linspace(0, T, N+1)

    plt.figure()
    plt.plot(t, S)
    plt.title("GBM Simulation of Stock Prices Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock price")
    plt.show()

    return {"paths": S, "t": t, "ST_mean": ST_mean, "ES": ES}


def input_preference():
    while True:
        preference = input("State your input preference (either manual or market data): ").strip().lower()
        try:
            if preference in ["manual", "market"]:
                return preference
            else:
                raise ValueError
        except ValueError:
            print("Input preference should be either 'manual' or 'market' ")
        

def manual_input():
    S0 = float(input("Input stock's initial price at t = 0: "))
    mu = float(input("Input expected (required) return (mu) on the stock in decimal format (ex.: 0.04): "))
    sigma = float(input("Input the estimate of stock's volatility in decimal format (ex.: 0.2): "))
    T = float(input("Input the timeline in years (ex.: 1 or 0.5) for stock's price path simulation: "))
    n = int(input("Input the number of stock price paths simulations as an integer: "))
    return S0, mu, sigma, T, n


def market_input():
    while True:
        try:
            tick = input("Input ticker of the stock whose price path you wish to simulate: ")
            ticker = yf.Ticker(tick)
            if ticker.history(period="1d").empty:
                raise ValueError
        except ValueError:
            print(f"No data found for {tick}")

        latest_stock_data = ticker.history(period="1d")
        S0 = round(float(latest_stock_data["Close"].iloc[-1]), 4)

        T = float(input("Input the timeline in years (ex.: 1 or 0.5) for stock's price path simulation: "))
        n = int(input("Input the number of stock price paths simulations as an integer: "))
        
        return tick, S0, T, n


def calculate_mu_and_sigma(ticker, T):
    df = yf.download(ticker, period="1y", interval="1d")
    returns = np.log(df['Close']/df['Close'].shift(1)).dropna()
    mu = returns.mean().item()

    sigma = returns.std().item()

    return mu*252, sigma*np.sqrt(252)


if __name__ == "__main__":
    main()