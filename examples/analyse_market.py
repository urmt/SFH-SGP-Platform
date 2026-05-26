"""
Analyse stock/crypto data with SFH-SGP.

Fetches price data from a free CSV source (Yahoo Finance via yfinance
if available, otherwise uses a direct CSV download from stooq.com).

Usage:
    python examples/analyse_market.py --symbol AAPL --days 365
    python examples/analyse_market.py --symbol BTC-USD --days 180
"""
import argparse
import sys
import urllib.request
import csv
import io
import numpy as np
from sfh_sgp.core.embedding import EmbeddingEngine
from sfh_sgp.core.transforms import TransformEngine
from sfh_sgp.core.geometry import GeometryAnalyser
from sfh_sgp.core.guard import ArchitectureGuard


def fetch_stooq(symbol: str, days: int = 365) -> np.ndarray:
    """Fetch closing prices from stooq.com (free, no API key needed)."""
    url = f'https://stooq.com/q/d/l/?s={symbol}&i=d'
    print(f'  Fetching {symbol} from stooq.com ...')
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        text = resp.read().decode()
        prices = []
        for row in csv.DictReader(io.StringIO(text)):
            val = row.get('Close', '').strip()
            if val:
                prices.append(float(val))
        if len(prices) < 10:
            raise ValueError(f'Only got {len(prices)} prices')
        arr = np.array(prices, dtype=float)
        print(f'  Got {len(arr)} daily closing prices')
        return arr
    except Exception as e:
        print(f'  Stooq fetch failed: {e}')
        print(f'  Try: https://stooq.com/q/d/l/?s={symbol}&i=d in your browser')
        sys.exit(1)


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Convert prices to log returns for a stationary signal."""
    return np.diff(np.log(prices))


def main():
    parser = argparse.ArgumentParser(description='Analyse market data with SFH-SGP')
    parser.add_argument('--symbol', default='AAPL', help='Ticker symbol (e.g. AAPL, BTC-USD, ^SPX)')
    parser.add_argument('--days', type=int, default=365, help='Days of history')
    parser.add_argument('--mode', choices=['prices', 'returns'], default='returns',
                        help='Signal type: raw prices or log returns')
    parser.add_argument('--serve', action='store_true', help='Also start API server')
    args = parser.parse_args()

    # 1. Fetch market data
    prices = fetch_stooq(args.symbol, args.days)

    # 2. Choose signal (prices or returns)
    if args.mode == 'returns':
        signal = compute_returns(prices)
        signal_name = f'{args.symbol}_log_returns'
    else:
        signal = prices
        signal_name = f'{args.symbol}_prices'

    print(f'\nSignal length: {len(signal)} points')
    print(f'Signal range:  {signal.min():.4f} to {signal.max():.4f}')

    # 3. Embed with V2_079
    print(f'\n--- V2_079 Embedding ---')
    guard = ArchitectureGuard()
    engine = EmbeddingEngine(guard)
    transformer = TransformEngine()

    result = engine.embed(signal, signal_id=signal_name, transform='base')
    print(f'  {signal_name}')
    print(f'  Architecture: {result.architecture} (frozen: {guard.validated})')
    print(f'  m1  signed_ordinal_flow:        {result.m1:.6f}')
    print(f'  m2  half_correlation:           {result.m2:.6f}')
    print(f'  m3  signed_compressibility:     {result.m3:.6f}')
    print(f'  m4  amplitude_transition_asym:  {result.m4:.6f}')
    print(f'  Embedding: [{result.m1:.4f}, {result.m2:.4f}, {result.m3:.4f}, {result.m4:.4f}]')

    # 4. Apply transforms
    print(f'\n--- Transform Analysis ---')
    for t_name in ['reverse', 'swap', 'replay', 'stitch']:
        tx = transformer.apply(signal, t_name)
        tr = engine.embed(tx, signal_id=f'{signal_name}_{t_name}', transform=t_name)
        delta = tr.vector - result.vector
        print(f'  {t_name:8s}  Δ=[{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}, {delta[3]:+.4f}]  '
              f'|Δ|={np.linalg.norm(delta):.4f}')

    # 5. Geometry analysis (needs multiple signals)
    print(f'\n--- Geometry Analysis ---')
    print(f'  (Requires multiple signals; run with --multi for batch mode)')
    print(f'  Single-signature: PC1 ~1.0 if transform-collapsed')

    # 6. Interpretation hints
    print(f'\n--- Interpretation ---')
    print(f'  m1 > 0.3 : Strong upward trend bias (bullish)')
    print(f'  m1 < -0.3: Strong downward trend bias (bearish)')
    print(f'  m2 ~ 1.0 : First and second halves are highly correlated (trend persistence)')
    print(f'  m2 ~ 0   : Regime change mid-series')
    print(f'  m2 < -0.3: Anti-correlated halves (possible reversal pattern)')
    print(f'  m3 >> 0  : First half more entropic (more random / volatile)')
    print(f'  m3 << 0  : Second half more entropic')
    print(f'  m4 ~ 0   : Symmetric amplitude transitions')
    print(f'  m4 > 0.2 : Large upward moves > large downward moves')
    print(f'  m4 < -0.2: Large downward moves > large upward moves')


if __name__ == '__main__':
    main()
