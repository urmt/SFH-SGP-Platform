"""
Analyse health time-series data with SFH-SGP.

Works with synthetic heart-rate data, or real CSV exports from
wearables (Apple Health, Fitbit, Garmin).

Usage:
    python examples/analyse_health.py                          # synthetic demo
    python examples/analyse_health.py --csv heart_rate.csv     # your own data
"""
import argparse
import csv
import numpy as np
from sfh_sgp.core.embedding import EmbeddingEngine
from sfh_sgp.core.transforms import TransformEngine
from sfh_sgp.core.geometry import GeometryAnalyser
from sfh_sgp.core.anomaly import AnomalyDetector
from sfh_sgp.core.guard import ArchitectureGuard


def synthetic_heart_rate(minutes: int = 1440, seed: int = 42) -> np.ndarray:
    """Generate synthetic heart-rate data (one day, ~60 bpm with variation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(minutes, dtype=float)
    base_hr = 60 + 10 * np.sin(2 * np.pi * t / 1440)  # circadian rhythm
    base_hr += 15 * np.exp(-((t - 720) ** 2) / (2 * 60 ** 2))  # lunch spike
    noise = rng.normal(0, 3, minutes)
    return base_hr + noise


def synthetic_blood_pressure(days: int = 30, seed: int = 42) -> np.ndarray:
    """Generate synthetic daily systolic BP readings."""
    rng = np.random.default_rng(seed)
    t = np.arange(days, dtype=float)
    systolic = 120 + 5 * np.sin(2 * np.pi * t / 7)  # weekly pattern
    systolic += rng.normal(0, 4, days)
    return systolic


def load_csv(path: str) -> np.ndarray:
    """Load a single-column CSV of numeric values."""
    values = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                try:
                    values.append(float(row[-1]))  # take last column
                except ValueError:
                    continue
    if len(values) < 10:
        raise ValueError(f'Only {len(values)} valid values in {path}')
    return np.array(values, dtype=float)


def main():
    parser = argparse.ArgumentParser(description='Analyse health data with SFH-SGP')
    parser.add_argument('--csv', help='CSV file with health readings (last column = values)')
    parser.add_argument('--type', choices=['heart_rate', 'blood_pressure', 'steps', 'other'],
                        default='heart_rate', help='Type of health signal')
    parser.add_argument('--minutes', type=int, default=1440,
                        help='Minutes of synthetic data (default: 1440 = 1 day)')
    args = parser.parse_args()

    # Load or generate signal
    if args.csv:
        signal = load_csv(args.csv)
        signal_name = args.csv
    else:
        if args.type == 'heart_rate':
            signal = synthetic_heart_rate(args.minutes)
            signal_name = f'synthetic_heart_rate_{args.minutes}min'
        elif args.type == 'blood_pressure':
            signal = synthetic_blood_pressure()
            signal_name = 'synthetic_bp_30days'
        else:
            signal = synthetic_heart_rate(args.minutes)
            signal_name = f'synthetic_{args.type}'

    print(f'Signal:       {signal_name}')
    print(f'Length:       {len(signal)} points')
    print(f'Range:        {signal.min():.1f} - {signal.max():.1f}')
    print(f'Mean:         {signal.mean():.1f} ± {signal.std():.1f}')

    # Embed
    print(f'\n--- V2_079 Health Embedding ---')
    guard = ArchitectureGuard()
    engine = EmbeddingEngine(guard)
    transformer = TransformEngine()

    result = engine.embed(signal, signal_id=signal_name, transform='base')
    print(f'  m1 (ordinal flow):           {result.m1:.6f}')
    print(f'  m2 (half correlation):       {result.m2:.6f}')
    print(f'  m3 (signed compressibility): {result.m3:.6f}')
    print(f'  m4 (amp transition asym):    {result.m4:.6f}')

    # Anomaly detection (train on synthetic windowed segments)
    print(f'\n--- Regime Detection ---')
    window = 120
    if len(signal) > window * 3:
        segments = np.array([
            signal[i:i + window] for i in range(0, len(signal) - window, window // 2)
        ])
        emb_segments = np.array([engine.embed(s).vector for s in segments])
        # Create pseudo-labels: first half vs second half
        mid = len(emb_segments) // 2
        pseudo_labels = np.array([0] * mid + [1] * (len(emb_segments) - mid))
        detector = AnomalyDetector()
        detector.train(emb_segments, pseudo_labels)
        for i in range(0, len(emb_segments), max(1, len(emb_segments) // 5)):
            event = detector.score(emb_segments[i])
            if event:
                ts = f'window_{i * (window // 2)}-{i * (window // 2) + window}'
                print(f'  Regime shift at {ts}: {event.regime} (confidence={max(event.probabilities):.3f})')
        if detector.score(emb_segments[-1]) is None:
            print('  No significant regime changes detected')

    # Interpretation
    print(f'\n--- Health Interpretation ---')
    print(f'  m1 ({result.m1:+.2f}):   ', end='')
    if result.m1 > 0.1:
        print('HR/pressure tending upward')
    elif result.m1 < -0.1:
        print('HR/pressure tending downward')
    else:
        print('Stable (no net trend)')

    print(f'  m2 ({result.m2:+.2f}):   ', end='')
    if result.m2 > 0.7:
        print('High correlation between halves — consistent regime')
    elif result.m2 < 0.3:
        print('Low correlation — possible regime change mid-series')
    else:
        print('Moderate correlation')

    print(f'  m3 ({result.m3:+.2f}):   ', end='')
    if result.m3 > 0.5:
        print('First half more variable than second (settling?)')
    elif result.m3 < -0.5:
        print('Second half more variable than first (destabilising?)')
    else:
        print('Even variability across series')

    print(f'  m4 ({result.m4:+.2f}):   ', end='')
    if result.m4 > 0.1:
        print('Sharp rises > sharp drops')
    elif result.m4 < -0.1:
        print('Sharp drops > sharp rises')
    else:
        print('Symmetric transitions')


if __name__ == '__main__':
    main()
