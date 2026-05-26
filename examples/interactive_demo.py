"""
Quick interactive demo: embed any signal you type in.

Usage:
    python examples/interactive_demo.py
"""
from sfh_sgp.core.embedding import EmbeddingEngine
from sfh_sgp.core.transforms import TransformEngine
from sfh_sgp.core.guard import ArchitectureGuard
import numpy as np

engine = EmbeddingEngine()
transformer = TransformEngine()
guard = ArchitectureGuard()

print('SFH-SGP Interactive Demo (V2_079 frozen)')
print(f'  Architecture: {guard.FROZEN_ARCH}  Hash: {guard.FROZEN_HASH[:16]}...')
print('  Enter comma-separated numbers, or "quit"')

while True:
    try:
        line = input('\nsignal> ').strip()
        if not line or line.lower() in ('quit', 'exit', 'q'):
            break
        values = np.array([float(v.strip()) for v in line.split(',')], dtype=float)
        if len(values) < 4:
            print('  Need at least 4 data points')
            continue

        result = engine.embed(values, signal_id='interactive')
        print(f'  m1 (ordinal flow):           {result.m1:.6f}')
        print(f'  m2 (half correlation):       {result.m2:.6f}')
        print(f'  m3 (signed compressibility): {result.m3:.6f}')
        print(f'  m4 (amp transition asym):    {result.m4:.6f}')

        # Show transforms
        print('  --- transforms ---')
        for t_name in ['reverse', 'swap', 'replay', 'stitch']:
            tx = transformer.apply(values, t_name)
            tr = engine.embed(tx, transform=t_name)
            delta = tr.vector - result.vector
            print(f'  Δ_{t_name:8s} = [{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}, {delta[3]:+.4f}]')
    except ValueError:
        print('  Invalid numbers. Try: 1, 2, 3, 4, 5, 6, 7, 8')
    except KeyboardInterrupt:
        break

print('Done.')
