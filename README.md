# SFH-SGP Platform

**Signal-Feature-Histogram / Signal Geometry Platform** — V2_079 (FROZEN)

Cross-platform scientific computing framework for time-series analysis via canonical four-dimensional embeddings. Architecture V2_079 is analytically validated and frozen — no modifications permitted without hash break.

## Quick Start

```bash
# Install the engine
cd engine && pip install -e .

# Verify installation
sfh-sgp experiment
```

Expected output (experiments should PASS):

```
F001_embeddings_separable                     [PASS]  {'1NN_accuracy': ..., 'chance': 0.2}
F005_manifold_effectively_1D                  [PASS]  {'pc1_variance': ..., 'dim95': 1}
F003_replay_robustness_universal              [PASS]  {'replay_accuracy': ..., 'm1_diff': ...}
F004_rw_trend_irreducible_overlap             [PASS]  {'sep_ratio': ...}
```

---

## 1. API Server

```bash
# Start the REST + WebSocket server
sfh-sgp serve

# Or with auto-reload during development
sfh-sgp serve --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/health` | GET | Returns architecture status and freeze state |
| `/embed` | POST | Compute V2_079 embedding for one signal |
| `/embed-many` | POST | Batch embed multiple signals |
| `/analyse` | POST | Full geometry analysis on a set of embeddings |
| `/experiment/run-all` | POST | Run all 4 empirical findings against synthetic data |
| `/proofs` | GET | Browse experiment results from the proof store |
| `/ws` | WebSocket | Real-time signal streaming |

### Example: embed a signal via curl

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"values": [100, 102, 101, 103, 105, 104, 106, 108], "signal_id": "test"}'
```

Response:

```json
{
  "signal_id": "test",
  "m1": 0.428571,
  "m2": -0.542857,
  "m3": 0.123456,
  "m4": 0.234567,
  "transform": "base",
  "architecture": "V2_079"
}
```

---

## 2. Python Library

```python
import numpy as np
from sfh_sgp.core.embedding import EmbeddingEngine
from sfh_sgp.core.transforms import TransformEngine
from sfh_sgp.core.geometry import GeometryAnalyser
from sfh_sgp.core.guard import ArchitectureGuard

# The architecture guard validates the frozen theory hash
guard = ArchitectureGuard()
engine = EmbeddingEngine(guard)
transformer = TransformEngine()

# Embed a signal
signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
result = engine.embed(signal, signal_id='my_signal')
print(f'm1={result.m1:.4f}  m2={result.m2:.4f}  m3={result.m3:.4f}  m4={result.m4:.4f}')
# m1=1.0000  m2=1.0000  m3=-1.3863  m4=-0.0000

# Apply transforms and re-embed
reversed_signal = transformer.apply(signal, 'reverse')
replay_signal   = transformer.apply(signal, 'replay')
swap_signal     = transformer.apply(signal, 'swap')
stitch_signal   = transformer.apply(signal, 'stitch')

# Geometry analysis (needs multiple signals)
geo = GeometryAnalyser()
signals = [np.sin(np.linspace(0, i*0.5, 100)) for i in range(20)]
embeddings = np.array([engine.embed(s).vector for s in signals])
result = geo.analyse(embeddings)
print(f'PC1 variance: {result["pc1"]}  Dim(95%): {result["dim95"]}')
```

---

## 3. Analyse Stock / Crypto Data

Use the included market analysis script:

```bash
# Analyse Apple stock (log returns)
python examples/analyse_market.py --symbol AAPL --days 365

# Analyse Bitcoin
python examples/analyse_market.py --symbol BTC-USD --days 180

# Analyse S&P 500
python examples/analyse_market.py --symbol ^SPX --days 500

# Raw prices instead of log returns
python examples/analyse_market.py --symbol AAPL --mode prices --days 365
```

### What the metrics tell you about markets

| Metric | Value | Meaning |
|--------|-------|---------|
| **m1** (ordinal flow) | `> 0.3` | More up-days than down-days (bullish bias) |
| | `< -0.3` | More down-days than up-days (bearish bias) |
| | `~ 0` | No directional bias (choppy / sideways) |
| **m2** (half correlation) | `> 0.7` | First half similar to second half (trend persistence) |
| | `< 0.3` | Regime change mid-series |
| | `< -0.3` | Anti-correlated halves (reversal pattern) |
| **m3** (compressibility) | `> 0.5` | First half more volatile / random |
| | `< -0.5` | Second half more volatile / random |
| | `~ 0` | Even volatility across series |
| **m4** (amplitude asym) | `> 0.2` | Big up-moves bigger than big down-moves |
| | `< -0.2` | Big down-moves bigger than big up-moves |
| | `~ 0` | Symmetric amplitude transitions |

### Transform analysis

The script also shows delta-vectors for each transform. A **large Δ** under a given transform means that transform significantly changes the signal's geometric signature. For example, if `Δ_replay` is large on m2, it suggests the half-correlation is sensitive to signal duplication — useful for detecting whether a pattern is structural or coincidental.

---

## 4. Analyse Health Data

```bash
# Synthetic heart-rate demo (24h at 1-min resolution)
python examples/analyse_health.py

# Synthetic blood pressure (30 days)
python examples/analyse_health.py --type blood_pressure

# Your own CSV export (last column = numeric readings)
python examples/analyse_health.py --csv my_heart_rate.csv
```

### What the metrics tell you about health signals

| Metric | Value | Meaning |
|--------|-------|---------|
| **m1** | `> 0.1` | Readings trending upward (worsening BP? improving HR?) |
| | `< -0.1` | Readings trending downward |
| | `~ 0` | Stable readings |
| **m2** | `> 0.7` | Consistent physiological regime |
| | `< 0.3` | Possible regime change (recovery? onset?) |
| **m3** | `> 0.5` | First half more erratic (initial stress?) |
| | `< -0.5` | Second half more erratic (destabilising?) |
| **m4** | `> 0.1` | Sharp rises > sharp drops (overshoots) |
| | `< -0.1` | Sharp drops > sharp rises (undershoots) |

The anomaly detector also runs automatically when you have enough data, checking for regime shifts between windows.

---

## 5. Interactive Demo

```bash
python examples/interactive_demo.py
```

Type comma-separated numbers and see the embedding plus all transform deltas instantly.

---

## 6. CLI Reference

```bash
sfh-sgp serve         # Start API server
sfh-sgp embed <file>  # Embed a CSV signal
sfh-sgp analyse <file> # Geometry analysis on CSV
sfh-sgp experiment     # Run all experiments
```

---

## 7. Running Tests

```bash
cd engine
pytest tests/ -v
```

All 34 tests cover:
- **Metric correctness** — each m1-m4 function tested edge-to-edge
- **ArchitectureGuard** — tamper detection for hash/arch/lineage/rewrite
- **Transforms** — reverse, replay, swap, stitch produce correct shapes
- **Geometry** — PCA, curvature, neighbour purity, Euclidean correlation
- **Experiments** — F001-F005 run on synthetic data
- **Falsification protocol** — verifies the system correctly FAILS on tampered metadata
- **ProofStore** — append-only record and query
- **IO** — synthetic generators, CSV import/export, WebSocket handler

---

## Project Structure

```
engine/                        # Python backend
├── sfh_sgp/
│   ├── core/
│   │   ├── embedding.py       # V2_079 metric functions (frozen)
│   │   ├── guard.py           # ArchitectureGuard + freeze constants
│   │   ├── transforms.py      # reverse/replay/swap/stitch
│   │   ├── geometry.py        # PCA + curvature analysis
│   │   ├── anomaly.py         # LDA regime detector
│   │   └── experiments.py     # F001-F005 runners
│   ├── io/
│   │   ├── ingestion.py       # CSV/array signal loading
│   │   ├── synthetic.py       # random_walk, regime_switch, etc.
│   │   ├── stream.py          # WebSocket handler
│   │   └── export.py          # JSON/CSV export
│   ├── storage/
│   │   └── proof_store.py     # Append-only experiment log
│   ├── plugins/               # Plugin bus (extensible)
│   └── api/
│       └── main.py            # FastAPI server
├── metadata/
│   └── V2_079_METADATA.json   # Frozen (do not modify)
└── tests/                     # 34 tests
examples/
├── analyse_market.py          # Stock/crypto analysis
├── analyse_health.py          # Health data analysis
└── interactive_demo.py        # REPL-style demo
```

---

## Architecture Integrity

```
FROZEN_ARCH       = V2_079
FROZEN_HASH       = 9336bf3849d2e891986c9cf33851dab8aed3519968880266c3cba548247b5e4d
FROZEN_TIMESTAMP  = 2026-05-15T21:31:37.037366
LINEAGE_VALID     = True
REWRITE_ALLOWED   = False
```

All four conditions are verified at engine init. Any tampering causes `HardFreezeViolation`.

---

## Falsification Protocol

The system must **correctly fail** on adversarial data. Run `pytest tests/test_falsification.py` to verify:

| Test | What it checks |
|------|---------------|
| `test_freeze_violation_on_tampered_metadata` | Hash mismatch → HardFreezeViolation |
| `test_freeze_violation_rewrite_allowed` | rewrite_allowed=True → HardFreezeViolation |
| `test_freeze_violation_hash_mismatch` | Wrong theory_hash → HardFreezeViolation |
| `test_freeze_violation_architecture_mismatch` | Wrong arch → HardFreezeViolation |
| `test_freeze_violation_lineage_invalid` | lineage_valid=False → HardFreezeViolation |
| `test_embedding_deterministic` | Same input → identical output (bit-for-bit) |
