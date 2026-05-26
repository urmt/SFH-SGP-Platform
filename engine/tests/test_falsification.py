import json, pytest
from sfh_sgp.core.guard import ArchitectureGuard, HardFreezeViolation, FROZEN_HASH, FROZEN_ARCH


TAMPERED_METADATA = {
    'timestamp': '2026-05-15T21:31:37.037366',
    'status': 'FROZEN',
    'architecture': 'V2_079',
    'lineage_valid': False,
    'rewrite_allowed': True,
    'theory_hash': '0000000000000000000000000000000000000000000000000000000000000000',
}


def test_freeze_violation_on_tampered_metadata():
    with pytest.raises(HardFreezeViolation):
        ArchitectureGuard(TAMPERED_METADATA)


def test_freeze_violation_rewrite_allowed():
    meta = dict(TAMPERED_METADATA)
    meta['rewrite_allowed'] = True
    meta['theory_hash'] = FROZEN_HASH
    meta['lineage_valid'] = True
    with pytest.raises(HardFreezeViolation, match='rewrite_allowed'):
        ArchitectureGuard(meta)


def test_freeze_violation_hash_mismatch():
    meta = dict(TAMPERED_METADATA)
    meta['rewrite_allowed'] = False
    meta['lineage_valid'] = True
    with pytest.raises(HardFreezeViolation, match='theory_hash'):
        ArchitectureGuard(meta)


def test_freeze_violation_architecture_mismatch():
    meta = dict(TAMPERED_METADATA)
    meta['rewrite_allowed'] = False
    meta['theory_hash'] = FROZEN_HASH
    meta['architecture'] = 'V2_OLD'
    meta['lineage_valid'] = True
    with pytest.raises(HardFreezeViolation, match='architecture'):
        ArchitectureGuard(meta)


def test_freeze_violation_lineage_invalid():
    meta = dict(TAMPERED_METADATA)
    meta['rewrite_allowed'] = False
    meta['theory_hash'] = FROZEN_HASH
    meta['lineage_valid'] = False
    with pytest.raises(HardFreezeViolation, match='lineage_valid'):
        ArchitectureGuard(meta)


def test_guard_from_path(tmp_path):
    valid = {
        'rewrite_allowed': False,
        'theory_hash': FROZEN_HASH,
        'architecture': FROZEN_ARCH,
        'lineage_valid': True,
    }
    p = tmp_path / 'metadata.json'
    p.write_text(json.dumps(valid))
    guard = ArchitectureGuard.from_path(str(p))
    guard.assert_frozen()


def test_embedding_deterministic():
    import numpy as np
    from sfh_sgp.core.embedding import EmbeddingEngine
    engine = EmbeddingEngine()
    x = np.sin(np.linspace(0, 4 * np.pi, 100))
    r1 = engine.embed(x, signal_id='ref')
    r2 = engine.embed(x, signal_id='ref')
    assert r1.m1 == pytest.approx(r2.m1, abs=1e-12)
    assert r1.m2 == pytest.approx(r2.m2, abs=1e-12)
    assert r1.m3 == pytest.approx(r2.m3, abs=1e-12)
    assert r1.m4 == pytest.approx(r2.m4, abs=1e-12)
