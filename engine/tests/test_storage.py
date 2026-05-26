import numpy as np
from sfh_sgp.storage.proof_store import ProofStore, ExperimentResult


def test_proof_store_record():
    store = ProofStore()
    r = ExperimentResult(experiment='F001_test', passed=True, metrics={'acc': 0.9})
    store.record(r)
    assert len(store) == 1


def test_proof_store_query():
    store = ProofStore()
    store.record(ExperimentResult('F001', True, {'a': 1}))
    store.record(ExperimentResult('F002', False, {'b': 2}))
    assert len(store.query('F001')) == 1
    assert len(store.query()) == 2


def test_proof_store_to_dict():
    r = ExperimentResult(
        experiment='F001', passed=True, metrics={'acc': 0.95},
        result_id='abc-123'
    )
    d = r.to_dict()
    assert d['id'] == 'abc-123'
    assert d['experiment'] == 'F001'
    assert d['passed'] is True
