from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
from ..core.embedding import EmbeddingEngine, EmbeddingResult
from ..core.transforms import TransformEngine
from ..core.guard import ArchitectureGuard, FROZEN_ARCH
from ..core.geometry import GeometryAnalyser
from ..core.experiments import ExperimentRunner
from ..core.anomaly import AnomalyDetector
from ..io.stream import WebSocketHandler
from ..storage.proof_store import ProofStore, ExperimentResult

app = FastAPI(title='SFH-SGP Platform', version='1.0.0')

guard = ArchitectureGuard()
engine = EmbeddingEngine(guard)
transformer = TransformEngine()
geometry = GeometryAnalyser()
experiments = ExperimentRunner(engine, transformer, geometry)
anomaly = AnomalyDetector()
ws_handler = WebSocketHandler()
proof_store = ProofStore()


class SignalIn(BaseModel):
    values: list[float]
    signal_id: str = ''
    transform: str = 'base'


class EmbeddingOut(BaseModel):
    signal_id: str
    m1: float
    m2: float
    m3: float
    m4: float
    transform: str
    architecture: str


@app.get('/health')
def health():
    return {'status': 'ok', 'architecture': FROZEN_ARCH, 'frozen': guard.validated}


@app.post('/embed', response_model=EmbeddingOut)
def embed_signal(signal: SignalIn):
    x = np.array(signal.values, dtype=float)
    if signal.transform != 'base':
        x = transformer.apply(x, signal.transform)
    result = engine.embed(x, signal_id=signal.signal_id, transform=signal.transform)
    return EmbeddingOut(
        signal_id=result.signal_id,
        m1=result.m1, m2=result.m2,
        m3=result.m3, m4=result.m4,
        transform=result.transform,
        architecture=result.architecture,
    )


@app.post('/embed-many')
def embed_many(signals: list[SignalIn]):
    results = []
    for s in signals:
        x = np.array(s.values, dtype=float)
        if s.transform != 'base':
            x = transformer.apply(x, s.transform)
        result = engine.embed(x, signal_id=s.signal_id, transform=s.transform)
        results.append(result.to_dict())
    return {'embeddings': results}


@app.post('/analyse')
def analyse(signals: list[SignalIn]):
    embeddings = []
    for s in signals:
        x = np.array(s.values, dtype=float)
        if s.transform != 'base':
            x = transformer.apply(x, s.transform)
        result = engine.embed(x, signal_id=s.signal_id, transform=s.transform)
        embeddings.append(result.vector)
    geo = geometry.analyse(np.array(embeddings))
    return geo


@app.post('/experiment/run-all')
def run_all_experiments():
    import uuid
    from ..io.synthetic import random_walk, regime_switch
    n_signals = 50
    rw_signals = [random_walk(seed=i) for i in range(n_signals)]
    rs_signals = [regime_switch(seed=i) for i in range(n_signals)]
    all_signals = rw_signals + rs_signals
    labels = [0] * n_signals + [1] * n_signals
    rw_embs = np.array([r.vector for r in engine.embed_many(rw_signals)])
    rs_embs = np.array([r.vector for r in engine.embed_many(rs_signals)])
    replay_signal = random_walk(seed=42)
    signal_set = {
        'classification_signals': all_signals,
        'classification_labels': labels,
        'replay_signal': replay_signal,
    }
    results = experiments.run_all(signal_set)
    f004 = experiments.run_f004_rw_trend_overlap(rw_embs, rs_embs)
    results.append(f004)
    stored = []
    for r in results:
        er = ExperimentResult(
            experiment=r['experiment'],
            passed=r['passed'],
            metrics=r['metrics'],
        )
        proof_store.record(er)
        stored.append(er.to_dict())
    return {'experiments': stored}


@app.get('/proofs')
def list_proofs(experiment: str = ''):
    return {'records': [r.to_dict() for r in proof_store.query(experiment)]}


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_handler.register(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f'ack: {data}')
    except WebSocketDisconnect:
        ws_handler.unregister(websocket)
