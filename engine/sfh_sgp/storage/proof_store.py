import json
import uuid
from datetime import datetime, timezone
from typing import Any


class ExperimentResult:
    def __init__(self, experiment: str, passed: bool, metrics: dict,
                 lineage_hash: str = '', result_id: str | None = None):
        self.id = result_id or str(uuid.uuid4())
        self.experiment = experiment
        self.passed = passed
        self.metrics = metrics
        self.lineage_hash = lineage_hash
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'experiment': self.experiment,
            'passed': self.passed,
            'metrics': self.metrics,
            'lineage_hash': self.lineage_hash,
            'created_at': self.created_at.isoformat(),
        }


class ProofStore:
    def __init__(self, path: str = ''):
        self.path = path
        self._records: list[ExperimentResult] = []

    def record(self, result: ExperimentResult):
        self._records.append(result)
        if self.path:
            with open(self.path, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')

    def get_all(self) -> list[ExperimentResult]:
        return list(self._records)

    def query(self, experiment: str = '') -> list[ExperimentResult]:
        if not experiment:
            return self.get_all()
        return [r for r in self._records if r.experiment == experiment]

    def export(self, path: str):
        with open(path, 'w') as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2)

    def __len__(self) -> int:
        return len(self._records)
