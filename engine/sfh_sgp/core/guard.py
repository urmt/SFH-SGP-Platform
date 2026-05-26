import json
import hashlib


FROZEN_ARCH = 'V2_079'
FROZEN_HASH = '9336bf3849d2e891986c9cf33851dab8aed3519968880266c3cba548247b5e4d'
FROZEN_TIMESTAMP = '2026-05-15T21:31:37.037366'
LINEAGE_VALID = True
REWRITE_ALLOWED = False


class HardFreezeViolation(Exception):
    pass


class ArchitectureGuard:
    def __init__(self, metadata: dict | None = None):
        if metadata is None:
            metadata = {
                'rewrite_allowed': REWRITE_ALLOWED,
                'theory_hash': FROZEN_HASH,
                'architecture': FROZEN_ARCH,
                'lineage_valid': LINEAGE_VALID,
            }
        self._validate(metadata)
        self.validated = True

    @classmethod
    def from_path(cls, metadata_path: str):
        with open(metadata_path) as f:
            meta = json.load(f)
        return cls(meta)

    def _validate(self, meta: dict):
        if meta.get('rewrite_allowed', True) is not False:
            raise HardFreezeViolation('rewrite_allowed must be False')
        if meta.get('theory_hash') != FROZEN_HASH:
            raise HardFreezeViolation(
                f'theory_hash mismatch: expected {FROZEN_HASH}, got {meta.get("theory_hash")}'
            )
        if meta.get('architecture') != FROZEN_ARCH:
            raise HardFreezeViolation(
                f'architecture mismatch: expected {FROZEN_ARCH}, got {meta.get("architecture")}'
            )
        if meta.get('lineage_valid') is not True:
            raise HardFreezeViolation('lineage_valid must be True')

    def assert_frozen(self):
        if not self.validated:
            raise HardFreezeViolation('V2_079 integrity compromised')


def compute_theory_hash(theory_json: dict) -> str:
    return hashlib.sha256(
        json.dumps(theory_json, sort_keys=True).encode()
    ).hexdigest()
