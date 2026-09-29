"""Embedding store: float16 + npz compression, atomic saves, in-memory matrix for fast matching."""
import os
import threading
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.getenv("FACE_DATA_DIR", Path(__file__).parent / "data"))
DB_FILE = DATA_DIR / "embeddings.npz"


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._vecs: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._mat = np.zeros((0, 128), np.float32)
        self.load()

    def load(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if DB_FILE.exists():
            with np.load(DB_FILE, allow_pickle=False) as z:
                ids, mat = z["ids"].tolist(), z["mat"].astype(np.float32)
            self._vecs = {i: mat[k] for k, i in enumerate(ids)}
        self._rebuild()

    def _rebuild(self):
        self._ids = list(self._vecs)
        self._mat = np.stack([self._vecs[i] for i in self._ids]) if self._ids else np.zeros((0, 128), np.float32)

    def _save(self):
        tmp = DB_FILE.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(self._ids), mat=self._mat.astype(np.float16))
        os.replace(tmp, DB_FILE)

    def put(self, emp_id: str, vec: np.ndarray):
        with self._lock:
            self._vecs[emp_id] = vec.astype(np.float32)
            self._rebuild()
            self._save()

    def delete(self, emp_id: str) -> bool:
        with self._lock:
            if emp_id not in self._vecs:
                return False
            del self._vecs[emp_id]
            self._rebuild()
            self._save()
            return True

    def keep_only(self, keep: list[str]) -> int:
        with self._lock:
            drop = [i for i in self._vecs if i not in set(keep)]
            for i in drop:
                del self._vecs[i]
            if drop:
                self._rebuild()
                self._save()
            return len(drop)

    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self):
        return len(self._ids)

    def match(self, vec: np.ndarray, exclude: str | None = None):
        """Returns (best_id, best_sim, margin_to_second) using cosine similarity."""
        mat, ids = self._mat, self._ids
        if not len(ids):
            return None, -1.0, 0.0
        sims = mat @ vec
        if exclude in ids:
            sims = sims.copy()
            sims[ids.index(exclude)] = -1.0
        order = np.argsort(-sims)
        best = int(order[0])
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        return ids[best], float(sims[best]), float(sims[best]) - second

    def similarity(self, emp_id: str, vec: np.ndarray) -> float:
        v = self._vecs.get(emp_id)
        return float(v @ vec) if v is not None else -1.0


store = Store()
