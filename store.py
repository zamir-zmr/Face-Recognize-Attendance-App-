"""Embedding store. Persistence: Postgres (DATABASE_URL) or compressed npz file. Reads use an atomic snapshot (no torn ids/matrix)."""
import os
import threading
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.getenv("FACE_DATA_DIR", Path(__file__).parent / "data"))
DB_FILE = DATA_DIR / "embeddings.npz"
REFRESH_S = 3.0   # multi-worker: how often to check Postgres for changes made by other workers


class FileBackend:
    name = "file"

    def load_all(self) -> dict:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not DB_FILE.exists():
            return {}
        with np.load(DB_FILE, allow_pickle=False) as z:
            ids, mat = z["ids"].tolist(), z["mat"].astype(np.float32)
        return {i: mat[k] for k, i in enumerate(ids)}

    def save_all(self, vecs: dict):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ids = list(vecs)
        mat = np.stack([vecs[i] for i in ids]).astype(np.float16) if ids else np.zeros((0, 128), np.float16)
        tmp = DB_FILE.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(ids), mat=mat)
        os.replace(tmp, DB_FILE)

    def put(self, emp_id, vec, vecs):
        self.save_all(vecs)

    def delete(self, emp_id, vecs):
        self.save_all(vecs)

    def keep_only(self, keep, vecs):
        self.save_all(vecs)

    def signature(self):
        return None


class PgBackend:
    name = "postgres"

    def __init__(self, url: str):
        import psycopg2
        self._pg = psycopg2
        self.url = url.replace("postgres://", "postgresql://", 1)
        c = self._conn()
        try:
            with c, c.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS face_embeddings (
                    emp_id text PRIMARY KEY, vec bytea NOT NULL, updated_at timestamptz NOT NULL DEFAULT now())""")
        finally:
            c.close()

    def _run(self, sql, params=None, fetch=False):
        c = self._conn()
        try:
            with c, c.cursor() as cur:      # `with c` commits/rolls back the transaction
                cur.execute(sql, params)
                return cur.fetchall() if fetch else None
        finally:
            c.close()

    def _conn(self):
        return self._pg.connect(self.url, connect_timeout=5)

    def load_all(self) -> dict:
        rows = self._run("SELECT emp_id, vec FROM face_embeddings", fetch=True)
        return {i: np.frombuffer(bytes(b), np.float16).astype(np.float32) for i, b in rows}

    def put(self, emp_id, vec, vecs):
        self._run("""INSERT INTO face_embeddings (emp_id, vec) VALUES (%s, %s)
            ON CONFLICT (emp_id) DO UPDATE SET vec = EXCLUDED.vec, updated_at = now()""",
                  (emp_id, self._pg.Binary(vec.astype(np.float16).tobytes())))

    def delete(self, emp_id, vecs):
        self._run("DELETE FROM face_embeddings WHERE emp_id = %s", (emp_id,))

    def keep_only(self, keep, vecs):
        self._run("DELETE FROM face_embeddings WHERE NOT (emp_id = ANY(%s))", (list(keep),))

    def signature(self):
        return tuple(self._run(
            "SELECT count(*), coalesce(extract(epoch FROM max(updated_at)), 0) FROM face_embeddings", fetch=True)[0])


def _make_backend():
    url = os.getenv("DATABASE_URL")
    if url:
        try:
            return PgBackend(url)
        except Exception as e:  # noqa: BLE001
            print("[store] Postgres unavailable, falling back to file:", e)
    return FileBackend()


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._be = _make_backend()
        self.backend = self._be.name
        self._next_check = 0.0
        self._sig = None
        self._vecs: dict[str, np.ndarray] = {}
        self._snap = ([], np.zeros((0, 128), np.float32))   # (ids, matrix) replaced atomically
        self._reload()

    def _reload(self):
        self._vecs = self._be.load_all()
        self._sig = self._be.signature()
        self._rebuild()

    def _rebuild(self):
        ids = list(self._vecs)
        mat = np.stack([self._vecs[i] for i in ids]).astype(np.float32) if ids else np.zeros((0, 128), np.float32)
        self._snap = (ids, mat)     # single assignment -> readers never see mismatched ids/mat

    def _maybe_refresh(self):
        if self.backend != "postgres" or time.time() < self._next_check:
            return
        with self._lock:
            if time.time() < self._next_check:
                return
            self._next_check = time.time() + REFRESH_S
            try:
                sig = self._be.signature()
                if sig != self._sig:
                    self._reload()
            except Exception as e:  # noqa: BLE001
                print("[store] refresh failed:", e)

    def put(self, emp_id: str, vec: np.ndarray):
        with self._lock:
            self._be.put(emp_id, vec, {**self._vecs, emp_id: vec.astype(np.float32)})   # persist first
            self._vecs[emp_id] = vec.astype(np.float32)
            self._sig = self._be.signature()
            self._rebuild()

    def delete(self, emp_id: str) -> bool:
        with self._lock:
            if emp_id not in self._vecs:
                return False
            rest = {k: v for k, v in self._vecs.items() if k != emp_id}
            self._be.delete(emp_id, rest)
            self._vecs = rest
            self._sig = self._be.signature()
            self._rebuild()
            return True

    def keep_only(self, keep: list[str]) -> int:
        with self._lock:
            keep_set = set(keep)
            rest = {k: v for k, v in self._vecs.items() if k in keep_set}
            dropped = len(self._vecs) - len(rest)
            if dropped:
                self._be.keep_only(keep, rest)
                self._vecs = rest
                self._sig = self._be.signature()
                self._rebuild()
            return dropped

    def ids(self) -> list[str]:
        self._maybe_refresh()
        return list(self._snap[0])

    def __len__(self):
        self._maybe_refresh()
        return len(self._snap[0])

    def match(self, vec: np.ndarray, exclude: str | None = None):
        """Returns (best_id, best_sim, margin_to_second) using cosine similarity."""
        self._maybe_refresh()
        ids, mat = self._snap
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
        ids, mat = self._snap
        return float(mat[ids.index(emp_id)] @ vec) if emp_id in ids else -1.0


store = Store()
