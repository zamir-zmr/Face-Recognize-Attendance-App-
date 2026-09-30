"""Embedding store + Firebase sync.
Synced to Firebase (Realtime Database REST + Cloud Storage REST, stdlib only - no new pip deps):
  /face_profiles/<emp>   face embeddings (+ image_path)      /employees/<emp>     employee records
  /attendance_logs/<id>  attendance logs                     /history/<id>        history events
  Storage: face_images/<emp>/profile.jpg                     (face images)
NOT synced: UI state, button actions, anything local to the browser.
Local npz file stays as offline cache; Firebase is the source of truth when reachable.
Env: FIREBASE_SYNC=0 disables Firebase (falls back to Postgres via DATABASE_URL, else file).
     FIREBASE_DB_URL / FIREBASE_BUCKET override config. FIREBASE_AUTH = database secret or ID token (if rules are not open)."""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.getenv("FACE_DATA_DIR", Path(__file__).parent / "data"))
DB_FILE = DATA_DIR / "embeddings.npz"
REFRESH_S = 3.0   # how often to check the remote store for changes made by other workers

FIREBASE_CONFIG = {
    "apiKey": "AIzaSyA0rirztMO13FyXcKYz1aEB1ERYH-HQUbA",
    "authDomain": "sweethouse-e3e49.firebaseapp.com",
    "databaseURL": "https://sweethouse-e3e49-default-rtdb.firebaseio.com",
    "projectId": "sweethouse-e3e49",
    "storageBucket": "sweethouse-e3e49.firebasestorage.app",
    "messagingSenderId": "579322038108",
    "appId": "1:579322038108:web:166b7fdb103080f56d6399",
}


# ---------- Firebase REST client ----------
class Firebase:
    def __init__(self):
        self.enabled = os.getenv("FIREBASE_SYNC", "1") != "0"
        self.db = os.getenv("FIREBASE_DB_URL", FIREBASE_CONFIG["databaseURL"]).rstrip("/")
        self.bucket = os.getenv("FIREBASE_BUCKET", FIREBASE_CONFIG["storageBucket"])
        self.auth = os.getenv("FIREBASE_AUTH", "")

    @staticmethod
    def key(raw: str) -> str:
        """RTDB keys may not contain . $ # [ ] /  -> percent-encode them."""
        return urllib.parse.quote(str(raw), safe="").replace(".", "%2E")

    def _call(self, method, url, data=None, ctype="application/json", timeout=8):
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw else None

    def _db(self, method, path, body=None, **q):
        if self.auth:
            q["auth"] = self.auth
        if q.get("shallow"):
            q["shallow"] = "true"
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        data = json.dumps(body).encode() if body is not None else None
        return self._call(method, "%s/%s.json%s" % (self.db, path.strip("/"), qs), data)

    def get(self, path, **q):
        return self._db("GET", path, **q)

    def put(self, path, body):
        return self._db("PUT", path, body)

    def patch(self, path, body):
        return self._db("PATCH", path, body)

    def push(self, path, body):
        return self._db("POST", path, body)

    def delete(self, path):
        return self._db("DELETE", path)

    # -- Cloud Storage (face images) --
    def _obj(self, path):
        q = ("?alt=media&" if False else "")  # placeholder keeps URL building in one place
        return "https://firebasestorage.googleapis.com/v0/b/%s/o/%s" % (self.bucket, urllib.parse.quote(path, safe=""))

    def upload_image(self, path: str, data: bytes, ctype="image/jpeg"):
        url = "https://firebasestorage.googleapis.com/v0/b/%s/o?name=%s" % (self.bucket, urllib.parse.quote(path, safe=""))
        if self.auth:
            url += "&auth=" + urllib.parse.quote(self.auth)
        return self._call("POST", url, data, ctype, timeout=20)

    def delete_image(self, path: str):
        url = self._obj(path) + (("?auth=" + urllib.parse.quote(self.auth)) if self.auth else "")
        try:
            self._call("DELETE", url)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise

    def touch_meta(self, count: int):
        self.put("face_meta", {"count": count, "updated_at": {".sv": "timestamp"}})


def _enc(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype(np.float16).tobytes()).decode()


def _dec(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)


# ---------- backends ----------
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


class FirebaseBackend:
    """Write-through: local npz cache first (works offline), then Firebase. Loads from Firebase, falls back to cache."""
    name = "firebase"

    def __init__(self, fb: Firebase):
        self.fb = fb
        self.local = FileBackend()
        fb.get("face_meta")          # raises if unreachable / rules deny -> _make_backend falls back

    def load_all(self) -> dict:
        try:
            meta = self.fb.get("face_meta")
            local = self.local.load_all()
            if meta is None and local:                     # first run: migrate existing local embeddings up
                for emp, v in local.items():
                    self._push(emp, v)
                self.fb.touch_meta(len(local))
                return local
            remote = self.fb.get("face_profiles") or {}
            vecs = {p["emp_id"]: _dec(p["vec"]) for p in remote.values() if p.get("vec") and p.get("emp_id")}
            self.local.save_all(vecs)                      # refresh offline cache
            return vecs
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase load failed, using local cache:", e)
            return self.local.load_all()

    def _push(self, emp_id, vec):
        self.fb.patch("face_profiles/" + Firebase.key(emp_id),
                      {"emp_id": emp_id, "vec": _enc(vec), "updated_at": {".sv": "timestamp"}})

    def put(self, emp_id, vec, vecs):
        self.local.put(emp_id, vec, vecs)
        try:
            self._push(emp_id, vec)
            self.fb.touch_meta(len(vecs))
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase put failed (saved locally):", e)

    def delete(self, emp_id, vecs):
        self.local.delete(emp_id, vecs)
        key = Firebase.key(emp_id)
        try:
            self.fb.delete("face_profiles/" + key)
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase profile delete failed:", e)
        try:                      # legacy / differently keyed nodes that still carry this emp_id
            for k, p in (self.fb.get("face_profiles") or {}).items():
                if k != key and isinstance(p, dict) and p.get("emp_id") == emp_id:
                    self.fb.delete("face_profiles/" + k)
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase stray-profile delete failed:", e)
        try:
            self.fb.delete_image("face_images/%s/profile.jpg" % emp_id)
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase image delete failed:", e)
        try:
            self.fb.touch_meta(len(vecs))
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase meta update failed:", e)

    def keep_only(self, keep, vecs):
        self.local.keep_only(keep, vecs)
        try:
            keep_keys = {Firebase.key(i) for i in vecs}
            for k in (self.fb.get("face_profiles", shallow=True) or {}):
                if k not in keep_keys:
                    self.fb.delete("face_profiles/" + k)
            self.fb.touch_meta(len(vecs))
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase prune failed (pruned locally):", e)

    def signature(self):
        m = self.fb.get("face_meta") or {}
        return (m.get("count"), m.get("updated_at"))


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


fb = Firebase()


def _make_backend():
    if fb.enabled:
        try:
            return FirebaseBackend(fb)
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase unavailable, falling back:", e)
            fb.enabled = False
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
        self._put_at: dict[str, float] = {}      # recently enrolled ids (employee record may not be saved yet)
        self._purge_next = 0.0
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
        if self.backend not in ("postgres", "firebase") or time.time() < self._next_check:
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
        self._put_at[emp_id] = time.time()
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

    ORPHAN_GRACE_S = 30 * 60

    def live_employee_ids(self):
        """Ids of employees that exist in Firebase (not soft-deleted). None if unknown -> never purge on doubt."""
        if not fb.enabled:
            return None
        try:
            emps = fb.get("employees")
        except Exception as e:  # noqa: BLE001
            print("[store] employee list unavailable:", e)
            return None
        if not isinstance(emps, dict) or not emps:
            return None
        live = set()
        for k, e in emps.items():
            if isinstance(e, dict) and not e.get("deleted"):
                live.add(str(e.get("id") or urllib.parse.unquote(k)))
        return live or None

    def is_orphan(self, emp_id: str, live=None) -> bool:
        live = live if live is not None else self.live_employee_ids()
        if not live or emp_id in live:
            return False
        return time.time() - self._put_at.get(emp_id, 0.0) > self.ORPHAN_GRACE_S

    def purge_orphans(self, force: bool = False) -> int:
        """Delete face embeddings whose employee no longer exists (deleted employees must never be recognised)."""
        if not force and time.time() < self._purge_next:
            return 0
        self._purge_next = time.time() + 20
        live = self.live_employee_ids()
        if not live:
            return 0
        n = 0
        for emp_id in [i for i in list(self._vecs) if self.is_orphan(i, live)]:
            try:
                if self.delete(emp_id):
                    n += 1
                    print("[store] purged orphan face:", emp_id)
            except Exception as e:  # noqa: BLE001
                print("[store] orphan purge failed:", emp_id, e)
        return n

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

    # ---------- Firebase data sync (durable data only; never UI state) ----------
    def _sync(self, fn, what):
        if not fb.enabled:
            return False
        try:
            fn()
            return True
        except Exception as e:  # noqa: BLE001
            print("[store] Firebase %s failed: %s" % (what, e))
            return False

    def save_face_image(self, emp_id: str, jpeg: bytes) -> str | None:
        """Face image -> Firebase Storage; path is recorded on the face profile. Returns the storage path."""
        path = "face_images/%s/profile.jpg" % emp_id
        ok = self._sync(lambda: (fb.upload_image(path, jpeg),
                                 fb.patch("face_profiles/" + Firebase.key(emp_id), {"image_path": path})), "image upload")
        return path if ok else None

    def upsert_employee(self, emp_id: str, data: dict) -> bool:
        return self._sync(lambda: fb.patch("employees/" + Firebase.key(emp_id), {**data, "id": emp_id}), "employee sync")

    def delete_employee(self, emp_id: str) -> bool:
        return self._sync(lambda: fb.delete("employees/" + Firebase.key(emp_id)), "employee delete")

    def log_attendance(self, emp_id: str, data: dict) -> bool:
        """Stored by date: attendance_logs/<YYYY-MM-DD>/<id> (same layout as the web app -> cheap date-range queries)."""
        now = int(time.time() * 1000)
        rec = {**data, "empId": emp_id, "employee_id": emp_id, "lastUpdated": data.get("lastUpdated") or now}
        day = rec.get("date") or time.strftime("%Y-%m-%d", time.gmtime())
        rec["date"] = day
        lid = Firebase.key(rec.pop("id", None) or "%x%s" % (now, os.urandom(2).hex()))
        return self._sync(lambda: fb.put("attendance_logs/%s/%s" % (day, lid), rec), "attendance log")

    def get_employees(self) -> list:
        return list((fb.get("employees") or {}).values()) if fb.enabled else []

    def get_logs(self, start: str, end: str) -> list:
        if not fb.enabled:
            return []
        by_date = fb.get("attendance_logs", orderBy='"$key"', startAt='"%s"' % start, endAt='"%s"' % end) or {}
        return [{**l, "id": i, "date": d} for d, items in by_date.items() for i, l in (items or {}).items()]

    def dashboard(self) -> dict:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        emps = list((fb.get("employees") or {}).values()) if fb.enabled else []
        todays = [{**l, "id": i} for i, l in ((fb.get("attendance_logs/" + today) or {}).items() if fb.enabled else [])]
        meta = (fb.get("face_meta") or {}) if fb.enabled else {}
        recent = sorted(todays, key=lambda l: l.get("lastUpdated", 0), reverse=True)[:8]
        return {"registered": len(emps), "checked_in": sum(1 for l in todays if l.get("entrance")),
                "checked_out": sum(1 for l in todays if l.get("exit")),
                "face_enrolled": max(sum(1 for e in emps if e.get("descriptor")), int(meta.get("count") or 0)),
                "recent": [{"name": l.get("name"), "id": l.get("empId"), "date": l.get("date"),
                            "time": l.get("exit") or l.get("entrance")} for l in recent]}

    def add_history(self, emp_id: str, event: str, data: dict | None = None) -> bool:
        rec = {**(data or {}), "employee_id": emp_id, "event": event, "ts": int(time.time() * 1000)}
        return self._sync(lambda: fb.push("history", rec), "history")


store = Store()
