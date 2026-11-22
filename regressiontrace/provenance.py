"""Versioned data provenance in DuckDB.

Every training row belongs to exactly one ingestion batch (with a content hash),
and every model is trained on an immutable *dataset version*: a set of batch ids
plus an optional repair (label overrides / row weights). Version ids are content
addressed, so the same inputs always produce the same id.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY, source VARCHAR, ingested_at TIMESTAMP,
    row_count INTEGER, content_sha256 VARCHAR, schema_hash VARCHAR);
CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id VARCHAR PRIMARY KEY, parent_version VARCHAR, batch_ids VARCHAR,
    repair VARCHAR, created_at TIMESTAMP, note VARCHAR);
CREATE TABLE IF NOT EXISTS label_overrides (version_id VARCHAR, row_id BIGINT, label TINYINT);
CREATE TABLE IF NOT EXISTS row_weights (version_id VARCHAR, row_id BIGINT, weight DOUBLE);
-- Ground truth for demos/benchmarks only. Diagnosis code never reads this table.
CREATE TABLE IF NOT EXISTS fault_log (batch_id INTEGER, fault VARCHAR, severity VARCHAR, description VARCHAR);
"""


def content_hash(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(h).hexdigest()


def schema_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(json.dumps([[c, str(t)] for c, t in df.dtypes.items()]).encode()).hexdigest()[:16]


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Store:
    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            for stmt in SCHEMA.split(";"):
                if "CREATE" in stmt:
                    self._x(stmt)

    def _x(self, sql: str, params=None):
        """Execute and fully consume the result (DuckDB >= 2 requires this before the next call)."""
        return self.con.execute(sql, params or []).fetchall()

    def _one(self, sql: str, params=None):
        rows = self._x(sql, params)
        return rows[0] if rows else None

    def close(self):
        self.con.close()

    # ---- ingestion -------------------------------------------------------
    def ingest_batch(self, batch_id: int, X: pd.DataFrame, y: np.ndarray, source: str) -> str:
        df = X.copy()
        df.insert(0, "label", np.asarray(y, dtype=np.int8))
        df.insert(0, "batch_id", np.int32(batch_id))
        sha = content_hash(df.drop(columns=["batch_id"]))
        exists = self._one("SELECT 1 FROM information_schema.tables WHERE table_name='rows'")
        start = 0
        if exists:
            start = self._one("SELECT COALESCE(MAX(row_id) + 1, 0) FROM rows")[0]
        df.insert(0, "row_id", np.arange(start, start + len(df), dtype=np.int64))
        self.con.register("_df", df)
        if exists:
            self._x("INSERT INTO rows SELECT * FROM _df")
        else:
            self._x("CREATE TABLE rows AS SELECT * FROM _df")
        self.con.unregister("_df")
        self._x("INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?)",
                [batch_id, source, _now(), len(df), sha, schema_hash(X)])
        return sha

    def set_holdout(self, X: pd.DataFrame, y: np.ndarray):
        df = X.copy()
        df["label"] = np.asarray(y, dtype=np.int8)
        self.con.register("_h", df)
        self._x("CREATE OR REPLACE TABLE holdout AS SELECT * FROM _h")
        self.con.unregister("_h")

    def log_fault(self, batch_id: int, fault: str, severity: str, description: str):
        self._x("INSERT INTO fault_log VALUES (?, ?, ?, ?)", [batch_id, fault, severity, description])

    # ---- versions --------------------------------------------------------
    def create_version(self, batch_ids, parent: str | None = None, repair: dict | None = None,
                       label_overrides: dict[int, int] | None = None,
                       weights: dict[int, float] | None = None, note: str = "") -> str:
        batch_ids = sorted(int(b) for b in batch_ids)
        payload = json.dumps({
            "batches": batch_ids, "parent": parent, "repair": repair,
            "labels": sorted((label_overrides or {}).items()),
            "weights": sorted((weights or {}).items()),
        }, sort_keys=True, default=str)
        vid = "v-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
        if self._one("SELECT 1 FROM dataset_versions WHERE version_id=?", [vid]):
            return vid
        self._x("INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?)",
                [vid, parent, json.dumps(batch_ids), json.dumps(repair), _now(), note])
        if label_overrides:
            self.con.executemany("INSERT INTO label_overrides VALUES (?, ?, ?)",
                                 [(vid, int(r), int(lab)) for r, lab in label_overrides.items()])
        if weights:
            self.con.executemany("INSERT INTO row_weights VALUES (?, ?, ?)",
                                 [(vid, int(r), float(w)) for r, w in weights.items()])
        return vid

    def get_version(self, vid: str) -> dict:
        df = self.con.execute("SELECT * FROM dataset_versions WHERE version_id=?", [vid]).df()
        if df.empty:
            raise KeyError(vid)
        out = df.iloc[0].to_dict()
        out["batch_ids"] = json.loads(out["batch_ids"])
        out["repair"] = json.loads(out["repair"]) if out["repair"] else None
        return out

    def list_versions(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT version_id, parent_version, json_array_length(batch_ids) AS n_batches, repair, created_at, note "
            "FROM dataset_versions ORDER BY created_at").df()

    def load_version(self, vid: str):
        """Return (X, y, batch, weight, row_id) for a dataset version."""
        v = self.get_version(vid)
        df = self.con.execute(
            "SELECT r.*, o.label AS _ovr, w.weight AS _w FROM rows r "
            "LEFT JOIN label_overrides o ON o.row_id = r.row_id AND o.version_id = ? "
            "LEFT JOIN row_weights w ON w.row_id = r.row_id AND w.version_id = ? "
            "WHERE list_contains(?::INTEGER[], r.batch_id) ORDER BY r.row_id",
            [vid, vid, v["batch_ids"]]).df()
        y = np.where(df["_ovr"].isna(), df["label"], df["_ovr"]).astype(int)
        w = df["_w"].fillna(1.0).to_numpy()
        X = df.drop(columns=["row_id", "batch_id", "label", "_ovr", "_w"])
        return X, y, df["batch_id"].to_numpy(), w, df["row_id"].to_numpy()

    def load_holdout(self):
        df = self.con.execute("SELECT * FROM holdout").df()
        return df.drop(columns=["label"]), df["label"].to_numpy().astype(int)

    # ---- inspection ------------------------------------------------------
    def list_batches(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM batches ORDER BY batch_id").df()

    def get_batch(self, batch_id: int) -> dict:
        df = self.con.execute("SELECT * FROM batches WHERE batch_id=?", [batch_id]).df()
        if df.empty:
            raise KeyError(batch_id)
        out = {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[0].to_dict().items()}
        out["versions"] = [row[0] for row in self.con.execute(
            "SELECT version_id FROM dataset_versions WHERE list_contains(batch_ids::INTEGER[], ?)",
            [batch_id]).fetchall()]
        out["ingested_at"] = str(out["ingested_at"])
        return out

    def verify_batch(self, batch_id: int) -> bool:
        """Recompute the content hash of a stored batch and compare with the recorded one."""
        df = self.con.execute("SELECT * EXCLUDE (row_id, batch_id) FROM rows WHERE batch_id=? ORDER BY row_id",
                              [batch_id]).df()
        df = df.astype({"label": np.int8})
        return content_hash(df) == self.get_batch(batch_id)["content_sha256"]

    def fault_log(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM fault_log ORDER BY batch_id").df()
