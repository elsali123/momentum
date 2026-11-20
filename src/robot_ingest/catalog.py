"""Catalog: one row per episode, camera and robot, kept current as episodes are committed.

It exists in two forms:
  catalog.sqlite   the live index. The pipeline writes an episode's rows right after committing its
                   bundle, so a run never rescans the store, and the viewer sees each episode as soon
                   as it lands. (Postgres in production; SQLite is the zero-setup local equivalent.)
  *.parquet        an analytics snapshot of the same tables, exported at the end of each run, which
                   DuckDB (`connect`, `robot-ingest query`) and training jobs read.

The bundles stay the source of truth. The write order keeps the catalog from ever pointing at files
that aren't there: rows are added after a bundle is committed and removed before a bundle is deleted.
A crash in between leaves at worst a stored episode missing from the catalog, which the next run that
reaches it, or a full rebuild (`robot-ingest catalog`), puts back.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from robot_ingest.config import Config, RobotSpec
from robot_ingest.schema import CAMERAS, EPISODES, ROBOTS
from robot_ingest.store import Store

# table -> (schema, primary key)
TABLES: dict[str, tuple[pa.Schema, tuple[str, ...]]] = {
    "episodes": (EPISODES, ("episode_id",)),
    "cameras": (CAMERAS, ("episode_id", "camera_id")),
    "robots": (ROBOTS, ("robot_id",)),
}
_TYPES = {f.name: f.type for schema, _ in TABLES.values() for f in schema}  # a name shared by tables has one type


class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as con:
            con.execute("PRAGMA journal_mode=WAL")  # readers (the viewer) and the writer don't block each other
            for table, (schema, key) in TABLES.items():
                columns = ", ".join(f"{f.name} {_sql_type(f.type)}{'' if f.nullable else ' NOT NULL'}" for f in schema)
                con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns}, PRIMARY KEY ({', '.join(key)}))")

    # ----------------------------------------------------------------------------- writes

    def put_episode(self, bundle: dict) -> None:
        """Add or replace one episode and its cameras, in one transaction."""
        check_bundle(bundle)
        with self._transaction() as con:
            _delete_episode(con, bundle["episode"]["episode_id"])
            _insert(con, "episodes", [bundle["episode"]])
            _insert(con, "cameras", bundle["cameras"])

    def remove_episode(self, episode_id: str) -> None:
        with self._transaction() as con:
            _delete_episode(con, episode_id)

    def put_robots(self, robots: Iterable[RobotSpec]) -> None:
        rows = [robot_row(r) for r in robots]
        with self._transaction() as con:
            con.executemany("DELETE FROM robots WHERE robot_id = ?", [(r["robot_id"],) for r in rows])
            _insert(con, "robots", rows)

    def rebuild(self, store: Store, robots: Iterable[RobotSpec]) -> dict[str, int]:
        """Recreate every row from the bundles: recovery, or a store written before the catalog existed."""
        bundles = list(store.iter_bundles())
        for bundle in bundles:
            check_bundle(bundle)
        rows = [robot_row(r) for r in robots]
        with self._transaction() as con:
            for table in TABLES:
                con.execute(f"DELETE FROM {table}")
            _insert(con, "episodes", [b["episode"] for b in bundles])
            _insert(con, "cameras", [c for b in bundles for c in b["cameras"]])
            _insert(con, "robots", rows)
        return {"episodes": len(bundles), "cameras": sum(len(b["cameras"]) for b in bundles), "robots": len(rows)}

    # ----------------------------------------------------------------------------- reads

    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        """Rows as dicts, with list and boolean columns decoded from their SQLite storage."""
        con = sqlite3.connect(self.path, timeout=30)
        try:
            cursor = con.execute(sql, tuple(params))
            names = [d[0] for d in cursor.description]
            return [{n: _decode(n, v) for n, v in zip(names, row)} for row in cursor.fetchall()]
        finally:
            con.close()

    def has(self, episode: dict) -> bool:
        """The catalog holds exactly this version of the episode."""
        return bool(self.query("SELECT 1 FROM episodes WHERE episode_id = ? AND ingested_at = ?",
                               (episode["episode_id"], episode["ingested_at"])))

    def export_parquet(self, root: Path) -> dict[str, int]:
        """Write the analytics snapshot: episodes/cameras/robots.parquet, each file replaced atomically."""
        counts = {}
        for table, (schema, key) in TABLES.items():
            rows = self.query(f"SELECT * FROM {table} ORDER BY {', '.join(key)}")
            for row in rows:
                if "ingested_at" in row:
                    row["ingested_at"] = datetime.fromisoformat(row["ingested_at"])
            tmp = root / f".{table}.parquet.tmp"
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
            os.replace(tmp, root / f"{table}.parquet")
            counts[table] = len(rows)
        return counts

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)  # waits for another writer instead of failing
        try:
            with con:  # commit on success, roll back on error
                yield con
        finally:
            con.close()


def build_catalog(store: Store, config: Config) -> dict[str, int]:
    """Rebuild the catalog from the bundles and export the Parquet snapshot (`robot-ingest catalog`)."""
    catalog = Catalog(store.catalog_path)
    counts = catalog.rebuild(store, config.robots.values())
    catalog.export_parquet(store.root)
    return counts


def check_bundle(bundle: dict) -> None:
    """Raise if a bundle's rows don't match the schema: a bug in a mapper or the pipeline, not bad data.

    The pipeline calls this before committing, so a mismatch fails that one episode instead of the catalog.
    """
    _check_keys("episodes", [bundle["episode"]])
    _check_keys("cameras", bundle["cameras"])


def robot_row(robot: RobotSpec) -> dict:
    return {**asdict(robot), "joint_names": list(robot.joint_names), "num_joints": robot.num_joints}


def connect(store_root: str | Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with views over the Parquet snapshot and the frames. Nothing is copied.

        con = connect("data/store")
        con.sql("SELECT robot_id, count(*), sum(duration_s) FROM episodes GROUP BY ALL").show()
    """
    root = Path(store_root).resolve()
    con = duckdb.connect()
    for table in TABLES:
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{root / table}.parquet')")
    con.execute(f"CREATE VIEW frames AS SELECT * FROM read_parquet('{root}/episodes/*/frames.parquet')")
    return con


# --------------------------------------------------------------------------------- helpers

def _check_keys(table: str, rows: list[dict]) -> None:
    names = set(TABLES[table][0].names)
    for row in rows:
        if set(row) != names:
            raise ValueError(f"{table}: row keys differ from schema: "
                             f"missing {names - set(row)}, unexpected {set(row) - names}")


def _sql_type(t: pa.DataType) -> str:
    if pa.types.is_integer(t) or pa.types.is_boolean(t):
        return "INTEGER"
    if pa.types.is_floating(t):
        return "REAL"
    return "TEXT"  # strings, timestamps (ISO 8601) and lists (JSON)


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_fixed_size_list(t)


def _encode(name: str, value: object) -> object:
    t = _TYPES[name]
    if value is None:
        return None
    if _is_list(t) or isinstance(value, dict):  # dict: source_extras, a JSON object stored as a string column
        return json.dumps(value, sort_keys=True)
    if pa.types.is_boolean(t):
        return int(value)
    return value


def _decode(name: str, value: object) -> object:
    t = _TYPES.get(name)  # computed columns (e.g. counts) pass through
    if value is None or t is None:
        return value
    if _is_list(t):
        return json.loads(value)
    if pa.types.is_boolean(t):
        return bool(value)
    return value


def _insert(con: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    names = TABLES[table][0].names
    con.executemany(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                    [[_encode(n, row[n]) for n in names] for row in rows])


def _delete_episode(con: sqlite3.Connection, episode_id: str) -> None:
    con.execute("DELETE FROM cameras WHERE episode_id = ?", (episode_id,))
    con.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
