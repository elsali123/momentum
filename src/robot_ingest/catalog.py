"""Catalog: consolidated Parquet tables built from the episode bundles, and a DuckDB query layer over them.

The bundles are the source of truth; these tables are derived and can be rebuilt at any time
(`robot-ingest catalog`). The UI reads one small episodes.parquet instead of N bundle files,
and training jobs filter episodes with SQL before touching any frames or video.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from robot_ingest.config import Config
from robot_ingest.schema import CAMERAS, EPISODES, ROBOTS
from robot_ingest.store import Store


def build_catalog(store: Store, config: Config) -> dict[str, int]:
    episodes, cameras = [], []
    for bundle in store.iter_bundles():
        row = dict(bundle["episode"])
        row["ingested_at"] = datetime.fromisoformat(row["ingested_at"])
        row["source_extras"] = json.dumps(row["source_extras"], sort_keys=True)
        episodes.append(row)
        cameras += bundle["cameras"]
    robots = [
        {**asdict(r), "joint_names": list(r.joint_names), "num_joints": r.num_joints}
        for r in config.robots.values()
    ]

    _write(store.root / "episodes.parquet", episodes, EPISODES)
    _write(store.root / "cameras.parquet", cameras, CAMERAS)
    _write(store.root / "robots.parquet", robots, ROBOTS)
    return {"episodes": len(episodes), "cameras": len(cameras), "robots": len(robots)}


def _write(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    for row in rows:  # a mismatch here is a bug in a mapper or the pipeline, not bad data
        if set(row) != set(schema.names):
            raise ValueError(f"{path.name}: row keys differ from schema: "
                             f"missing {set(schema.names) - set(row)}, unexpected {set(row) - set(schema.names)}")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def connect(store_root: str | Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with views over the store. Nothing is copied; queries read the Parquet files.

        con = connect("data/store")
        con.sql("SELECT robot_id, count(*), sum(duration_s) FROM episodes GROUP BY ALL").show()
    """
    root = Path(store_root).resolve()
    con = duckdb.connect()
    for table in ("episodes", "cameras", "robots"):
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{root / table}.parquet')")
    con.execute(f"CREATE VIEW frames AS SELECT * FROM read_parquet('{root}/episodes/*/frames.parquet')")
    return con
