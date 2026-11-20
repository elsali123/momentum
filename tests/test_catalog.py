"""The catalog is updated as each episode is committed, never by rescanning the store, and never lists missing files."""

import pyarrow.parquet as pq
import pytest

from conftest import ROBOT, episode_id, statuses
from robot_ingest import pipeline
from robot_ingest.catalog import Catalog, build_catalog
from robot_ingest.config import Config, PipelineSettings
from robot_ingest.store import Store


def catalog_ids(fake) -> set[str]:
    return {r["episode_id"] for r in Catalog(fake.store.catalog_path).query("SELECT episode_id FROM episodes")}


def snapshot_ids(fake) -> set[str]:
    return set(pq.read_table(fake.store.root / "episodes.parquet", columns=["episode_id"])["episode_id"].to_pylist())


def test_a_run_updates_the_catalog_per_episode_without_rescanning_the_store(fake, monkeypatch):
    def no_rescan(self):
        raise AssertionError("the run read every bundle in the store")

    monkeypatch.setattr(Store, "iter_bundles", no_rescan)

    fake.run(episodes=(0, 1, 2))

    assert catalog_ids(fake) == snapshot_ids(fake) == {episode_id(i) for i in (0, 1, 2)}


def test_catalog_rows_round_trip_lists_booleans_and_nulls(fake):
    fake.plan[0] = "jump"
    fake.run(episodes=(0,))

    [row] = Catalog(fake.store.catalog_path).query("SELECT * FROM episodes")
    stored = fake.bundle(0)["episode"]

    assert row["camera_ids"] == ["cam_front"]
    assert row["validation_warnings"] == stored["validation_warnings"] != []
    assert row["success"] is True and row["commercial_use_allowed"] is True
    assert row["quality_rating"] is None
    assert row["ingested_at"] == stored["ingested_at"]


def test_a_rejected_episode_leaves_the_catalog_before_its_files_are_deleted(fake, monkeypatch):
    fake.run(episodes=(0,))
    catalog = Catalog(fake.store.catalog_path)
    listed_while_deleting = []
    real_reject = Store.reject

    def recording_reject(self, eid, record):
        listed_while_deleting.append(bool(catalog.query("SELECT 1 FROM episodes WHERE episode_id = ?", [eid])))
        return real_reject(self, eid, record)

    monkeypatch.setattr(Store, "reject", recording_reject)
    monkeypatch.setattr(pipeline, "RULES_VERSION", "2")
    fake.plan[0] = "nan"

    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "rejected"}
    assert listed_while_deleting == [False]
    assert catalog_ids(fake) == snapshot_ids(fake) == set()


def test_a_crash_between_commit_and_catalog_write_is_repaired_by_the_next_run(fake, monkeypatch):
    real_put = Catalog.put_episode

    def crash_on_episode_0(self, bundle):
        if bundle["episode"]["episode_id"] == episode_id(0):
            raise OSError("process killed")
        return real_put(self, bundle)

    monkeypatch.setattr(Catalog, "put_episode", crash_on_episode_0)
    first = fake.run(episodes=(0, 1))
    assert statuses(first)[episode_id(0)] == "failed"
    assert fake.bundle(0) is not None and catalog_ids(fake) == {episode_id(1)}  # stored, but not listed

    monkeypatch.setattr(Catalog, "put_episode", real_put)
    second = fake.run(episodes=(0, 1))

    assert statuses(second) == {episode_id(0): "skipped", episode_id(1): "skipped"}
    assert catalog_ids(fake) == snapshot_ids(fake) == {episode_id(0), episode_id(1)}


def test_a_schema_mismatch_fails_the_episode_before_anything_is_committed(fake, monkeypatch):
    real_row = pipeline.episode_row

    def row_with_extra_field(episode, *args):
        row = real_row(episode, *args)
        return row | {"surprise": 1} if episode.episode_id == episode_id(0) else row

    monkeypatch.setattr(pipeline, "episode_row", row_with_extra_field)

    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "failed", episode_id(1): "ingested"}
    assert "unexpected {'surprise'}" in outcomes[0].messages[0]
    assert fake.bundle(0) is None
    assert catalog_ids(fake) == {episode_id(1)}


def test_a_full_rebuild_from_the_bundles_recreates_the_same_catalog(fake):
    fake.plan[1] = "black"
    fake.run(episodes=(0, 1, 2))
    before = {t: Catalog(fake.store.catalog_path).query(f"SELECT * FROM {t} ORDER BY 1, 2") for t in ("episodes", "cameras")}

    fake.store.catalog_path.unlink()
    counts = build_catalog(fake.store, Config(PipelineSettings(), {ROBOT.robot_id: ROBOT}, ()))

    assert counts == {"episodes": 3, "cameras": 3, "robots": 1}
    after = {t: Catalog(fake.store.catalog_path).query(f"SELECT * FROM {t} ORDER BY 1, 2") for t in ("episodes", "cameras")}
    assert after == before


@pytest.mark.parametrize("table", ["episodes", "cameras", "robots"])
def test_the_parquet_snapshot_matches_the_catalog(fake, table):
    fake.run(episodes=(0, 1))

    catalog_rows = Catalog(fake.store.catalog_path).query(f"SELECT * FROM {table}")
    snapshot = pq.read_table(fake.store.root / f"{table}.parquet")

    assert snapshot.num_rows == len(catalog_rows)
    assert snapshot.schema.names == list(catalog_rows[0])
