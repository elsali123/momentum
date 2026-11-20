"""End to end: the real pipeline on a tiny offline dataset (tests/conftest.py), with failures injected.

One test per failure path in the pipeline diagram. Each checks the outcome, what readers can see
in the store, and what is left on disk for a person to inspect.
"""

import json

import av
import pyarrow.parquet as pq
import pytest

from conftest import FRAMES_PER_EPISODE, NUM_EPISODES, episode_id, statuses
from robot_ingest import pipeline
from robot_ingest import video
from robot_ingest.catalog import build_catalog, connect
from robot_ingest.config import Config, PipelineSettings
from robot_ingest.pipeline import select_candidates
from robot_ingest.store import Store


def catalog_ids(store: Store) -> set[str]:
    return set(pq.read_table(store.root / "episodes.parquet", columns=["episode_id"])["episode_id"].to_pylist())


# --- the happy path, as a baseline ------------------------------------------------------------

def test_clean_run_keeps_every_episode_with_one_video_frame_per_data_row(fake):
    outcomes = fake.run(episodes=(0, 1, 2))

    assert statuses(outcomes) == {episode_id(i): "ingested" for i in (0, 1, 2)}
    con = connect(fake.store.root)
    assert con.execute("SELECT count(*) FROM episodes").fetchone()[0] == 3
    for (path, num_frames) in con.execute("SELECT video_path, num_frames FROM cameras").fetchall():
        with av.open(str(fake.store.root / path)) as c:
            assert sum(1 for _ in c.decode(video=0)) == num_frames == FRAMES_PER_EPISODE
    assert fake.report()["summary"] == {"fake": {"ingested": 3}}
    assert not (fake.store.root / ".staging").exists() or not any((fake.store.root / ".staging").iterdir())


# --- rejected: the data itself is bad ---------------------------------------------------------

def test_unusable_source_data_is_rejected_with_its_reason(fake):
    fake.plan[1] = "source_error"

    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "ingested", episode_id(1): "rejected"}
    assert fake.rejection(1)["reasons"] == ["19 data rows but the episode index says 20"]
    assert fake.bundle(1) is None
    assert catalog_ids(fake.store) == {episode_id(0)}


def test_impossible_numbers_are_rejected_but_the_raw_copy_is_kept_for_inspection(fake):
    fake.plan[0] = "nan"

    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "rejected"}
    assert "obs_ee_pos contains NaN or inf" in fake.rejection(0)["reasons"]
    assert fake.store.raw_episode_path("fake", 0).exists()
    assert fake.bundle(0) is None


def test_a_video_with_the_wrong_frame_count_is_rejected_after_three_tries(fake, sleeps):
    fake.plan[0] = "short_video"

    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "rejected"}
    assert fake.rejection(0)["reasons"] == [f"source segment holds {FRAMES_PER_EPISODE - 5} frames, "
                                            f"episode has {FRAMES_PER_EPISODE}"]
    assert sleeps == [2, 4]  # tried three times, in case the download was cut short


# --- kept with warnings: odd but usable -------------------------------------------------------

def test_odd_but_usable_data_is_kept_with_warnings(fake):
    fake.plan.update({0: "jump", 1: "black"})

    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "ingested", episode_id(1): "ingested"}
    assert fake.bundle(0)["episode"]["validation_warnings"] == ["end effector jumps 0.20 m between frames 9 and 10"]
    [warning] = fake.bundle(1)["episode"]["validation_warnings"]
    assert warning.startswith("camera cam_front is black")
    assert fake.bundle(1)["cameras"][0]["mean_luma"] < 3


# --- failed: an unexpected error --------------------------------------------------------------

def test_a_dropped_download_is_retried_and_the_episode_kept(fake, sleeps, monkeypatch):
    real_transcode, calls = video._transcode, []

    def drops_first_try(segment, *args):
        calls.append(segment.start_s)
        if len(calls) == 1:
            raise OSError("Connection reset by peer")
        return real_transcode(segment, *args)

    monkeypatch.setattr(video, "_transcode", drops_first_try)

    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "ingested"}
    assert len(calls) == 2
    assert sleeps == [2]


def test_a_download_that_never_recovers_fails_that_episode_and_the_run_carries_on(fake, sleeps):
    fake.plan[1] = "unreachable"

    outcomes = fake.run(episodes=(0, 1, 2))

    assert statuses(outcomes) == {episode_id(0): "ingested", episode_id(1): "failed", episode_id(2): "ingested"}
    assert sleeps == [2, 4]
    assert fake.rejection(1) is None  # a network problem says nothing about the data
    [failed] = [o for o in fake.report()["episodes"] if o["status"] == "failed"]
    assert failed["episode_id"] == episode_id(1) and "missing.mp4" in failed["messages"][0]


def test_a_mapper_bug_fails_one_episode_and_is_reported(fake):
    fake.plan[0] = "bug"

    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "failed", episode_id(1): "ingested"}
    assert fake.report()["summary"] == {"fake": {"failed": 1, "ingested": 1}}
    assert fake.report()["episodes"][0]["messages"] == ["RuntimeError: bug in the mapper"]


# --- crashes never leave half an episode visible ----------------------------------------------

def test_an_error_while_committing_leaves_nothing_half_written(fake, monkeypatch):
    real_commit = Store.commit

    def disk_full_for_episode_1(self, eid, staging):
        if eid == episode_id(1):
            raise OSError("No space left on device")
        return real_commit(self, eid, staging)

    monkeypatch.setattr(Store, "commit", disk_full_for_episode_1)

    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes)[episode_id(1)] == "failed"
    assert not fake.store.episode_dir(episode_id(1)).exists()
    assert not (fake.store.root / ".staging" / episode_id(1)).exists()
    assert catalog_ids(fake.store) == {episode_id(0)}


def test_leftovers_from_a_killed_run_are_invisible_and_replaced(fake):
    leftover = fake.store.root / ".staging" / episode_id(0)
    leftover.mkdir(parents=True)
    (leftover / "frames.parquet").write_bytes(b"half written when the power went out")
    assert list(fake.store.iter_bundles()) == []  # readers never look in .staging

    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "ingested"}
    frames = pq.read_table(fake.store.episode_dir(episode_id(0)) / "frames.parquet")
    assert frames.num_rows == FRAMES_PER_EPISODE


# --- re-runs -----------------------------------------------------------------------------------

def test_a_rerun_skips_finished_episodes_and_retries_the_rest(fake):
    fake.plan[1] = "source_error"
    fake.run(episodes=(0, 1))
    video_0 = fake.store.episode_dir(episode_id(0)) / "videos" / "cam_front.mp4"
    written = video_0.stat().st_mtime_ns

    fake.plan.clear()  # the customer fixed episode 1
    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "skipped", episode_id(1): "ingested"}
    assert video_0.stat().st_mtime_ns == written
    assert fake.rejection(1) is None  # committing clears the old rejection note


def test_new_code_reprocesses_and_removes_a_kept_copy_that_now_fails(fake, monkeypatch):
    fake.run(episodes=(0,))
    assert catalog_ids(fake.store) == {episode_id(0)}

    monkeypatch.setattr(pipeline, "__version__", "0.2.0-test")
    fake.plan[0] = "nan"  # e.g. the new version checks something the old one didn't
    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "rejected"}
    assert fake.bundle(0) is None  # the viewer must not keep showing the old copy
    assert catalog_ids(fake.store) == set()


# --- the replacement loop ----------------------------------------------------------------------

def test_rejects_are_replaced_by_the_next_candidate_in_the_shuffled_list(fake):
    order = select_candidates(list(range(NUM_EPISODES)), seed=0)
    fake.plan.update({order[0]: "source_error", order[2]: "nan"})

    outcomes = fake.run(num_episodes=3)

    kept = {o.episode_id for o in outcomes if o.status == "ingested"}
    rejected = {o.episode_id for o in outcomes if o.status == "rejected"}
    assert kept == {episode_id(i) for i in (order[1], order[3], order[4])}
    assert rejected == {episode_id(i) for i in (order[0], order[2])}


def test_running_out_of_candidates_keeps_what_passed_and_warns(fake, caplog):
    fake.plan.update({i: "source_error" for i in range(NUM_EPISODES) if i not in (2, 5)})

    outcomes = fake.run(num_episodes=3)

    assert {o.episode_id for o in outcomes if o.status == "ingested"} == {episode_id(2), episode_id(5)}
    assert len(outcomes) == NUM_EPISODES
    assert "only 2 of 3 episodes accepted" in caplog.text


# --- the catalog refuses to write a table that doesn't match the schema --------------------------

def test_the_catalog_refuses_fields_the_schema_does_not_know(fake):
    fake.run(episodes=(0,))
    path = fake.store.episode_dir(episode_id(0)) / "episode.json"
    bundle = json.loads(path.read_text())
    bundle["episode"]["surprise"] = 1
    path.write_text(json.dumps(bundle))

    with pytest.raises(ValueError, match="unexpected {'surprise'}"):
        build_catalog(fake.store, Config(PipelineSettings(), {}, ()))
