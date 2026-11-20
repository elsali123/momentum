"""Re-runs redo only what changed: per-step build info in episode.json, and videos reused by fingerprint."""

import json

import pytest

from conftest import FakeMapper, episode_id, statuses
from robot_ingest import pipeline, video
from robot_ingest.config import PipelineSettings


@pytest.fixture
def encoded(monkeypatch) -> list[float]:
    """Start times of the video segments actually encoded, as opposed to reused."""
    real_transcode, calls = video._transcode, []

    def counting(segment, *args):
        calls.append(segment.start_s)
        return real_transcode(segment, *args)

    monkeypatch.setattr(video, "_transcode", counting)
    return calls


def video_file(fake, index: int):
    return fake.store.episode_dir(episode_id(index)) / "videos" / "cam_front.mp4"


def test_unchanged_inputs_skip_the_episode_entirely(fake, encoded):
    fake.run(episodes=(0, 1))
    encoded.clear()

    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "skipped", episode_id(1): "skipped"}
    assert encoded == []


@pytest.mark.parametrize("change", [
    pytest.param(lambda mp: mp.setattr(pipeline, "RULES_VERSION", "2"), id="new validation rules"),
    pytest.param(lambda mp: mp.setattr(FakeMapper, "version", "2"), id="new mapper version"),
])
def test_a_data_step_change_reprocesses_the_episode_but_keeps_its_videos(fake, encoded, monkeypatch, change):
    fake.run(episodes=(0, 1))
    inode = video_file(fake, 0).stat().st_ino
    encoded.clear()

    change(monkeypatch)
    outcomes = fake.run(episodes=(0, 1))

    assert statuses(outcomes) == {episode_id(0): "ingested", episode_id(1): "ingested"}
    assert [o.videos_reused for o in outcomes] == [1, 1]
    assert encoded == []
    assert video_file(fake, 0).stat().st_ino == inode  # the same file, linked into the new version, not copied
    build = fake.bundle(0)["build"]["data"]
    assert (build["rules"], build["mapper"]) in {("2", "fake@1"), ("1", "fake@2")}


def test_new_encoding_settings_re_encode_every_video(fake, encoded):
    fake.run(episodes=(0, 1))
    encoded.clear()

    outcomes = fake.run(episodes=(0, 1), settings=PipelineSettings(video_crf=28))

    assert [o.videos_reused for o in outcomes] == [0, 0]
    assert sorted(encoded) == [0.0, 2.0]


def test_only_the_video_whose_source_changed_is_re_encoded(fake, encoded, monkeypatch):
    fake.run(episodes=(0, 1))
    encoded.clear()

    monkeypatch.setattr(FakeMapper, "version", "2")  # the new mapper reads episode 1's camera from another file
    fake.plan[1] = "moved_video"
    outcomes = fake.run(episodes=(0, 1))

    assert {o.episode_id: o.videos_reused for o in outcomes} == {episode_id(0): 1, episode_id(1): 0}
    assert encoded == [2.0]


def test_a_video_missing_from_disk_is_re_encoded_rather_than_reused(fake, encoded, monkeypatch):
    fake.run(episodes=(0,))
    video_file(fake, 0).unlink()
    encoded.clear()

    monkeypatch.setattr(pipeline, "RULES_VERSION", "2")
    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "ingested"}
    assert outcomes[0].videos_reused == 0 and encoded == [0.0]
    assert video_file(fake, 0).exists()


def test_force_rebuilds_everything_including_videos(fake, encoded):
    fake.run(episodes=(0, 1))
    encoded.clear()

    outcomes = fake.run(episodes=(0, 1), force=True)

    assert [o.videos_reused for o in outcomes] == [0, 0]
    assert sorted(encoded) == [0.0, 2.0]


def test_a_bundle_without_build_info_is_rebuilt(fake, encoded):
    fake.run(episodes=(0,))
    path = fake.store.episode_dir(episode_id(0)) / "episode.json"
    bundle = json.loads(path.read_text())
    del bundle["build"]  # as written by a version of the pipeline before build info existed
    path.write_text(json.dumps(bundle))
    encoded.clear()

    outcomes = fake.run(episodes=(0,))

    assert statuses(outcomes) == {episode_id(0): "ingested"}
    assert encoded == [0.0]
    assert "build" in fake.bundle(0)
