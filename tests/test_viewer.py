"""Viewer API on a store built by the real pipeline from the fake dataset (tests/conftest.py)."""

import pytest
from starlette.testclient import TestClient

from conftest import episode_id
from robot_ingest.viewer.app import create_app


@pytest.fixture
def client(fake, sleeps):
    fake.plan.update({0: "jump", 1: "black", 2: "nan", 3: "unreachable"})
    fake.run(episodes=(0, 1, 2, 3, 4))
    return TestClient(create_app(fake.store.root))


def test_run_report_lists_what_is_not_in_the_store_and_why(client):
    run = client.get("/api/run").json()

    assert run["summary"] == {"fake": {"ingested": 3, "rejected": 1, "failed": 1}}
    [rejected] = run["rejected"]
    assert rejected["episode_id"] == episode_id(2)
    assert "obs_ee_pos contains NaN or inf" in rejected["reasons"]
    [failed] = run["failed"]
    assert failed["episode_id"] == episode_id(3) and "missing.mp4" in failed["messages"][0]


def test_arm_jumps_come_back_as_frames_to_mark_on_the_plots(client):
    jumpy = client.get(f"/api/episodes/{episode_id(0)}").json()
    smooth = client.get(f"/api/episodes/{episode_id(4)}").json()

    assert jumpy["series"]["events"] == [{"frame": 10, "label": "arm jumps 0.20 m"}]
    assert jumpy["episode"]["validation_warnings"] == ["end effector jumps 0.20 m between frames 9 and 10"]
    assert smooth["series"]["events"] == []


def test_black_cameras_are_flagged_per_tile(client):
    black = client.get(f"/api/episodes/{episode_id(1)}").json()
    normal = client.get(f"/api/episodes/{episode_id(4)}").json()

    assert [c["black"] for c in black["cameras"]] == [True]
    assert [c["black"] for c in normal["cameras"]] == [False]


def test_rejected_and_failed_recordings_are_not_listed_as_episodes(client):
    listed = {e["episode_id"] for e in client.get("/api/episodes").json()}

    assert listed == {episode_id(i) for i in (0, 1, 4)}
    assert client.get(f"/api/episodes/{episode_id(2)}").status_code == 404
