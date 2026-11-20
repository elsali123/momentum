"""A tiny offline stand-in for a customer dataset, so the real pipeline can run end to end in tests.

Like a LeRobot v3 dataset, one MP4 holds every episode back to back. Each episode is
FRAMES_PER_EPISODE frames of a flat gray image, and its data is a slow straight-line motion
that passes every check. Tests inject a fault per episode through `fake.plan`:

    source_error   the mapper finds the source unusable (SourceDataError)    -> rejected
    bug            the mapper crashes with an unexpected exception            -> failed
    nan            a NaN in the end-effector position                         -> rejected
    jump           the arm teleports 20 cm between frames 9 and 10            -> kept, warned
    black          the camera recorded black frames                           -> kept, warned
    short_video    the video holds 5 frames fewer than the data               -> rejected
    unreachable    the video file can't be opened, on every try               -> failed
    moved_video    the same footage, read from a different file               -> kept
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pytest

from robot_ingest import pipeline, video
from robot_ingest.config import Config, PipelineSettings, RobotSpec, SourceSpec
from robot_ingest.pipeline import EpisodeOutcome, run_ingest
from robot_ingest.schema import frames_table
from robot_ingest.sources import MAPPERS
from robot_ingest.sources.base import CameraSource, Mapper, NormalizedEpisode, SourceDataError, make_episode_id
from robot_ingest.sources.lerobot import VideoSegment
from robot_ingest.store import Store

FPS = 10.0
FRAMES_PER_EPISODE = 20
NUM_EPISODES = 8
SOURCE_ID = "fake"
GRAY, BLACK = 120, 0
ROBOT = RobotSpec("fake_arm", "Fake arm", "Fake gripper", tuple(f"j{i}" for i in range(6)), 0.08, False)


def episode_id(index: int) -> str:
    return make_episode_id(SOURCE_ID, index)


def write_concatenated_video(path: Path, black_episodes: set[int]) -> None:
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=int(FPS), options={"crf": "18", "x264-params": "keyint=5:bframes=0"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(NUM_EPISODES * FRAMES_PER_EPISODE):
            level = BLACK if i // FRAMES_PER_EPISODE in black_episodes else GRAY
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), level, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, int(FPS))
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))


class FakeRepo:
    """The slice of LeRobotRepo the pipeline and mappers use."""

    version = "v3.0"
    fps = FPS

    def __init__(self, root: Path, plan: dict[int, str]):
        self.root = root
        self.plan = plan
        self._lock = threading.Lock()
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": self.version, "fps": FPS}))

    def fetch(self, path: str) -> Path:
        return self.root / path

    def video_path(self) -> Path:
        """Written on first use, so a test can still add `black` faults to the plan before running."""
        path = self.root / "videos" / "cam_front.mp4"
        with self._lock:  # episodes are normalized in parallel
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                write_concatenated_video(path, {i for i, fault in self.plan.items() if fault == "black"})
        return path

    def moved_video_path(self) -> Path:
        path = self.root / "videos" / "moved" / "cam_front.mp4"
        source = self.video_path()
        with self._lock:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
        return path


class FakeMapper(Mapper):
    def episode_indices(self) -> list[int]:
        return list(range(NUM_EPISODES))

    def normalize(self, episode_index: int) -> NormalizedEpisode:
        fault = self.repo.plan.get(episode_index)
        if fault == "source_error":
            raise SourceDataError("19 data rows but the episode index says 20")
        if fault == "bug":
            raise RuntimeError("bug in the mapper")

        n = FRAMES_PER_EPISODE
        t = np.arange(n)
        pos = np.column_stack([0.4 + 0.002 * t, np.zeros(n), np.full(n, 0.3)])
        if fault == "nan":
            pos[5, 0] = np.nan
        if fault == "jump":
            pos[10:, 0] += 0.2
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        frames = frames_table(episode_id(episode_index), {
            "frame_index": t, "timestamp_s": t / FPS,
            "obs_ee_pos": pos, "obs_ee_quat": quat, "obs_joint_pos": np.zeros((n, 6)),
            "obs_gripper_closure": np.linspace(0, 1, n),
            "act_ee_pos": pos, "act_ee_quat": quat, "act_gripper_closure": np.linspace(0, 1, n),
        })

        start = episode_index * n / FPS
        video_frames = n - 5 if fault == "short_video" else n
        if fault == "unreachable":
            url = str(self.repo.root / "missing.mp4")
        elif fault == "moved_video":
            url = str(self.repo.moved_video_path())
        else:
            url = str(self.repo.video_path())
        camera = CameraSource("cam_front", "external", "observation.images.cam_front",
                              VideoSegment(url, start, start + video_frames / FPS, "h264"), serial="front")
        return NormalizedEpisode(
            episode_id=episode_id(episode_index),
            source_episode_index=episode_index,
            frames=frames,
            raw_frames=pa.table({"index": t}),
            cameras=[camera],
            metadata={
                "task": "stack the blocks", "language_instruction": "Put the red block on the blue block",
                "language_instructions": ["Put the red block on the blue block"], "success": True,
                "quality_rating": None, "native_capture_hz": None, "action_ee_semantics": "commanded_target",
                "operator_id": None, "scene_id": None, "license": "cc-by-4.0", "source_extras": {},
            },
        )


@dataclass
class FakeSource:
    store: Store
    repo: FakeRepo
    plan: dict[int, str] = field(default_factory=dict)

    def run(self, num_episodes: int = 3, episodes: tuple[int, ...] = (), force: bool = False,
            settings: PipelineSettings = PipelineSettings()) -> list[EpisodeOutcome]:
        spec = SourceSpec(source_id=SOURCE_ID, customer_id="customer_test", mapper="fake", repo_id="test/fake",
                          revision="rev-1", robot_id=ROBOT.robot_id, num_episodes=num_episodes, seed=0,
                          episodes=tuple(episodes), license="cc-by-4.0")
        config = Config(settings, {ROBOT.robot_id: ROBOT}, (spec,))
        return run_ingest(config, self.store, self.repo.root / "cache", workers=2, force=force)

    def bundle(self, index: int) -> dict | None:
        return self.store.load_bundle(episode_id(index))

    def rejection(self, index: int) -> dict | None:
        path = self.store.root / "rejected" / f"{episode_id(index)}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def report(self) -> dict:
        return json.loads((self.store.root / "reports" / "ingest_report.json").read_text())


@pytest.fixture
def fake(tmp_path, monkeypatch) -> FakeSource:
    plan: dict[int, str] = {}
    repo = FakeRepo(tmp_path / "source", plan)
    monkeypatch.setitem(MAPPERS, "fake", FakeMapper)
    monkeypatch.setattr(pipeline, "LeRobotRepo", lambda repo_id, revision, cache_dir: repo)
    return FakeSource(Store(tmp_path / "store"), repo, plan)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Skip the retry back-off waits, and record them."""
    waits: list[float] = []
    monkeypatch.setattr(video, "time", SimpleNamespace(sleep=waits.append))
    return waits


def statuses(outcomes: list[EpisodeOutcome]) -> dict[str, str]:
    return {o.episode_id: o.status for o in outcomes}
