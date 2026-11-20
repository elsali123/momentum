"""Read-only access to a LeRobot dataset on the Hugging Face Hub (layout v2.1 or v3.0), pinned to a revision.

This layer knows *where* things live (which file, which rows, which seconds of which video) but
nothing about what the columns mean; that is the customer mapper's job (rh20t.py, droid.py).
A new customer shipping LeRobot data therefore needs a mapper, not a new reader.

  v2.1: one parquet + one MP4 per camera per episode
  v3.0: many episodes concatenated per parquet / MP4; meta/episodes/*.parquet holds row and
        time ranges. Videos are read remotely with HTTP range requests (see video.py), so we
        never download a whole 200 MB video file to get one 10 MB episode out of it.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

SUPPORTED_VERSIONS = ("v2.1", "v3.0")


@dataclass(frozen=True)
class VideoSegment:
    """Seconds [start_s, end_s) of one MP4, which hold exactly one episode of one camera."""

    url: str
    start_s: float
    end_s: float
    codec: str


class LeRobotRepo:
    def __init__(self, repo_id: str, revision: str, cache_dir: Path):
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = Path(cache_dir)
        self._data_files: dict[str, pa.Table] = {}
        self._lock = threading.Lock()
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"{repo_id}: unsupported LeRobot codebase_version {self.version!r}")

    # ------------------------------------------------------------------ files

    def fetch(self, path: str) -> Path:
        """Download one repo file (cached, resumable, pinned to the revision)."""
        return Path(hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision, cache_dir=self.cache_dir))

    def url(self, path: str) -> str:
        return hf_hub_url(self.repo_id, path, repo_type="dataset", revision=self.revision)

    # ------------------------------------------------------------------ dataset-level metadata

    @cached_property
    def info(self) -> dict:
        return json.loads(self.fetch("meta/info.json").read_text())

    @property
    def version(self) -> str:
        return self.info["codebase_version"]

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def video_keys(self) -> list[str]:
        return [k for k, f in self.info["features"].items() if f["dtype"] == "video"]

    def video_codec(self, video_key: str) -> str:
        return self.info["features"][video_key].get("info", {}).get("video.codec", "unknown")

    @cached_property
    def episodes(self) -> dict[int, dict]:
        """Per-episode index rows keyed by episode_index (lengths, tasks, file locations)."""
        if self.version == "v2.1":
            lines = self.fetch("meta/episodes.jsonl").read_text().splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
        else:
            files = sorted(
                f for f in HfApi().list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision)
                if f.startswith("meta/episodes/") and f.endswith(".parquet")
            )
            rows = []
            for f in files:
                path = self.fetch(f)
                cols = [c for c in pq.read_schema(path).names if not c.startswith("stats/")]  # skip ~230 stats columns
                rows += pq.read_table(path, columns=cols).to_pylist()
        return {int(r["episode_index"]): r for r in rows}

    @cached_property
    def tasks(self) -> dict[int, str]:
        if self.version == "v2.1":
            lines = self.fetch("meta/tasks.jsonl").read_text().splitlines()
            return {int(r["task_index"]): r["task"] for r in map(json.loads, filter(str.strip, lines))}
        table = pq.read_table(self.fetch("meta/tasks.parquet"))
        return dict(zip(table["task_index"].to_pylist(), table["task"].to_pylist()))

    # ------------------------------------------------------------------ per-episode access

    def read_frames(self, episode_index: int) -> pa.Table:
        """All source rows of one episode, ordered by frame_index, columns untouched."""
        ep = self.episodes[episode_index]
        if self.version == "v2.1":
            chunk = episode_index // self.info["chunks_size"]
            path = self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
            table = pq.read_table(self.fetch(path))
        else:
            path = self.info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
            data = self._load_data_file(path)
            table = data.filter(pc.equal(data["episode_index"], episode_index))
        return table.sort_by("frame_index")

    def video_segment(self, episode_index: int, video_key: str) -> VideoSegment:
        ep = self.episodes[episode_index]
        codec = self.video_codec(video_key)
        if self.version == "v2.1":
            chunk = episode_index // self.info["chunks_size"]
            path = self.info["video_path"].format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)
            return VideoSegment(self.url(path), 0.0, ep["length"] / self.fps, codec)
        prefix = f"videos/{video_key}"
        path = self.info["video_path"].format(
            video_key=video_key, chunk_index=ep[f"{prefix}/chunk_index"], file_index=ep[f"{prefix}/file_index"]
        )
        return VideoSegment(self.url(path), float(ep[f"{prefix}/from_timestamp"]), float(ep[f"{prefix}/to_timestamp"]), codec)

    def _load_data_file(self, path: str) -> pa.Table:
        """v3.0 data files hold many episodes; load each one once and share it across worker threads."""
        with self._lock:
            if path not in self._data_files:
                self._data_files[path] = pq.read_table(self.fetch(path))
            return self._data_files[path]
