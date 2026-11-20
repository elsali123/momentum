"""Ingest orchestration: select -> read -> normalize -> validate -> transcode -> commit -> catalog.

Per source:
  1. resolve the episode candidates (seeded shuffle, or an explicit list)
  2. ingest candidates in parallel until `num_episodes` are accepted; a rejected episode is
     replaced by the next candidate, so the sample size holds and stays deterministic
Per episode (one bundle, built in staging and committed atomically):
  mapper.normalize -> validate_frames -> raw copy -> frames.parquet -> one H.264 video per
  camera -> episode.json -> commit.

Re-runs redo only what changed. Each bundle records what its steps were made from (`build_info`):
an episode whose inputs are all unchanged is skipped; when only the data step changed (a new mapper
or rule version, say), it is re-normalized and re-validated but keeps its videos, because each video
is fingerprinted by its own inputs and re-encoded only when those change.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from robot_ingest import __version__
from robot_ingest.catalog import build_catalog
from robot_ingest.config import Config, PipelineSettings, SourceSpec
from robot_ingest.schema import SCHEMA_VERSION
from robot_ingest.sources import MAPPERS
from robot_ingest.sources.base import Mapper, NormalizedEpisode, SourceDataError, commercial_use_allowed, make_episode_id
from robot_ingest.sources.lerobot import LeRobotRepo
from robot_ingest.store import Store
from robot_ingest.validate import RULES_VERSION, validate_camera, validate_frames
from robot_ingest.video import ENCODER_VERSION, TranscodeResult, VideoError, transcode_segment, video_fingerprint

log = logging.getLogger(__name__)


@dataclass
class EpisodeOutcome:
    episode_id: str
    source_id: str
    status: str  # "ingested" | "skipped" (already up to date) | "rejected" (bad data) | "failed" (error)
    messages: list[str] = field(default_factory=list)
    seconds: float = 0.0
    videos_reused: int = 0  # camera videos carried over from the episode's previous version instead of re-encoded

    @property
    def accepted(self) -> bool:
        return self.status in ("ingested", "skipped")


def select_candidates(available: list[int], seed: int, explicit: tuple[int, ...] = ()) -> list[int]:
    """Deterministic candidate order. The pipeline walks it until enough episodes are accepted."""
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f"episodes not in source: {missing}")
        return list(explicit)
    order = sorted(available)
    random.Random(seed).shuffle(order)
    return order


def run_ingest(
    config: Config,
    store: Store,
    cache_dir: Path,
    source_ids: list[str] | None = None,
    num_episodes: int | None = None,
    workers: int = 4,
    force: bool = False,
) -> list[EpisodeOutcome]:
    sources = [config.source(s) for s in source_ids] if source_ids else list(config.sources)
    outcomes: list[EpisodeOutcome] = []
    for spec in sources:
        outcomes += ingest_source(spec, config, store, cache_dir, num_episodes, workers, force)
    build_catalog(store, config)
    write_report(store, outcomes)
    return outcomes


def ingest_source(spec: SourceSpec, config: Config, store: Store, cache_dir: Path,
                  num_episodes: int | None, workers: int, force: bool) -> list[EpisodeOutcome]:
    log.info("source %s: %s @ %s", spec.source_id, spec.repo_id, spec.revision[:10])
    repo = LeRobotRepo(spec.repo_id, spec.revision, cache_dir)
    mapper = MAPPERS[spec.mapper](spec, config.robots[spec.robot_id], repo)
    snapshot_source_metadata(mapper, store)

    candidates = select_candidates(mapper.episode_indices(), spec.seed, spec.episodes)
    target = len(candidates) if spec.episodes else (num_episodes or spec.num_episodes)
    outcomes: list[EpisodeOutcome] = []
    accepted, next_candidate = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while accepted < target and next_candidate < len(candidates):
            batch = candidates[next_candidate:next_candidate + target - accepted]
            next_candidate += len(batch)
            for outcome in pool.map(lambda i: ingest_episode(mapper, i, config, store, force), batch):
                reused = f"{outcome.videos_reused} videos reused; " if outcome.videos_reused else ""
                log.info("  %-22s %-9s %4.1fs %s", outcome.episode_id, outcome.status, outcome.seconds,
                         (reused + "; ".join(outcome.messages))[:160])
                outcomes.append(outcome)
                accepted += outcome.accepted
    if accepted < target:
        log.warning("source %s: only %d of %d episodes accepted", spec.source_id, accepted, target)
    return outcomes


def ingest_episode(mapper: Mapper, index: int, config: Config, store: Store, force: bool) -> EpisodeOutcome:
    spec, settings = mapper.spec, config.pipeline
    episode_id = make_episode_id(spec.source_id, index)
    started = time.monotonic()
    build = build_info(mapper, settings)

    def outcome(status: str, messages: list[str], videos_reused: int = 0) -> EpisodeOutcome:
        return EpisodeOutcome(episode_id, spec.source_id, status, messages, time.monotonic() - started, videos_reused)

    previous = None if force else store.load_bundle(episode_id)  # --force rebuilds everything, videos included
    if is_up_to_date(previous, build):
        return outcome("skipped", [])

    staging = None
    try:
        episode = mapper.normalize(index)
        # Bronze copy first, so a rejected episode can still be inspected.
        raw_path = store.raw_episode_path(spec.source_id, index)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode.raw_frames, raw_path, compression="zstd")

        fps = mapper.repo.fps
        validation = validate_frames(episode.frames, mapper.robot, fps, settings.min_episode_frames)
        validation.warnings[:0] = episode.warnings
        if validation.errors:
            return reject(store, episode_id, spec, validation.errors)

        staging = store.new_staging_dir(episode_id)
        pq.write_table(episode.frames, staging / "frames.parquet", compression="zstd")
        camera_rows, fingerprints, reused = [], {}, 0
        encoding = {"fps": fps, "expected_frames": episode.frames.num_rows,
                    "crf": settings.video_crf, "keyframe_interval": settings.video_keyframe_interval}
        for camera in episode.cameras:
            video = staging / "videos" / f"{camera.camera_id}.mp4"
            thumb = staging / "thumbs" / f"{camera.camera_id}.jpg"
            fingerprints[camera.camera_id] = video_fingerprint(camera.segment, **encoding)
            old = reusable_video(previous, camera.camera_id, fingerprints[camera.camera_id], store)
            if old:
                store.reuse_file(store.root / old["video_path"], video)
                store.reuse_file(store.root / old["thumbnail_path"], thumb)
                result = TranscodeResult(old["num_frames"], old["width"], old["height"], old["mean_luma"])
                reused += 1
            else:
                result = transcode_segment(camera.segment, video, thumb, **encoding)
            validation.warnings += validate_camera(camera.camera_id, result.mean_luma)
            final_dir = store.episode_dir(episode_id)
            camera_rows.append({
                "episode_id": episode_id,
                "camera_id": camera.camera_id,
                "role": camera.role,
                "serial": camera.serial,
                "width": result.width,
                "height": result.height,
                "fps": fps,
                "num_frames": result.num_frames,
                "video_path": store.relative(final_dir / "videos" / video.name),
                "thumbnail_path": store.relative(final_dir / "thumbs" / thumb.name),
                "codec": "h264",
                "source_video_key": camera.source_video_key,
                "source_codec": camera.segment.codec,
                "extrinsics_pos": camera.extrinsics_pos,
                "extrinsics_quat": camera.extrinsics_quat,
                "mean_luma": result.mean_luma,
            })

        bundle = {"episode": episode_row(episode, mapper, validation.warnings), "cameras": camera_rows,
                  "build": {**build, "videos": fingerprints}}
        (staging / "episode.json").write_text(json.dumps(bundle, indent=2))
        store.commit(episode_id, staging)
        return outcome("ingested", validation.warnings, reused)
    except (SourceDataError, VideoError) as exc:
        return reject(store, episode_id, spec, [str(exc)])
    except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the run; it is reported
        log.exception("episode %s failed", episode_id)
        return outcome("failed", [f"{type(exc).__name__}: {exc}"])
    finally:
        if staging is not None and staging.exists():
            store.discard_staging(staging)


def episode_row(episode: NormalizedEpisode, mapper: Mapper, warnings: list[str]) -> dict:
    spec, fps, n = mapper.spec, mapper.repo.fps, episode.frames.num_rows
    return {
        "episode_id": episode.episode_id,
        "customer_id": spec.customer_id,
        "source_id": spec.source_id,
        "source_repo": spec.repo_id,
        "source_revision": spec.revision,
        "source_format": mapper.source_format,
        "source_episode_index": episode.source_episode_index,
        "robot_id": spec.robot_id,
        **episode.metadata,
        "commercial_use_allowed": commercial_use_allowed(episode.metadata["license"]),
        "num_frames": n,
        "fps": fps,
        "duration_s": n / fps,
        "camera_ids": [c.camera_id for c in episode.cameras],
        "has_wrench": episode.frames["obs_wrench"].null_count == 0,
        "validation_warnings": warnings,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": __version__,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def build_info(mapper: Mapper, settings: PipelineSettings) -> dict:
    """What an episode's stored output is made from, per step. Stored in episode.json as "build".

    data:  frames, metadata and validation; cheap to redo
    video: how videos are encoded; expensive, so each video also gets its own fingerprint (`video_fingerprint`)
    """
    spec = mapper.spec
    info = {
        "data": {
            "source_revision": spec.revision,
            "schema": SCHEMA_VERSION,
            "mapper": f"{spec.mapper}@{mapper.version}",
            "rules": RULES_VERSION,
            "robot": asdict(mapper.robot),
            "license": spec.license,
            "min_episode_frames": settings.min_episode_frames,
        },
        "video": {"encoder": ENCODER_VERSION, "crf": settings.video_crf, "keyframe_interval": settings.video_keyframe_interval},
    }
    return json.loads(json.dumps(info))  # in the form it is stored (tuples become lists), so the two compare equal


def is_up_to_date(bundle: dict | None, build: dict) -> bool:
    """Nothing an episode is made from has changed. Bundles written before build info existed are never up to date."""
    stored = (bundle or {}).get("build", {})
    return stored.get("data") == build["data"] and stored.get("video") == build["video"]


def reusable_video(previous: dict | None, camera_id: str, fingerprint: str, store: Store) -> dict | None:
    """The previous version's camera row, if its video was made from exactly the same inputs and is still on disk."""
    if previous is None or previous.get("build", {}).get("videos", {}).get(camera_id) != fingerprint:
        return None
    row = next((c for c in previous["cameras"] if c["camera_id"] == camera_id), None)
    if row is None or not all((store.root / row[k]).exists() for k in ("video_path", "thumbnail_path")):
        return None
    return row


def reject(store: Store, episode_id: str, spec: SourceSpec, reasons: list[str]) -> EpisodeOutcome:
    store.reject(episode_id, {
        "episode_id": episode_id,
        "source_id": spec.source_id,
        "source_revision": spec.revision,
        "reasons": reasons,
        "rejected_at": datetime.now(timezone.utc).isoformat(),
    })
    return EpisodeOutcome(episode_id, spec.source_id, "rejected", reasons)


def snapshot_source_metadata(mapper: Mapper, store: Store) -> None:
    target = store.raw_meta_dir(mapper.spec.source_id)
    target.mkdir(parents=True, exist_ok=True)
    for path in mapper.snapshot_files():
        shutil.copyfile(mapper.repo.fetch(path), target / Path(path).name)
    (target / "source.json").write_text(json.dumps(asdict(mapper.spec), indent=2))


def write_report(store: Store, outcomes: list[EpisodeOutcome]) -> None:
    path = store.root / "reports" / "ingest_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    for o in outcomes:
        counts = summary.setdefault(o.source_id, {})
        counts[o.status] = counts.get(o.status, 0) + 1
    path.write_text(json.dumps({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": __version__,
        "summary": summary,
        "episodes": [asdict(o) | {"seconds": round(o.seconds, 2)} for o in outcomes],
    }, indent=2))
