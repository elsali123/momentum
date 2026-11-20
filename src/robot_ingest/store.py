"""On-disk layout of the normalized store. Everything under the root is produced by the pipeline.

<root>/
  raw/<source_id>/meta/...                         bronze: snapshot of the source's metadata files
  raw/<source_id>/episodes/<index:06d>.parquet     bronze: the episode's source rows, untouched
  episodes/<episode_id>/episode.json               silver: episode + camera rows (the bundle manifest)
  episodes/<episode_id>/frames.parquet             silver: canonical frames
  episodes/<episode_id>/videos/<camera_id>.mp4     silver: canonical H.264 video
  episodes/<episode_id>/thumbs/<camera_id>.jpg
  episodes.parquet, cameras.parquet, robots.parquet   catalog tables, rebuilt from the bundles
  rejected/<episode_id>.json                       episodes that failed validation, with reasons
  reports/ingest_report.json                       outcome of the last run

An episode bundle is the unit of work: built in .staging/, then moved into episodes/ in one
rename, so a crash never leaves a half-written episode visible to readers.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # bronze
    def raw_meta_dir(self, source_id: str) -> Path:
        return self.root / "raw" / source_id / "meta"

    def raw_episode_path(self, source_id: str, source_episode_index: int) -> Path:
        return self.root / "raw" / source_id / "episodes" / f"{source_episode_index:06d}.parquet"

    # silver
    def episode_dir(self, episode_id: str) -> Path:
        return self.root / "episodes" / episode_id

    def relative(self, path: Path) -> str:
        """Paths stored in tables are relative to the root, so the store can be moved or synced to object storage."""
        return path.relative_to(self.root).as_posix()

    def load_bundle(self, episode_id: str) -> dict | None:
        path = self.episode_dir(episode_id) / "episode.json"
        return json.loads(path.read_text()) if path.exists() else None

    def iter_bundles(self) -> Iterator[dict]:
        for path in sorted((self.root / "episodes").glob("*/episode.json")):
            yield json.loads(path.read_text())

    def new_staging_dir(self, episode_id: str) -> Path:
        staging = self.root / ".staging" / episode_id
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        return staging

    def commit(self, episode_id: str, staging: Path) -> None:
        """Swap the finished staging dir into place, replacing any previous version of the episode."""
        final = self.episode_dir(episode_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        trash = None
        if final.exists():
            trash = self.root / ".trash" / f"{episode_id}-{uuid.uuid4().hex[:8]}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final, trash)
        os.replace(staging, final)
        if trash:
            shutil.rmtree(trash, ignore_errors=True)
        (self.root / "rejected" / f"{episode_id}.json").unlink(missing_ok=True)

    def reuse_file(self, existing: Path, target: Path) -> None:
        """Place an already-stored file into a new bundle without copying its bytes when possible.

        A hard link here; on object storage, a server-side copy. The old bundle can then be replaced
        as usual while the new one keeps the file.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(existing, target)
        except OSError:  # a filesystem without hard links
            shutil.copy2(existing, target)

    def discard_staging(self, staging: Path) -> None:
        shutil.rmtree(staging, ignore_errors=True)

    # quarantine
    def reject(self, episode_id: str, record: dict) -> None:
        path = self.root / "rejected" / f"{episode_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        shutil.rmtree(self.episode_dir(episode_id), ignore_errors=True)  # never leave a stale accepted copy
