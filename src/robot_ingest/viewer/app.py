"""Episode viewer: a small JSON API over the catalog, plus static files (page, videos, thumbnails).

Videos are served by Starlette's StaticFiles, which answers HTTP range requests, so the browser
can seek inside an MP4 without downloading all of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from robot_ingest.catalog import connect
from robot_ingest.schema import column_matrix
from robot_ingest.validate import BLACK_LUMA, ee_jumps

STATIC_DIR = Path(__file__).parent / "static"

LIST_SQL = """
SELECT e.episode_id, e.customer_id, e.source_id, e.robot_id, r.arm_model, r.gripper_model,
       e.task, e.language_instruction, e.success, e.quality_rating, e.num_frames, e.fps,
       e.duration_s, len(e.camera_ids) AS num_cameras, len(e.validation_warnings) AS num_warnings,
       e.commercial_use_allowed,
       (SELECT c.thumbnail_path FROM cameras c WHERE c.episode_id = e.episode_id
        ORDER BY c.role = 'wrist', c.camera_id LIMIT 1) AS thumbnail_path   -- prefer a scene view
FROM episodes e JOIN robots r USING (robot_id)
ORDER BY e.customer_id, e.source_id, e.source_episode_index
"""


def _rows(relation) -> list[dict]:
    columns = [d[0] for d in relation.description]
    return [dict(zip(columns, row)) for row in relation.fetchall()]


def create_app(store_root: str | Path) -> FastAPI:
    root = Path(store_root).resolve()
    if not (root / "episodes.parquet").exists():
        raise SystemExit(f"no catalog in {root}: run `robot-ingest ingest` first")
    con = connect(root)
    app = FastAPI(title="Episode viewer")

    @app.get("/api/episodes")
    def list_episodes() -> list[dict]:
        return _rows(con.cursor().execute(LIST_SQL))

    @app.get("/api/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict:
        cur = con.cursor()
        episodes = _rows(cur.execute(
            "SELECT e.*, r.arm_model, r.gripper_model, r.joint_names FROM episodes e "
            "JOIN robots r USING (robot_id) WHERE e.episode_id = ?", [episode_id]))
        if not episodes:  # also guards the file path below against arbitrary input
            raise HTTPException(404, f"unknown episode {episode_id}")
        cameras = _rows(cur.execute(
            "SELECT * FROM cameras WHERE episode_id = ? ORDER BY role <> 'wrist', camera_id", [episode_id]))
        for camera in cameras:
            camera["black"] = camera["mean_luma"] < BLACK_LUMA  # same threshold as the ingest warning
        return {"episode": episodes[0], "cameras": cameras, "series": _series(root / "episodes" / episode_id / "frames.parquet")}

    @app.get("/api/run")
    def run_report() -> dict:
        """The last run's outcome counts, and every recording that is not in the store, with the reason."""
        report_path = root / "reports" / "ingest_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        # Rejection notes stay until the recording passes on a later run; failures are only known for the last run.
        rejected = sorted((json.loads(p.read_text()) for p in (root / "rejected").glob("*.json")),
                          key=lambda r: r["episode_id"])
        return {
            "finished_at": report["finished_at"] if report else None,
            "pipeline_version": report["pipeline_version"] if report else None,
            "summary": report["summary"] if report else {},
            "rejected": rejected,
            "failed": [o for o in report["episodes"] if o["status"] == "failed"] if report else [],
        }

    app.mount("/store/episodes", StaticFiles(directory=root / "episodes"), name="store")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _series(path: Path) -> dict:
    """Frames as plot-ready columns: one list per scalar trace, rounded to keep the payload small."""
    frames = pq.read_table(path)

    def traces(name: str) -> list[list[float]] | None:
        if frames[name].null_count == frames.num_rows:
            return None
        values = column_matrix(frames, name).round(4)
        return [values[:, i].tolist() for i in range(values.shape[1])] if values.ndim == 2 else [values.tolist()]

    return {
        "t": frames["timestamp_s"].to_pylist(),
        **{name: traces(name) for name in (
            "obs_ee_pos", "act_ee_pos", "obs_gripper_closure", "act_gripper_closure",
            "obs_joint_pos", "act_joint_pos", "obs_wrench",
        )},
        # Frames to mark on the plots, found by the same check that wrote the ingest warning.
        "events": [{"frame": frame, "label": f"arm jumps {distance:.2f} m"} for frame, distance in ee_jumps(frames)],
    }
