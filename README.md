# robot-ingest

Ingests two customers' robot-learning datasets into one canonical store that serves both the
application UI and model training (world models and policies):

| Customer | Source | Robot | Episodes ingested |
|---|---|---|---|
| A | RH20T config 3 (`robot-lev/rh20t_cfg3`, LeRobot v3.0) | UR5 + WSG-50, up to 9 cameras | 10 (seeded random) |
| A | RH20T config 7 (`robot-lev/rh20t_cfg7`, LeRobot v3.0) | KUKA iiwa + Robotiq 2F-85, up to 10 cameras | 10 (seeded random) |
| B | DROID 1.0.1 (`cadene/droid_1.0.1`, LeRobot v2.1) | Franka Panda + Robotiq 2F-85, 3 cameras | 10 (seeded random) |

Every source is normalized to one schema: SI units, one orientation convention verified against
the robots' kinematics, gripper closure 0–1, labeled action semantics, per-episode licensing,
and frame-exact H.264 video per camera. It also comes with a small web viewer.
**[DESIGN.md](DESIGN.md) records every design decision and the data problems found in the sources.**
[DEPLOYMENT.md](DEPLOYMENT.md) is the plan for running this as a production service.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12
itself) and internet access to huggingface.co. No system FFmpeg is needed.

```bash
uv sync                          # create the environment from uv.lock
uv run robot-ingest ingest       # download, normalize, validate, store (~7 min, ~1.7 GB downloaded)
uv run robot-ingest serve        # then open http://127.0.0.1:8000
```

The ingest prints a per-source summary and writes everything under `data/`. It is idempotent:
running it again skips episodes that are already up to date, and an interrupted run resumes where
it stopped. After a code change only the affected steps rerun: a new mapper or validation-rule
version re-normalizes episodes but keeps their videos, which are re-encoded only when their own
inputs change (`--force` rebuilds everything). `rm -rf data/` resets everything.

Reference run (Apple-silicon laptop, default 4 workers, about 7 minutes):

| Source | Episodes | Frames | Robot time | Camera videos | Success / failure / unknown | Non-commercial | Warnings |
|---|---|---|---|---|---|---|---|
| `rh20t_cfg3` | 10 | 4,998 | 8.3 min | 70 | 8 / 2 / 0 | 8 | 0 |
| `rh20t_cfg7` | 10 | 4,559 | 7.6 min | 100 | 9 / 0 / 1 | 3 | 2 (end-effector jumps in the source recording) |
| `droid` | 10 | 2,279 | 2.5 min | 30 | 6 / 4 / 0 | 0 | 4 (no language instruction) |

0 rejected; every video's frame count matches its episode. The store is 808 MB, almost all of it
video; downloads are ~170 MB of Parquet and metadata plus ~1.5 GB of video byte ranges.

## Viewer

`uv run robot-ingest serve` starts a local web app over the store:

- episode list with filters (customer, robot, outcome, license) and text search
- all cameras of an episode playing in sync under one scrubber; `Space` play/pause, `←`/`→` step one frame
- end-effector position (measured vs. action target), gripper closure, joint and force plots with a time cursor (click a plot to seek)
- validation warnings, provenance (source repo, revision, episode index) and license for each episode
- arm jumps marked on every plot, with a button that moves the playhead to the frame
- a "black video" tag on any camera tile whose stream is black
- a run report (sidebar, "Last run"): kept / rejected / failed per source, and every rejected or
  failed recording with its reasons

## Querying the store

The catalog (`catalog.sqlite`) is updated as each episode is committed, and the viewer reads it
directly. At the end of every run it is also exported as plain Parquet, which `robot-ingest query`
runs SQL over with DuckDB (views: `episodes`, `cameras`, `robots`, `frames`):

```bash
uv run robot-ingest query "SELECT robot_id, count(*) AS episodes, round(sum(duration_s)/60, 1) AS minutes,
                                  count(*) FILTER (WHERE success) AS successful
                           FROM episodes GROUP BY ALL"
uv run robot-ingest query "SELECT episode_id, unnest(validation_warnings) AS warning FROM episodes"
```

From Python, e.g. to build a training set:

```python
import av
import pyarrow.parquet as pq
from robot_ingest.catalog import connect

con = connect("data/store")
ids = [r[0] for r in con.sql("""
    SELECT episode_id FROM episodes
    WHERE success AND commercial_use_allowed AND action_ee_semantics = 'commanded_target'""").fetchall()]

frames = pq.read_table(f"data/store/episodes/{ids[0]}/frames.parquet")    # one row per frame
video = con.execute("SELECT video_path FROM cameras WHERE episode_id = ? ORDER BY role <> 'wrist' LIMIT 1",
                    [ids[0]]).fetchone()[0]                                # the wrist camera if the episode has one
with av.open(f"data/store/{video}") as container:
    images = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
assert len(images) == frames.num_rows                                    # video frame i == frames row i
```

## Output layout

```
data/
  hf_cache/                                  Hugging Face download cache (Parquet + metadata only)
  store/
    catalog.sqlite                           catalog: episodes, cameras, robots (updated per episode)
    episodes.parquet  cameras.parquet  robots.parquet     the same tables as Parquet, exported each run
    episodes/<episode_id>/
      episode.json                           episode + camera metadata, validation warnings, build info
      frames.parquet                         canonical state/action timeseries
      videos/<camera_id>.mp4                 H.264, frame-exact with frames.parquet
      thumbs/<camera_id>.jpg
    raw/<source_id>/                         source rows exactly as received + source metadata snapshot
    rejected/<episode_id>.json               episodes that failed validation, with reasons
    reports/ingest_report.json               outcome and warnings of the last run
```

Column-by-column schema with units: [DESIGN.md §5](DESIGN.md#5-canonical-schema-reference) and
[`src/robot_ingest/schema.py`](src/robot_ingest/schema.py).

## Configuration and options

Sources, their pinned Hub revisions, sample sizes, seeds and robot specs are in
[`configs/sources.toml`](configs/sources.toml).

```bash
uv run robot-ingest ingest --source droid           # one source (repeatable)
uv run robot-ingest ingest --num-episodes 3         # override the sample size
uv run robot-ingest ingest --workers 8 --force      # more parallelism, rebuild everything
uv run robot-ingest catalog                         # rebuild the catalog from the episode folders
uv run robot-ingest --store /path/to/store serve --port 8080
```

## Checks

```bash
uv run pytest                                       # all tests (offline, a few seconds)
uv run pytest tests/test_pipeline_failures.py -v    # end to end on a fake dataset, every failure path
uv run python scripts/check_orientation.py          # onboarding check: poses vs. joint kinematics, per source
uv run python scripts/check_orientation.py --as-declared   # the same, without the RH20T fixes (shows the bugs)
```

## Code map

| Path | What it does |
|---|---|
| `src/robot_ingest/sources/lerobot.py` | reads LeRobot v2.1 / v3.0 datasets on the Hub (files, row ranges, video segments) |
| `src/robot_ingest/sources/rh20t.py`, `droid.py` | customer mappers: what each column means → canonical schema |
| `src/robot_ingest/transforms.py`, `kinematics.py` | rotation/gripper conventions; forward kinematics and pose checks |
| `src/robot_ingest/schema.py` | canonical tables, columns, types and units |
| `src/robot_ingest/video.py` | frame-exact cut + H.264 re-encode, reading only the needed byte ranges |
| `src/robot_ingest/validate.py` | rejection rules and warnings |
| `src/robot_ingest/pipeline.py`, `store.py`, `catalog.py` | orchestration, on-disk layout, catalog |
| `src/robot_ingest/viewer/` | FastAPI app + single-page viewer |
| `src/robot_ingest/cli.py` | `robot-ingest` command |

## Without uv

```bash
python3 -m venv .venv && .venv/bin/pip install -e .     # Python >= 3.11
.venv/bin/robot-ingest ingest && .venv/bin/robot-ingest serve
```
