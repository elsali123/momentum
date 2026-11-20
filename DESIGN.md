# Design decisions: robot data ingest

This document records every design choice in the pipeline: what was decided, what else was
considered, why, and what it costs. [README.md](README.md) covers how to run it.

Contents: [1. Requirements](#1-requirements-as-understood) ·
[2. Source survey](#2-source-survey-what-the-data-actually-looks-like) ·
[3. Architecture](#3-architecture) · [4. Decisions](#4-decisions) ·
[5. Schema reference](#5-canonical-schema-reference) · [6. Known limitations](#6-known-limitations)

---

## 1. Requirements as understood

- **Sources.** Customer A ships RH20T as LeRobot copies: config 3 (UR5 + WSG-50, 9 cameras) and
  config 7 (KUKA + Robotiq 2F-85, 10 cameras), 10 episodes each. Customer B ships DROID 1.0.1,
  10 random episodes.
- **Consumers of the normalized data.**
  1. *The application UI*: list and filter episodes, play every camera in sync, inspect state and
     action over time. Needs fast metadata queries, browser-playable video and thumbnails.
  2. *Training*: world models (video conditioned on actions) and policies (observations → actions),
     across embodiments. Needs frame-exact alignment between video and state, one action/state
     convention across robots, provenance, and filters (robot, success, license).
- **Operations.** Re-runnable from a clean checkout; a simple video viewer as a stretch goal.

---

## 2. Source survey (what the data actually looks like)

Every claim below was checked against the data, not taken from dataset cards.

| | RH20T cfg3 | RH20T cfg7 | DROID 1.0.1 |
|---|---|---|---|
| Customer | A | A | B |
| Hub repo @ pinned revision | `robot-lev/rh20t_cfg3` @ `343835b` | `robot-lev/rh20t_cfg7` @ `7b9a4de` | `cadene/droid_1.0.1` @ `56b622a` |
| Container | LeRobot v3.0: ~20 episodes per 200 MB MP4; all frames in one 78 MB Parquet | LeRobot v3.0 | LeRobot v2.1: one Parquet + one MP4 per camera per episode |
| Episodes available | 798 | 896 | 95,600 |
| Arm (joints) | UR5 (6) | KUKA LBR iiwa (7) | Franka Panda (7) |
| Gripper signal | WSG-50 opening width, mm (0–110) | Robotiq 2F-85 width, mm (0–85) | Robotiq 2F-85 closure, 0 open … 1 closed |
| Force/torque | ATI wrist sensor (+ the arm's own estimate) | same | none |
| Cameras | 9 streams named by serial, 1 wrist | 10 streams, 2 wrist | 3 named streams, 1 wrist |
| Frame rate | 10 fps, resampled from 2.2–11.8 Hz native | 10 fps, same | 15 fps |
| Video | AV1 640×360, keyframe every 2 frames | same | AV1 320×180 |
| EE orientation | quaternion, labelled `xyzw` | quaternion, labelled `xyzw` | roll/pitch/yaw Euler angles |
| Action | EE pose at t+1 + commanded gripper width | same | commanded EE pose, joint targets, gripper (6 variants) |
| Language | none (tasks are just "task 1") | none | up to 3 instructions, often empty |
| Outcome | rating 0–9 (0 robot failure, 1 task failure, 2–9 quality) | same | success boolean |

### Findings and how each is handled

| # | Finding | Handled by |
|---|---|---|
| F1 | RH20T stores a stream for every camera in every episode, but not all of them recorded: no cfg3 episode has footage on all 9 streams (3–8 do), while the 10 cfg7 episodes sampled used all 10. Cameras that didn't record are present in the file as **black filler** (verified by decoding). Each episode lists its real cameras in `meta/rh20t_episodes.json`. | Only real cameras are ingested ([D11](#d11-cameras)); a black-stream check backs it up |
| F2 | RH20T was captured at 2.2–11.8 Hz and resampled to 10 fps, so many frames are repeats. | `native_capture_hz` column + warning below 5 Hz ([D10](#d10-time-base)) |
| F3 | The RH20T port forces the first quaternion component ≥ 0, which flips the quaternion sign **6,702 times** in cfg3 alone. | Sign made continuous in time ([D6](#d6-orientation-convention-verified-against-kinematics)) |
| F4 | RH20T cfg3 quaternions are really **w, x, y, z**, not the declared x, y, z, w. Read as declared, orientation disagrees with the joint angles by 3.9° (median) and implies a base frame 171° away from the one the positions use. | Reordered; verified to 0.16° ([D6](#d6-orientation-convention-verified-against-kinematics)) |
| F5 | RH20T cfg7 quaternions are x, y, z, w, but expressed in a base frame **turned 180° about z** relative to the positions, with the tool z-axis pointing back into the arm. | Two constant rotations; verified to 0.01° ([D6](#d6-orientation-convention-verified-against-kinematics)) |
| F6 | RH20T's action EE pose is *exactly* the next frame's measured pose (not a command); its gripper action is a commanded width. DROID's `action.cartesian_position` is a teleop command that leads the measured pose by up to 7 cm. | `action_ee_semantics` column ([D9](#d9-actions)) |
| F7 | The RH20T port reduced task names to "task N"; there is no natural-language instruction. | `language_instruction` is null; flagged as a customer gap ([§6](#6-known-limitations)) |
| F8 | RH20T is licensed **per scene**: scenes 1–5 CC BY-SA 4.0, scenes 6–10 CC BY-NC 4.0 (no commercial use, including models trained on them). | Per-episode `license` + `commercial_use_allowed` ([D17](#d17-licensing-and-usage-rights)) |
| F9 | RH20T has two wrenches: `force`/`torque` from the ATI wrist sensor (±20 N on cfg3) and `robot_ft`, the arm's own estimate (±220 N on cfg3). | `obs_wrench` = ATI sensor; `robot_ft` stays in the raw copy |
| F10 | DROID's port mis-maps metadata: `task_category` equals `building` and `date` equals `collector_id` in every episode checked. | Both dropped; `building`/`collector_id` kept ([D15](#d15-metadata-repair-and-enrichment)) |
| F11 | DROID labels the 7 joint values with 6 Cartesian axis names (x … yaw). | Names ignored; joint order comes from the robot spec |
| F12 | DROID's `action.original` duplicates `action.cartesian_position` + gripper. | Ignored |
| F13 | DROID's wrist-camera extrinsics are constant through the episode although the camera moves with the arm. | Extrinsics stored for fixed cameras only ([D11](#d11-cameras)) |
| F14 | DROID has episodes as short as 1 frame, and many without a language instruction. | Minimum-length rejection; "no language instruction" warning ([D16](#d16-validation-and-quarantine)) |
| F15 | DROID reports the **flange** pose (0.0 mm from forward kinematics); RH20T reports a fingertip TCP 0.25 m (cfg3) / 0.227 m (cfg7) along the tool axis. | Offset recorded per robot, not silently converted ([D7](#d7-end-effector-reference-point)) |
| F16 | All three sources store AV1, which Safari only plays on M3-or-newer Macs. | Re-encode to H.264 ([D12](#d12-video-encoding)) |

---

## 3. Architecture

```
 configs/sources.toml: sources pinned to a Hub revision, robot specs, pipeline settings
                │
 Hugging Face ──┴─► LeRobotRepo ──────────────► Mapper ───────────────────► NormalizedEpisode
 repo @ revision    format reader (v2.1/v3.0)   customer semantics          canonical frames table
                    where: files, row ranges,   (rh20t.py, droid.py):       + cameras + metadata
                    video segments               units, conventions,         + raw rows
                                                 metadata repair                    │
                                                                                    ▼
             ┌──────────────────────────── per episode, in a thread pool ───────────────────────┐
             │ raw copy ─► validate_frames ─► frames.parquet ─► transcode each camera ─► commit  │
             │              │ errors            (HTTP range reads → H.264, frame-exact)    atomic │
             │              ▼                                                             rename │
             │        rejected/<id>.json  (next candidate is tried instead)                      │
             └───────────────────────────────────────────────────────────────────────────────────┘
                                                                                    │
                     build_catalog: episodes/cameras/robots.parquet  ◄──────────────┘
                                         │
                     DuckDB views (catalog.connect) ──► viewer API, training queries, `robot-ingest query`
```

| Module | Responsibility |
|---|---|
| `config.py` | Typed view of `configs/sources.toml` (sources, robots, pipeline settings) |
| `sources/lerobot.py` | Format reader: LeRobot v2.1 and v3.0 layouts on the Hub, pinned revision |
| `sources/rh20t.py`, `sources/droid.py` | Customer mappers: source columns → canonical schema |
| `transforms.py` | Rotation conventions, quaternion continuity, gripper normalization (pure functions) |
| `kinematics.py` | Forward kinematics and the pose-vs-joints consistency fit |
| `schema.py` | Every table, column, type and unit; builds and checks the frames table |
| `video.py` | Cut one episode from a (remote) MP4 and re-encode it frame-exactly |
| `validate.py` | Hard errors and warnings per episode |
| `store.py` | On-disk layout, staging, atomic commit, quarantine |
| `pipeline.py` | Orchestration: selection, parallelism, bundles, report |
| `catalog.py` | Consolidated tables and DuckDB views |
| `viewer/` | FastAPI API + one static HTML page |
| `scripts/check_orientation.py` | Onboarding check of pose conventions against kinematics |

---

## 4. Decisions

Each decision: **what** was decided, the **alternatives**, **why**, and the **cost**.

### D1. Our own canonical schema, not LeRobot

- **Decision.** A pipeline-owned schema (section 5) for every customer, rather than writing
  normalized LeRobot datasets.
- **Alternatives.** (a) LeRobot v3 output, one dataset per embodiment. (b) Keep each customer's
  format and normalize at read time.
- **Why.** LeRobot fixes one feature set per dataset (same joints, same cameras), but our store mixes
  6- and 7-joint arms, 3 to 10 cameras, and episodes whose camera sets differ (F1). One schema
  across customers is what makes cross-customer UI queries and multi-embodiment training simple.
  Normalizing at read time would push every quirk in section 2 into every consumer.
- **Cost.** We maintain the schema and its docs. A LeRobot *export* (per embodiment, for
  off-the-shelf training code) is a straightforward derived view and isn't built here.

### D2. Storage: Parquet + per-episode MP4 + DuckDB views, with the episode as the unit of work

- **Decision.** Each episode is a self-contained bundle: `episode.json` (metadata and camera rows),
  `frames.parquet`, one MP4 and one thumbnail per camera. A catalog step consolidates the bundles
  into `episodes.parquet`, `cameras.parquet` and `robots.parquet`, and `catalog.connect()` exposes
  them plus a `frames` view (glob over all episodes) through in-memory DuckDB.
- **Alternatives.** Postgres for metadata (heavier to run for reviewers); one big frames table per
  source (fewer files, but every re-ingest rewrites it); a persisted `.duckdb` file (DuckDB
  resolves relative paths against the working directory, so a persisted catalog breaks when the
  store moves).
- **Why.** Parquet is columnar, typed and readable by every training stack; one file per episode
  makes ingest idempotent and parallel (each worker writes only its own bundle) and makes deleting
  one customer's episode trivial. The bundles are the source of truth; the catalog is derived and
  rebuildable (`robot-ingest catalog`). All paths inside the tables are relative to the store root,
  so the same layout works on S3.
- **Cost.** At DROID scale (95k episodes) per-episode files become many small files; the fix is a
  periodic compaction into large sorted files, with the episodes table pointing at row ranges
  (what LeRobot v3 does). Not needed at 30 episodes.

### D3. Raw (bronze) and normalized (silver) zones

- **Decision.** For every episode we keep the source rows exactly as received in
  `raw/<source>/episodes/<index>.parquet`, plus a snapshot of the source metadata files in
  `raw/<source>/meta/`. Source video is **not** copied: we pin the Hub revision instead.
- **Why.** Normalization bugs get fixed (D6 is the proof). With the raw rows we can re-normalize
  without re-downloading, and debug a rejected episode (raw is written before validation).
  Copying video would mean downloading whole 200 MB files for 10 MB of content (D13).
- **Cost.** A pinned Hub revision is only as durable as the repo. In production, customer uploads
  should land in our own bucket first and that becomes the raw zone.

### D4. Episode identity

- **Decision.** `episode_id = <source_id>-<source_episode_index:06d>`, e.g. `droid-087235`.
  `source_revision` is stored alongside it.
- **Alternatives.** Random UUIDs (re-runs would duplicate); content hashes (stable across
  re-exports but opaque in the UI and logs).
- **Why.** Deterministic, so re-runs overwrite rather than duplicate; human-readable in the UI.
- **Cost.** If a customer re-exports with a different episode order, indices shift. Natural keys
  are kept where they exist (RH20T's source folder name is in `source_extras`) to re-key if needed.

### D5. Units and frames

- **Decision.** SI everywhere (m, rad, s, N, N·m). Poses are in **each robot's own base frame, z
  up**. There is no shared world frame.
- **Why.** None of the sources ship base-to-world or camera calibration we could trust (D11), so a
  common world frame would be invented. Per-robot base frames are what policies consume anyway.
- **Cost.** Cross-robot spatial comparisons need a calibration step later.

### D6. Orientation convention, verified against kinematics

- **Decision.** Orientation is a unit quaternion in **x, y, z, w** order (scipy/ROS), expressed in
  the same base frame as the position, for a tool frame whose **z-axis is the gripper's approach
  direction**. Its sign is made continuous over time, starting with w ≥ 0 on frame 0. Action
  quaternions are put in the same hemisphere as the state quaternion of the same frame.
- **How the source conventions were established.** A reported pose must equal forward kinematics
  of the recorded joint angles up to two constant rotations (how the base and the tool are
  mounted). `kinematics.fit_pose_to_fk` fits those constants in closed form and reports what's
  left over. `scripts/check_orientation.py` runs it on 20 episodes per source:

  | Source | Read as declared: orientation residual / base mismatch / tool z | After the mapper's fix | Position residual |
  |---|---|---|---|
  | RH20T cfg3 (UR5) | 3.94° / 171° / −0.11 | 0.16° / 0.7° / +1.00 | 0.7 mm |
  | RH20T cfg7 (KUKA) | 0.01° / 180° / −1.00 | 0.01° / 0.6° / +1.00 | 3.0 mm |
  | DROID (Franka, Euler xyz extrinsic) | 0.00° / 0.0° / +1.00 | (no fix needed) | 0.0 mm |

  For DROID the same fit also settles the Euler convention: extrinsic `xyz` fits to 0.00°;
  intrinsic `XYZ` and `zyx` are off by up to 78°.
- **The fixes**, in `sources/rh20t.py`: cfg3 reorders w, x, y, z → x, y, z, w; cfg7 applies
  `Rz(180°) · R · Ry(180°)`. The mapper refuses to run on an RH20T config without a verified entry.
- **Ongoing guard.** Every episode is re-checked at ingest (`validate.check_orientation_vs_joints`):
  if orientation disagrees with the joints by more than 1° (correct conventions fit to < 0.25°;
  wrong ones are off by 2–80°), the episode gets a warning. That check can't see a whole-frame
  rotation like cfg7's (it needs positions), which is why the onboarding script exists.
- **Alternatives.** Trust the dataset card (would have shipped two silently wrong configs);
  replace orientation with FK everywhere (depends on nominal DH parameters and drops the
  customer's own calibration).
- **Cost.** We carry nominal DH tables for each arm. Their accuracy shows in the position residual
  (≤ 3 mm), which is fine for checking conventions but not for calibration.
- **Not stored:** 6D rotation or axis-angle variants. They are cheap to derive at load time and
  every training recipe wants a different one.

### D7. End-effector reference point

- **Decision.** Keep the source's reported EE point and record where it is:
  `robots.ee_offset_from_flange_m` (0 for DROID's flange, 0.25 / 0.227 m for RH20T's TCPs, both
  measured with the FK fit in D6).
- **Why.** Converting everything to the flange (or to a fingertip) would need tool geometry we
  only estimated. Making the offset explicit lets a training job align them in one line, and
  stops it being silently mixed.

### D8. Gripper

- **Decision.** `gripper_closure` ∈ [0, 1], **0 = fully open, 1 = fully closed**, for both state
  and action. Physical width = (1 − closure) × `robots.gripper_max_width_m`.
- **Why.** Every parallel gripper maps onto it, so policies can share one gripper output across
  embodiments. The direction matches DROID and the Robotiq command convention, so the largest
  source needs no transform, and the name says which way it goes. Out-of-range values (sensor
  overshoot) are clipped and counted in a warning.
- **Cost.** Grip force and non-parallel grippers (suction, dexterous hands) don't fit; they'd get
  their own columns.

### D9. Actions

- **Decision.** The canonical action is an **absolute** EE target pose (`act_ee_pos`,
  `act_ee_quat`) plus `act_gripper_closure`; `act_joint_pos` is filled when the source has
  joint-space actions (DROID) and null otherwise. `episodes.action_ee_semantics` says what the EE
  target is: `commanded_target` (DROID teleop command) or `next_observed_state` (RH20T, F6).
- **Alternatives.** Deltas or velocities (derivable from absolute targets at load time, in whatever
  frame and normalization a recipe wants; the reverse isn't lossless); keeping each source's own
  action vector only (unusable across customers).
- **Why.** Absolute targets are the common denominator of both customers. The semantics column
  matters: a policy trained on "next observed state" learns where the arm *went*, not what was
  *commanded*, and mixing them silently is a classic multi-dataset bug.
- **Cost.** The original action vectors live only in the raw zone.

### D10. Time base

- **Decision.** Keep each source's native frame rate (10 fps RH20T, 15 fps DROID); no resampling
  at ingest. `timestamp_s` is relative to episode start (float64). RH20T's native capture rate is
  stored as `native_capture_hz`, with a warning when it's under half the stored fps.
- **Why.** Resampling is lossy and recipe-specific (world models and policies want different
  rates). Ingest stays a faithful normalization; loaders resample.
- **Cost.** Every consumer has to read `fps` instead of assuming one.

### D11. Cameras

- **Decision.** Ingest only cameras that actually recorded (F1); give each a `role` (`wrist` or
  `external`) and list wrist cameras first. Extrinsics (camera pose in the base frame) are stored
  only for fixed cameras whose pose the source provides (DROID exterior cameras). Intrinsics
  columns aren't in v1 (no source provides them).
- **Why.** Black filler streams waste storage and decode time and would teach a world model that
  cameras sometimes go black. `role` lets a policy pick "the wrist camera" across robots whose
  camera names have nothing in common. A per-camera mean-brightness check catches a dead stream
  even when the source claims it recorded.
- **Consequence.** Camera sets really do vary per episode: 2 of the 10 cfg3 episodes have **no
  wrist footage at all**. A policy that needs a wrist view has to filter on `cameras.role`; the
  schema makes that one join instead of a decode-and-inspect pass.
- **Cost.** RH20T camera *identity* is a serial number, so "the same viewpoint" across RH20T
  configs isn't expressible yet. That needs the customer's rig description.

### D12. Video encoding

- **Decision.** Decode each episode's frames and write one MP4 per camera: **H.264 (libx264),
  yuv420p, CRF 20, preset medium, keyframe every 10 frames, no B-frames, `+faststart`**, at the
  source resolution and frame rate. Video frame *i* is frames row *i*: the frame count is checked
  against the episode length and a mismatch rejects the episode.
- **Alternatives.** Remux the source AV1 without re-encoding (bit-exact and smaller, but Safari
  needs an M3+ Mac); keep AV1 for training plus an H.264 preview (double storage, more code).
- **Why.** Plays in every browser; decoded by every training stack; random access costs at most
  10 frames of decode; no B-frames means decode order is display order, which keeps seeking
  simple for loaders; faststart lets the browser start playback before downloading the whole file.
  Re-encoding also replaces the source's keyframe-every-2-frames encoding, so our files come out
  **smaller** than the source despite H.264 (e.g. 7.7 MB vs 14.8 MB for a 93-second camera).
- **Cost.** One generation of lossy compression (CRF 20 is visually lossless at these sizes) and
  about 3–4 s of encode per camera-episode.

### D13. Read only the bytes we need

- **Decision.** Videos are opened straight from the pinned Hub URL. FFmpeg reads the MP4 index,
  seeks to the keyframe before the episode, and downloads only that episode's byte range. The
  decoder then skips to the episode start and stops at its end, selecting frames by their nearest
  frame slot.
- **Why.** RH20T packs ~20 episodes into each 200 MB file per camera, so 10 random episodes across
  9–10 cameras would mean ~18 GB of whole-file downloads per config. Range reads fetch ~10 MB per
  camera-episode instead, and the same code handles DROID's per-episode files.
- **Cost.** A network dependency during transcoding: reads are retried (3 attempts, backoff) and
  FFmpeg reconnects on drops. Parquet data is downloaded normally (78 MB per RH20T config, one row
  group, so no partial reads).

### D14. Episode selection

- **Decision.** For every source, candidates are a seeded shuffle of all episode indices
  (`seed` in the config; an explicit `episodes` list overrides it). The pipeline walks the order
  until `num_episodes` episodes are accepted, so **a rejected episode is replaced by the next
  candidate** and the sample size holds deterministically. DROID failures are not filtered out.
- **Why.** The same rule for every source, representative across tasks, scenes and operators, and
  reproducible. Failed episodes are valuable for world models and for learning recovery; they are
  labelled (`success`) and filterable instead of being dropped at ingest.

### D15. Metadata repair and enrichment

- **Decision.** Mappers own the episode metadata: outcome (`success`: DROID's flag; RH20T
  `rating ≥ 2`, since 0 is robot failure and 1 is task failure; −1 → unknown), `quality_rating`,
  `operator_id`, `scene_id`, `task`, `language_instructions` (DROID's up-to-three paraphrases,
  deduplicated, empty ones dropped). Fields the source gets wrong are dropped, not propagated
  (DROID `task_category` and `date`, F10). Source-specific leftovers with no canonical column go
  in `source_extras` (JSON).
- **Why.** The UI and training filters need the same columns for every customer; wrong metadata is
  worse than missing metadata.
- **Cost.** Per-source knowledge lives in mapper code, so it has to stay small and documented
  (each mapper's docstring is the column-meaning table).

### D16. Validation and quarantine

- **Decision.** Two severities:
  - **Errors reject the episode** (written to `rejected/<id>.json` with reasons; the raw rows are
    kept): fewer than 10 frames; frame indices with gaps; timestamps not increasing or drifting more
    than half a frame from `index / fps`; NaN or inf; joint count ≠ robot; joints beyond one turn
    (degrees?); positions beyond 2 m (millimetres?); non-unit quaternions; gripper outside [0, 1];
    a video segment holding the wrong number of frames.
  - **Warnings are stored on the episode** (`validation_warnings`) and shown in the UI: EE jumps
    > 10 cm between frames, orientation disagreeing with joint angles (D6), clipped gripper values,
    heavy resampling, missing language, black camera streams.
- **Why.** The thresholds are loose physical bounds aimed at *mapper* mistakes (units,
  conventions, wrong columns), which affect every episode, rather than at judging demonstration
  quality. Quality signals become filterable warnings, because what counts as bad data depends
  on the consumer.
- **Cost.** Warnings need someone to read them; the ingest report (`reports/ingest_report.json`)
  and the UI are where they surface.

### D17. Licensing and usage rights

- **Decision.** Every episode carries `license` and `commercial_use_allowed`. RH20T's license is
  set per scene (F8); DROID uses its dataset card's license. `commercial_use_allowed` is derived
  from the license, and `null` means unknown, so a human decides.
- **Why.** 11 of the 20 RH20T episodes in this sample are non-commercial, and the restriction
  extends to models trained on them. Training-set selection has to be able to filter on it, and
  so does anything shown to third parties.

### D18. Idempotency, atomicity, lineage, versioning

- **Decision.** Each bundle is built in `.staging/` and moved into `episodes/` with a rename.
  Every episode records its repo, revision, source format and index, the schema and pipeline
  versions, and `ingested_at`. Units and docs are embedded in the Parquet field metadata.
  Re-runs are incremental per step (D23); `--force` rebuilds everything.
- **Why.** A crash never leaves a half-written episode visible; re-runs are cheap and resumable;
  any row can be traced back to the exact source bytes.

### D19. Format readers vs customer mappers; config-driven onboarding

- **Decision.** Two layers. A *format reader* knows a container layout (`LeRobotRepo`: v2.1 and
  v3.0) but not what the columns mean. A *customer mapper* knows the semantics (units,
  conventions, metadata quirks) but not where the bytes are. Sources, pinned revisions, robots,
  sample sizes and seeds live in `configs/sources.toml`.
- **Why.** The next LeRobot customer needs only a mapper; a customer shipping RLDS, HDF5 or MCAP
  needs only a reader. A new embodiment is a config block.

### D20. Concurrency and failure handling

- **Decision.** Episodes run in a thread pool (`--workers`, default 4); cameras within an episode
  run sequentially. One episode failing (exception) or being rejected (bad data) never stops the
  run: it's reported, and the exit code is non-zero only for unexpected failures.
- **Why.** The work is mostly network reads and FFmpeg calls, which release the GIL, so threads are
  enough. Episode-level parallelism already overlaps network with encoding; nested pools would add
  complexity for little gain at this scale.
- **Cost.** One machine. At full-dataset scale this becomes a queue of episode jobs (same
  function, one job per episode).

### D21. Viewer

- **Decision.** A FastAPI app (`robot-ingest serve`) with two JSON endpoints over the DuckDB
  catalog, plus one static HTML page with no build step. Features: episode list with filters
  (customer, robot, outcome, license, text), synchronized multi-camera playback with one scrubber,
  frame stepping and speed control, EE/gripper/joint/force plots with a time cursor (click to
  seek), validation warnings and provenance.
- **How sync works.** The first video is the clock; the others are re-seeked when they drift more
  than 1.5 frames. Seeks target the middle of a frame's display interval, `(i + 0.5) / fps`, so the
  shown frame is exactly frame *i*. Videos are served with HTTP range support, so seeking doesn't
  download whole files.
- **Alternatives.** Rerun (robotics-native but a large dependency and a separate desktop app, not
  something resembling a product UI); Streamlit (awkward for synced video).

### D22. Tooling

- **uv** with a lockfile, so "clean checkout → same environment" is one command.
- **PyAV** bundles its own FFmpeg (with libdav1d for AV1 decoding and libx264 for encoding), so
  there's no dependency on a system FFmpeg.
- **scipy** for rotation conversions: rotation conventions are the classic source of silent bugs,
  so we use the reference implementation and test our conventions against explicit matrices.
- **DuckDB** for SQL over Parquet without a server.
- **pytest**: 39 tests, all offline. 20 unit tests cover rotation conventions, quaternion
  continuity, gripper normalization, validation rules, deterministic selection, kinematics, and
  frame-exact video cutting on a synthetic video whose frames encode their own index.
  `tests/test_pipeline_failures.py` (15 tests) runs the real pipeline end to end on a tiny fake
  dataset (`tests/conftest.py`) and injects every failure path: bad source data, impossible
  numbers, short video, a dropped and a dead download, a mapper bug, an error during commit,
  leftovers from a killed run, re-runs, a new code version that now rejects a kept episode,
  the replacement loop, running out of candidates, and a catalog/schema mismatch.
  `tests/test_viewer.py` (4 tests) checks the viewer API on a store built the same way.

### D23. Re-runs redo only the steps whose inputs changed

- **Decision.** Each step has its own version: `Mapper.version` per customer mapper,
  `RULES_VERSION` for validation, `ENCODER_VERSION` for video. Every bundle records what it was
  made from in `episode.json` under `build`: the *data* step (source revision, schema, mapper and
  rules versions, robot spec, license, minimum length) and the *video* settings (encoder version,
  CRF, keyframe interval), plus a fingerprint per camera video that hashes its exact inputs
  (source URL, which includes the pinned revision; time range; frame count; encoding settings).
  A re-run skips an episode when both steps are unchanged. When only the data step changed, the
  episode is re-normalized and re-validated, and each video whose fingerprint still matches is
  hard-linked into the new bundle instead of re-encoded. `--force` rebuilds everything.
- **Alternatives.** One pipeline version for everything (the previous design: any code change
  re-encoded every video); content-addressed video storage shared across bundles (no linking, but
  a different store layout and garbage collection of unreferenced files).
- **Why.** Video encoding is nearly all of the compute (roughly 100 CPU-hours for all of DROID, by
  estimate), while mapper and rule changes are the common case. Fingerprinting each video by its
  inputs makes reuse safe: a mapper change that moves a camera's segment re-encodes only that video.
- **Cost.** A version must be bumped when a step's output changes; forgetting means stale data.
  Keying the fingerprint on the source file's content hash instead of the revision would also
  survive unrelated upstream commits (a README edit bumps the revision today). On object storage,
  the hard link becomes a server-side copy.

---

## 5. Canonical schema reference

The source of truth is `src/robot_ingest/schema.py`; the same docs and units are embedded in the
Parquet field metadata.

**`frames`**: one row per episode frame, `episodes/<episode_id>/frames.parquet`

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `episode_id` | string | | episode |
| `frame_index` | int32 | | 0-based; equals the frame index in every camera video |
| `timestamp_s` | float64 | s | since episode start |
| `obs_ee_pos` | float32[3] | m | measured EE position, robot base frame |
| `obs_ee_quat` | float32[4] | | measured EE orientation, xyzw, sign-continuous, tool z = approach |
| `obs_joint_pos` | list<float32> | rad | in `robots.joint_names` order |
| `obs_gripper_closure` | float32 | 0–1 | 0 open, 1 closed |
| `obs_wrench` | float32[6] (nullable) | N, N·m | wrist F/T sensor; null without a sensor |
| `act_ee_pos`, `act_ee_quat` | float32[3], [4] | m | EE target; see `action_ee_semantics` |
| `act_gripper_closure` | float32 | 0–1 | commanded closure |
| `act_joint_pos` | list<float32> (nullable) | rad | joint target, when the source has one |

**`episodes`**: one row per episode. Identity and lineage (`episode_id`, `customer_id`,
`source_id`, `source_repo`, `source_revision`, `source_format`, `source_episode_index`,
`schema_version`, `pipeline_version`, `ingested_at`), content (`robot_id`, `task`,
`language_instruction`, `language_instructions`, `num_frames`, `fps`, `duration_s`, `camera_ids`,
`has_wrench`), outcome and quality (`success`, `quality_rating`, `native_capture_hz`,
`validation_warnings`), semantics (`action_ee_semantics`), people and place (`operator_id`,
`scene_id`), rights (`license`, `commercial_use_allowed`), and `source_extras` (JSON).

**`cameras`**: one row per episode × camera: `camera_id`, `role`, `serial`, `width`, `height`,
`fps`, `num_frames`, `video_path`, `thumbnail_path`, `codec`, `source_video_key`,
`source_codec`, `extrinsics_pos`, `extrinsics_quat`, `mean_luma`.

**`robots`**: one row per embodiment: `robot_id`, `arm_model`, `gripper_model`, `num_joints`,
`joint_names`, `gripper_max_width_m`, `has_ft_sensor`, `kinematics`, `ee_offset_from_flange_m`.

---

## 6. Known limitations

Things this version deliberately doesn't do, or can't do with the data as delivered:

- **No language for RH20T.** The port kept only "task N". The original RH20T task descriptions
  have to come from the customer.
- **Force/torque bias compensation is unverified.** RH20T's official config lists sensor offsets
  and tool gravity; whether the port applied them isn't documented. Stored as delivered.
- **No camera calibration.** No intrinsics in any source, no extrinsics for RH20T, and DROID's
  wrist extrinsics are unusable (F13).
- **Nominal kinematics.** The DH tables are manufacturer values (≤ 3 mm position residual):
  good enough to verify conventions, not to calibrate.
- **Small files at scale.** One frames file per episode (D2) needs compaction for full datasets.
- **Single machine, thread pool.** No orchestrator, no per-episode retries across runs beyond
  re-running the command (which resumes).
- **Lossy video.** One generation of H.264 at CRF 20 (D12).
- **Privacy.** RH20T notes that volunteers' faces and voices may appear. Nothing is blurred and
  no audio is ingested.
