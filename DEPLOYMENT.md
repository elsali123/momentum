# Deployment plan

How the take-home pipeline becomes a production service: customers upload data continuously, the
app shows it, training reads it, and each customer's data stays separate and governed.
[DESIGN.md](DESIGN.md) explains the current design; this document covers what changes to run it
for real. Durations are rough estimates for a small team.

## Target architecture

```
Customer ──upload──► Raw bucket (per customer, versioned, never modified)
                          │  new data triggers onboarding
                          ▼
                 Orchestrator ──fan out: one job per recording──► Worker pool (containers, autoscaled)
                                                                     │  translate · check · re-encode
                                                                     ▼
            Normalized bucket (same layout as today)  +  Postgres catalog
                  │                                           │
     Training shards ◄── compaction                           ▼
                  │                             App API (FastAPI) + CDN for video
            GPU training jobs                                 │
                                                 Web UI (customers + internal team)
```

## What changes from today

| Part | Today | Production | Why |
|---|---|---|---|
| Storage | `data/store` on a laptop | Object storage (S3 or GCS) with the same layout; file access through fsspec (`s3://…`) | One layout everywhere. Paths inside the tables are already relative to the store root. |
| Raw data | A pinned Hugging Face revision | Customer uploads land in our own versioned raw bucket first; Hugging Face becomes one import path | We own the untouched copy, so reprocessing never depends on someone else's repo. |
| Committing a recording | Build in a staging folder, then rename | Write every file, then `episode.json` last as the "done" marker; readers ignore folders without it | Object storage has no atomic folder rename. |
| Catalog | Parquet tables + in-memory DuckDB | Postgres for the app; Parquet + DuckDB (reading object storage directly) for analysis and training | The app needs fast filtered lookups and per-customer access control. |
| Execution | One machine, a thread pool | An orchestrator (e.g. Dagster) runs `ingest_episode` as one job per recording on autoscaled container workers, using cheap spot capacity | Jobs are already idempotent, so retries and interrupted machines are safe. |
| Viewer | Local FastAPI serving files from disk | Hosted FastAPI behind single sign-on; videos through a CDN with expiring links (seeking works the same way) | Scale and access control. |
| Monitoring | Log lines + `reports/ingest_report.json` | Metrics, dashboards and alerts: failures, warning rate per customer, queue backlog, cost | A customer changing their format shows up as a warning spike, not a silent bug. |

## Rollout

### Phase 1: cloud-ready (1–2 weeks)
- Storage abstraction (fsspec), a container image built from `uv.lock`, settings and secrets from the environment.
- "Done"-marker commits; a Postgres catalog writer next to the Parquet tables.
- CI running the test suite plus a small end-to-end ingest.

### Phase 2: production MVP (3–4 weeks)
- Raw buckets and an upload/import flow per customer.
- Orchestrator with one job per recording, autoscaled workers, retries with backoff, and a list of rejected and failed recordings for follow-up.
- Viewer behind single sign-on, video through a CDN.
- Dashboards and alerts. The orientation check (`scripts/check_orientation.py`) becomes a required sign-off before a new source goes live.

### Phase 3: training at scale (1–2 months)
- Compaction: per-recording files merged into large sorted files with row-range pointers, plus per-robot training shards next to the GPUs.
- Training-set snapshots: a saved list of recordings with the schema and pipeline versions and the filters used, referenced by every training run so models are reproducible.
- A PyTorch data loader (time windows, action chunks, resampling, per-robot normalization statistics) and a LeRobot export.
- Backfills: bumping a step version (mapper, validation rules or encoder) reprocesses the affected recordings in the background, re-encoding video only where its inputs changed.

### Phase 4: customer product
- Self-serve upload with an automatic data-quality report (validation warnings and the orientation check).
- Annotation, search over task instructions, 3D replay built on the existing kinematics code, face blurring.

## Security and data rights
- **Isolation:** a bucket (or prefix) and access role per customer; workers get only what a job needs; encryption in storage and in transit.
- **Audit:** a log of who accessed raw and normalized data.
- **Deletion:** one workflow removes a recording or a whole customer everywhere: raw objects, normalized folders, training shards, catalog rows and snapshot references, flagging any models trained on them.
- **Licensing:** `commercial_use_allowed` is enforced by the training-set builder, and each snapshot records the licenses it includes.
- **Privacy:** faces are blurred before data is visible beyond the core team. RH20T warns that volunteers' faces and voices may appear.

## Operations
- **Onboarding a customer:**
  1. Add a config entry and a mapper, plus a reader if the file format is new.
  2. Run the orientation check.
  3. Review its report, then go live.
- **Reprocessing:** a version bump triggers a background backfill; the previous version stays readable until it completes.
- **Targets to agree:** new uploads visible in the app within an hour; failed recordings surfaced within minutes; no silent drops (every recording ends up ingested, rejected with reasons, or failed with an error).

## Cost drivers
- **Video storage dominates** (almost all of today's store). Move raw uploads to cold storage after a set period, and keep one normalized copy.
- **Re-encoding:** roughly 100 CPU-hours for all of DROID (estimated from the reference run). Spot capacity keeps it cheap; GPU encoding is an option if speed matters more than cost.
- **CDN traffic** from people watching in the app: serve low-resolution previews in the UI and full resolution only to training.

## Open decisions
- Cloud provider and orchestrator (Dagster, Airflow or Temporal).
- One bucket per customer, or a shared bucket with per-customer prefixes.
- H.264 only, or AV1 originals plus H.264 previews.
- Training input: streamed from object storage, or pre-staged shards next to the GPUs.
