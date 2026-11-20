"""Command line: `robot-ingest ingest | catalog | query | serve`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CONFIG = "configs/sources.toml"
DEFAULT_STORE = "data/store"
DEFAULT_CACHE = "data/hf_cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robot-ingest", description="Normalize customer robot datasets into the canonical store.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"sources/robots config (default {DEFAULT_CONFIG})")
    parser.add_argument("--store", default=DEFAULT_STORE, help=f"normalized store root (default {DEFAULT_STORE})")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download, normalize, validate and store episodes")
    ingest.add_argument("--source", action="append", dest="sources", help="only this source_id (repeatable)")
    ingest.add_argument("--num-episodes", type=int, help="override num_episodes for every source")
    ingest.add_argument("--workers", type=int, default=4, help="episodes processed in parallel (default 4)")
    ingest.add_argument("--force", action="store_true", help="re-ingest episodes that are already up to date")
    ingest.add_argument("--cache-dir", default=DEFAULT_CACHE, help=f"Hugging Face download cache (default {DEFAULT_CACHE})")

    sub.add_parser("catalog", help="rebuild the catalog from the episode folders (recovery; ingest keeps it current)")

    query = sub.add_parser("query", help="run SQL against the store (views: episodes, cameras, robots, frames)")
    query.add_argument("sql")

    serve = sub.add_parser("serve", help="start the episode viewer")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    # Imports are deferred so `--help` is instant.
    from robot_ingest.config import load_config
    from robot_ingest.store import Store

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub", "urllib3", "libav", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # after imports: some libraries reset their level on import

    store = Store(args.store)
    if args.command == "ingest":
        from robot_ingest.pipeline import run_ingest

        outcomes = run_ingest(load_config(args.config), store, Path(args.cache_dir), args.sources,
                              args.num_episodes, args.workers, args.force)
        counts = Counter((o.source_id, o.status) for o in outcomes)
        statuses = ("ingested", "skipped", "rejected", "failed")
        print("\nsource        ingested  skipped  rejected  failed")
        for source_id in dict.fromkeys(o.source_id for o in outcomes):
            print(f"{source_id:13s} " + "  ".join(f"{counts[(source_id, s)]:>7d}" for s in statuses))
        if reused := sum(o.videos_reused for o in outcomes):
            print(f"\n{reused} camera videos reused from the previous version instead of re-encoded")
        print(f"\nstore: {store.root.resolve()}   report: {store.root / 'reports/ingest_report.json'}")
        return 1 if any(o.status == "failed" for o in outcomes) else 0

    if args.command == "catalog":
        from robot_ingest.catalog import build_catalog

        print(build_catalog(store, load_config(args.config)))
        return 0

    if args.command == "query":
        from robot_ingest.catalog import connect

        connect(store.root).sql(args.sql).show(max_width=200)
        return 0

    if args.command == "serve":
        import uvicorn

        from robot_ingest.viewer.app import create_app

        print(f"viewer: http://{args.host}:{args.port}")
        uvicorn.run(create_app(store.root), host=args.host, port=args.port, log_level="warning")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
