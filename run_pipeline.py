#!/usr/bin/env python3
"""
Run the full where-to-live pipeline end-to-end.

Usage:
    uv run python run_pipeline.py            # run all steps
    uv run python run_pipeline.py --from 3   # resume from step 3
    uv run python run_pipeline.py --only 6   # run only step 6
"""

import argparse
import sys
import time
from pathlib import Path

import logging
from pipeline.log import setup as setup_logging

logger = logging.getLogger(__name__)


STEPS = [
    (1, "Geographic pre-filter (block group lat bands + population density)",
     "pipeline.01_geo_filter"),
    (2, "Airport proximity filter (60-min drive isochrones)",
     "pipeline.02_airport_proximity"),
    (3, "Walk Score + Bike Score (scrape walkscore.com for all candidates)",
     "pipeline.03_walkscore"),
    (4, "OSM amenities + affordability data (Census ACS / Zillow ZHVI)",
     "pipeline.04_osm_amenities"),
    (5, "Composite scoring and ranking",
     "pipeline.05_score_and_rank"),
    (6, "Interactive maps (folium HTML)",
     "pipeline.06_visualize"),
]


def run_step(number: int, label: str, module_path: str) -> None:
    import importlib
    logger.info(f"{'='*50}")
    logger.info(f"STEP {number}: {label}")
    logger.info(f"{'='*50}")
    t0 = time.time()
    mod = importlib.import_module(module_path)
    mod.run()
    elapsed = time.time() - t0
    logger.info(f"Step {number} complete in {elapsed:.1f}s")


def main() -> None:
    setup_logging(log_file=Path(__file__).parent / "pipeline.log")

    parser = argparse.ArgumentParser(description="Run the where-to-live pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="from_step", type=int, metavar="N",
                       help="Start from step N (skip earlier steps)")
    group.add_argument("--only", type=int, metavar="N",
                       help="Run only step N")
    args = parser.parse_args()

    if args.only:
        steps_to_run = [s for s in STEPS if s[0] == args.only]
        if not steps_to_run:
            logger.error(f"Unknown step: {args.only}. Valid steps: 1–{len(STEPS)}")
            sys.exit(1)
    elif args.from_step:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]
    else:
        steps_to_run = STEPS

    logger.info(f"Running {len(steps_to_run)} pipeline step(s)…")

    total_start = time.time()
    for number, label, module in steps_to_run:
        run_step(number, label, module)

    total = time.time() - total_start
    logger.info(f"Pipeline complete in {total/60:.1f} minutes.")
    logger.info("Outputs: outputs/north_candidates.csv | outputs/south_candidates.csv | outputs/maps/")


if __name__ == "__main__":
    main()
