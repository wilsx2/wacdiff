#!/usr/bin/env python3

"""
Plots we are interested in generating

strong scaling, nodes vs cpus (line plot w/ marks, incl. theoretical linear ideal)
    independent: nodes/cpus
    dependent: total runtime compared with N=1/C=1 (efficiency)
weak scaling, nodes (line plot w/ marks, incl. theoretical ideal flat line)
    independent: nodes/cpus
    dependent: total runtime compared with N=1/C=1 (efficiency)
io scaling (line plot w/ marks, incl. ideal flat line)
    independent: nodes
    dependent: proportion of runtime compared with N=1 (efficiency)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Analysis

RUN_DIR_RE = re.compile(
    r"^wacdiff-"
    r"N(?P<nodes>\d+)-"
    r"C(?P<cpus_per_node>\d+)-"
    r"S(?P<size>\d+)-"
    r"IO(?P<io_frequency>\d+)"
    r"(?:-(?P<job_id>.+))?$"
)

PHASE_TAGS = ("computation", "communication", "io")


def parse_timing_xml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    def element_text(tag: str) -> str:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
        if match is None:
            raise ValueError(f"<{tag}> not found in {path}")
        return match.group(1).strip()

    units = element_text("units")
    timings = {tag: float(element_text(tag)) for tag in PHASE_TAGS}
    timings["units"] = units
    return timings


def parse_run_dir(name: str) -> dict[str, Any] | None:
    match = RUN_DIR_RE.match(name)
    if match is None:
        return None

    groups = match.groupdict()
    return {
        "nodes": int(groups["nodes"]),
        "cpus_per_node": int(groups["cpus_per_node"]),
        "size": int(groups["size"]),
        "io_frequency": int(groups["io_frequency"]),
        "job_id": groups["job_id"],
    }


def load_timings(outputs_dir: Path | str) -> pd.DataFrame:
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        raise NotADirectoryError(f"outputs directory not found: {outputs_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(outputs_dir.rglob("*")):
        if not run_dir.is_dir():
            continue

        params = parse_run_dir(run_dir.name)
        if params is None:
            continue

        timing_path = run_dir / "timing.xml"
        if not timing_path.is_file():
            print(f"warning: no timing.xml in {run_dir}; skipping", file=sys.stderr)
            continue

        timing = parse_timing_xml(timing_path)
        units = timing.pop("units")
        total_time = sum(timing.values())

        sweep = run_dir.parent.name if run_dir.parent != outputs_dir else None

        rows.append(
            {
                "sweep": sweep,
                "run": run_dir.name,
                **params,
                "units": units,
                "computation_time": timing["computation"],
                "communication_time": timing["communication"],
                "io_time": timing["io"],
                "total_time": total_time,
            }
        )

    if not rows:
        print(
            f"warning: no wacdiff run directories found under {outputs_dir}",
            file=sys.stderr,
        )

    columns = [
        "sweep",
        "run",
        "nodes",
        "cpus_per_node",
        "size",
        "io_frequency",
        "job_id",
        "units",
        "computation_time",
        "communication_time",
        "io_time",
        "total_time",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(
            [
                "sweep",
                "nodes",
                "cpus_per_node",
                "size",
                "io_frequency",
                "job_id",
            ],
            ignore_index=True,
        )
    return df

# gnuplot
