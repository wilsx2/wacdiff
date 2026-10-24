#!/usr/bin/env python3

"""
Plots we are interested in generating

strong scaling, nodes/cpus (line plot w/ marks, incl. theoretical linear ideal / efficiency flatline)
    independent: nodes/cpus
    dependent: total runtime compared with N=1/C=1 (speedup and efficiency)
weak scaling, nodes/cpus (line plot w/ marks, incl. theoretical linear ideal / efficiency flatline)
    independent: nodes/cpus
    dependent: total runtime compared with N=1/C=1 (speedup and efficiency)
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

# Plot generation

SWEEPS = (
    "strong_scaling_node",
    "strong_scaling_cpu",
    "weak_scaling_node",
    "weak_scaling_cpu",
    "io_scaling",
)

GROUP_COLUMNS = ["nodes", "cpus_per_node", "size", "io_frequency"]


def aggregate_sweep(df: pd.DataFrame, sweep: str) -> pd.DataFrame | None:
    """Average repeated runs of the same sweep configuration."""
    sweep_df = df[df["sweep"] == sweep]
    if sweep_df.empty:
        return None

    grouped = sweep_df.groupby(GROUP_COLUMNS, as_index=False).agg(
        n_runs=("run", "count"),
        total_time=("total_time", "mean"),
        total_time_std=("total_time", "std"),
        computation_time=("computation_time", "mean"),
        communication_time=("communication_time", "mean"),
        io_time=("io_time", "mean"),
    )
    grouped["total_time_std"] = grouped["total_time_std"].fillna(0.0)
    grouped["total_cores"] = grouped["nodes"] * grouped["cpus_per_node"]
    grouped = grouped.sort_values(
        ["nodes", "cpus_per_node", "size", "io_frequency"]
    ).reset_index(drop=True)
    return grouped


def _baseline_index(grouped: pd.DataFrame, x_col: str) -> int:
    return int(grouped[x_col].idxmin())


def build_strong_data(grouped: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Data for strong scaling: speedup and efficiency vs. scaling factor."""
    x = grouped[x_col].astype(float)
    base_idx = _baseline_index(grouped, x_col)
    base_total = float(grouped.loc[base_idx, "total_time"])

    speedup = base_total / grouped["total_time"]
    efficiency = speedup / x

    return pd.DataFrame(
        {
            "scaling_factor": x,
            "total_cores": grouped["total_cores"],
            "size": grouped["size"],
            "total_time": grouped["total_time"],
            "total_time_std": grouped["total_time_std"],
            "speedup": speedup,
            "efficiency": efficiency,
        }
    )


def build_weak_data(grouped: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Data for weak scaling: efficiency vs. scaling factor (ideal is flat at 1)."""
    x = grouped[x_col].astype(float)
    base_idx = _baseline_index(grouped, x_col)
    base_total = float(grouped.loc[base_idx, "total_time"])

    efficiency = base_total / grouped["total_time"]

    return pd.DataFrame(
        {
            "scaling_factor": x,
            "total_cores": grouped["total_cores"],
            "size": grouped["size"],
            "total_time": grouped["total_time"],
            "total_time_std": grouped["total_time_std"],
            "efficiency": efficiency,
        }
    )


def build_io_data(grouped: pd.DataFrame) -> pd.DataFrame:
    """Data for I/O scaling: I/O fraction and efficiency vs. nodes."""
    nodes = grouped["nodes"].astype(float)
    total = grouped["total_time"].astype(float)
    io = grouped["io_time"].astype(float)

    base_idx = _baseline_index(grouped, "nodes")
    base_io_fraction = float(io.loc[base_idx] / total.loc[base_idx])

    io_fraction = io / total
    if base_io_fraction > 0.0:
        io_fraction_ratio = io_fraction / base_io_fraction
        io_efficiency = base_io_fraction / io_fraction
    else:
        io_fraction_ratio = pd.Series(float("nan"), index=grouped.index)
        io_efficiency = pd.Series(float("nan"), index=grouped.index)

    return pd.DataFrame(
        {
            "nodes": nodes,
            "total_cores": grouped["total_cores"],
            "size": grouped["size"],
            "io_frequency": grouped["io_frequency"],
            "total_time": total,
            "total_time_std": grouped["total_time_std"],
            "io_time": io,
            "io_fraction": io_fraction,
            "io_fraction_ratio": io_fraction_ratio,
            "io_efficiency": io_efficiency,
        }
    )


def write_data_file(plot_dir: Path, sweep: str, data: pd.DataFrame) -> Path:
    header = [
        f"# {sweep}",
        f"# columns: {' '.join(str(column) for column in data.columns)}",
    ]
    body = data.to_csv(None, sep=" ", index=False, header=False, float_format="%.6f")
    path = plot_dir / f"{sweep}.dat"
    path.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
    return path


def write_strong_gp(plot_dir: Path, sweep: str, xlabel: str) -> Path:
    title = "Strong scaling (nodes)" if sweep.endswith("_node") else "Strong scaling (CPUs)"
    script = f"""set terminal pngcairo size 1200,900 enhanced font 'Verdana,12'
set output '{sweep}.png'

set title '{title}' font ',16'
set xlabel '{xlabel}' font ',12'
set ylabel 'Efficiency (T_1 / (p T_p))' font ',12'
set y2label 'Speedup (T_1 / T_p)' font ',12'
set y2tics
set grid
set key top right
set logscale x 2
set xrange [1:64]
set xtics (1,2,4,8,16,32,64)
set yrange [0:1.2]
set y2range [0:70]

plot '{sweep}.dat' using 1:7 axes x1y1 with linespoints linewidth 2 pointtype 7 pointsize 1.2 title 'Efficiency', 1 axes x1y1 with lines dashtype 2 linewidth 1.5 title 'Ideal efficiency', '{sweep}.dat' using 1:6 axes x1y2 with linespoints linewidth 2 pointtype 6 pointsize 1.2 title 'Speedup', x axes x1y2 with lines dashtype 2 linewidth 1.5 title 'Ideal speedup'
"""
    path = plot_dir / f"{sweep}.gp"
    path.write_text(script, encoding="utf-8")
    return path


def write_weak_gp(plot_dir: Path, sweep: str, xlabel: str) -> Path:
    title = "Weak scaling (nodes)" if sweep.endswith("_node") else "Weak scaling (CPUs)"
    script = f"""set terminal pngcairo size 1200,900 enhanced font 'Verdana,12'
set output '{sweep}.png'

set title '{title}' font ',16'
set xlabel '{xlabel}' font ',12'
set ylabel 'Efficiency (T_1 / T_p)' font ',12'
set grid
set key top left
set logscale x 2
set xrange [1:64]
set xtics (1,2,4,8,16,32,64)
set yrange [0:1.5]

plot '{sweep}.dat' using 1:6 with linespoints linewidth 2 pointtype 7 pointsize 1.2 title 'Measured efficiency', 1 with lines dashtype 2 linewidth 1.5 title 'Ideal efficiency'
"""
    path = plot_dir / f"{sweep}.gp"
    path.write_text(script, encoding="utf-8")
    return path


def write_io_gp(plot_dir: Path, sweep: str) -> Path:
    script = f"""set terminal pngcairo size 1200,900 enhanced font 'Verdana,12'
set output '{sweep}.png'

set title 'I/O scaling' font ',16'
set xlabel 'Nodes' font ',12'
set ylabel 'I/O efficiency (fraction at N=1 / fraction at N)' font ',12'
set grid
set key top right
set logscale x 2
set xrange [1:64]
set xtics (1,2,4,8,16,32,64)
set yrange [0:1.2]

plot '{sweep}.dat' using 1:9 with linespoints linewidth 2 pointtype 7 pointsize 1.2 title 'I/O efficiency', 1 with lines dashtype 2 linewidth 1.5 title 'Ideal (flat)'
"""
    path = plot_dir / f"{sweep}.gp"
    path.write_text(script, encoding="utf-8")
    return path


def generate_plots(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    written: list[Path] = []

    for sweep in SWEEPS:
        grouped = aggregate_sweep(df, sweep)
        if grouped is None:
            print(f"warning: no runs for sweep '{sweep}'; skipping", file=sys.stderr)
            continue

        plot_dir = plots_dir / sweep
        plot_dir.mkdir(parents=True, exist_ok=True)

        if sweep.startswith("strong"):
            x_col = "nodes" if sweep.endswith("_node") else "cpus_per_node"
            xlabel = "Nodes" if sweep.endswith("_node") else "CPUs per node"
            data = build_strong_data(grouped, x_col)
            written.append(write_data_file(plot_dir, sweep, data))
            written.append(write_strong_gp(plot_dir, sweep, xlabel))
        elif sweep.startswith("weak"):
            x_col = "nodes" if sweep.endswith("_node") else "cpus_per_node"
            xlabel = "Nodes" if sweep.endswith("_node") else "CPUs per node"
            data = build_weak_data(grouped, x_col)
            written.append(write_data_file(plot_dir, sweep, data))
            written.append(write_weak_gp(plot_dir, sweep, xlabel))
        elif sweep == "io_scaling":
            data = build_io_data(grouped)
            written.append(write_data_file(plot_dir, sweep, data))
            written.append(write_io_gp(plot_dir, sweep))

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate gnuplot data files and scripts from wacdiff timing XML."
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="directory containing wacdiff sweep output directories (default: outputs)",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="directory in which to write plots/{sweep}/{{.dat,.gp}} files (default: plots)",
    )
    args = parser.parse_args(argv)

    try:
        df = load_timings(args.outputs_dir)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print(
            f"error: no timing data found under {args.outputs_dir}",
            file=sys.stderr,
        )
        return 1

    written = generate_plots(df, Path(args.plots_dir))
    for path in written:
        print(f"wrote {path}")
    print(f"wrote {len(written)} plot files under {args.plots_dir}")
    print(
        "render a plot with: "
        f"(cd {args.plots_dir}/<sweep> && gnuplot <sweep>.gp)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
