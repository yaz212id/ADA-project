"""Run the whole analysis: every stage notebook, in order, each in a fresh kernel.

This file is the single entry point of the project. It contains **no analysis**:
only the list of stage notebooks and the loop that executes them. Each stage
lives in its own notebook under notebooks/ and must run from top to bottom in a
fresh kernel, as required by the course rules.

    python run_all.py              run every stage in order
    python run_all.py --only 02    run one stage (match by its number or name)
    python run_all.py --list       show the stages without running anything
    python run_all.py --timeout 1200   raise the per-cell time limit

The script stops at the first stage that fails, names the notebook and the cell,
and prints the running time of every stage that did run, so the figures can be
copied into the README.

Written with the help of Claude (Anthropic), which was asked to produce the
pipeline runner for this project (declared in the paper appendix).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"

# The stages of the project, in the order they must run. Add a new notebook here
# once its draft runs clean from top to bottom (see the drafts/ workflow in the
# README). Nothing else in this file needs to change.
STAGES: list[str] = [
    "01_download_data.ipynb",
    "02_clean_data.ipynb",
    "03_features.ipynb",
    # "04_models.ipynb",
    # "05_backtest.ipynb",
    # "06_figures_tables.ipynb",
]

DEFAULT_TIMEOUT = 900  # seconds allowed for a single cell


def stage_paths(only: str | None = None) -> list[Path]:
    """The notebooks to run, as full paths, in order.

    `only` selects one stage by any unambiguous part of its name, so
    `--only 02`, `--only 02_clean` and `--only 02_clean_data.ipynb` all work.
    """
    paths = [NOTEBOOKS_DIR / name for name in STAGES]
    if only is None:
        return paths

    matches = [p for p in paths if only in p.name]
    if not matches:
        raise SystemExit(
            f"No stage matches {only!r}. Known stages:\n  "
            + "\n  ".join(STAGES)
        )
    if len(matches) > 1:
        raise SystemExit(
            f"{only!r} matches more than one stage:\n  "
            + "\n  ".join(p.name for p in matches)
        )
    return matches


def run_notebook(path: Path, timeout: int = DEFAULT_TIMEOUT) -> float:
    """Execute one notebook in a fresh kernel, in place. Returns the seconds taken.

    A new kernel is started for each notebook, so nothing carries over between
    stages: a notebook that only works because an earlier one left a variable
    behind will fail here, which is the point.
    """
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        # run as if the kernel had been started inside notebooks/, like Jupyter
        resources={"metadata": {"path": str(path.parent)}},
    )
    started = time.perf_counter()
    try:
        client.execute()
    finally:
        # keep whatever did run, so a failure can be inspected in the notebook
        nbformat.write(notebook, path)
    return time.perf_counter() - started


def describe_failure(path: Path, error: Exception) -> str:
    """A short, readable report of which cell failed and why."""
    from nbclient.exceptions import CellExecutionError

    if not isinstance(error, CellExecutionError):
        return f"{path.name} failed: {type(error).__name__}: {error}"

    message = str(error)
    # CellExecutionError's text ends with the traceback; the last line is the useful part
    lines = [line for line in message.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else "no error message"
    return f"{path.name} failed in a cell:\n    {last}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the stage notebooks of the project, each in a fresh kernel."
    )
    parser.add_argument("--only", metavar="STAGE",
                        help="run a single stage, e.g. --only 02")
    parser.add_argument("--list", action="store_true",
                        help="list the stages and exit")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"seconds allowed per cell (default {DEFAULT_TIMEOUT})")
    args = parser.parse_args(argv)

    if args.list:
        print("Stages, in order:")
        for i, name in enumerate(STAGES, start=1):
            exists = "" if (NOTEBOOKS_DIR / name).exists() else "   [MISSING]"
            print(f"  {i}. {name}{exists}")
        return 0

    paths = stage_paths(args.only)
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("These notebooks are listed in STAGES but do not exist:", file=sys.stderr)
        for p in missing:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        return 1

    print(f"Running {len(paths)} stage(s) from {NOTEBOOKS_DIR.relative_to(ROOT)}/\n")
    timings: list[tuple[str, float]] = []
    for path in paths:
        print(f"--> {path.name} ... ", end="", flush=True)
        try:
            seconds = run_notebook(path, timeout=args.timeout)
        except Exception as exc:
            print("FAILED\n")
            print(describe_failure(path, exc), file=sys.stderr)
            if timings:
                print("\nStages that completed before the failure:", file=sys.stderr)
                for name, took in timings:
                    print(f"  {name}: {took:.1f} s", file=sys.stderr)
            print(f"\nPipeline stopped at {path.name}.", file=sys.stderr)
            return 1
        timings.append((path.name, seconds))
        print(f"ok  ({seconds:.1f} s)")

    print("\nRunning time per stage:")
    for name, took in timings:
        print(f"  {name:28s} {took:6.1f} s")
    print(f"  {'TOTAL':28s} {sum(t for _, t in timings):6.1f} s")
    print("\nAll stages ran successfully in fresh kernels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
