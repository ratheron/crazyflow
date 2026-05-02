from __future__ import annotations

import json
import logging
import multiprocessing
import timeit
from pathlib import Path
from queue import Empty
from typing import Callable

import numpy as np
from numpy.typing import NDArray

TIMEOUT = 600  # seconds
CONTROL_FREQ = 500  # Hz
SIM_FREQ = 500  # Hz
WORLD_RANGE = [2**i for i in range(0, 17, 2)]  # number of parallel environments
DRONE_RANGE = [1]
RESOLUTIONS: list[tuple[int, int]] = [(64, 64)]  # render benchmark resolutions (width, height)
REPEAT = 50  # number of repetitions for benchmarking
NUMBER = 50  # number of executions per repetition

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parents[1] / "results/performance"


class WorkerError(Exception):
    """Raised when an isolated worker process fails, times out, or produces no result."""


def _isolated_target(fn: Callable, config: dict, queue: multiprocessing.Queue) -> None:
    """Top-level worker target (must be module-level to be picklable with spawn)."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(fn.__module__).setLevel(logging.INFO)
    queue.put(fn(**config))


def run_isolated(fn: Callable, config: dict) -> NDArray:
    """Run ``fn(**config)`` in an isolated spawned process and return the result.

    Uses ``multiprocessing.get_context("spawn")`` so each call gets a fresh
    interpreter — safe for libraries (Isaac Gym, JAX, PyBullet) that allow only
    one instance per process.

    Args:
        fn: Module-level callable that accepts ``**config`` and returns an NDArray.
        config: Keyword arguments forwarded to *fn*.

    Returns:
        The NDArray returned by *fn* inside the worker process.

    Raises:
        WorkerError: On timeout, non-zero exit code, or missing result.
    """
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_isolated_target, args=(fn, config, queue))
    proc.start()
    proc.join(timeout=TIMEOUT)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise WorkerError(f"timed out after {TIMEOUT}s for {config}")
    if proc.exitcode != 0:
        raise WorkerError(f"exit code {proc.exitcode} for {config}")
    try:
        return queue.get_nowait()
    except Empty:
        raise WorkerError(f"no result returned for {config}")


def _is_dominated(config: dict, exhausted: list[dict]) -> bool:
    """Return True if config is dominated by any exhausted config.

    A config is dominated when it runs on the same device, has at least as many
    worlds and drones as the exhausted config, and all other keys match exactly.
    """
    for e in exhausted:
        if config.get("device") != e.get("device"):
            continue
        if config.get("n_worlds", 0) < e.get("n_worlds", 0):
            continue
        if config.get("n_drones", 0) < e.get("n_drones", 0):
            continue
        other_keys = set(e.keys()) - {"n_worlds", "n_drones", "device"}
        if all(config.get(k) == e[k] for k in other_keys) or not other_keys:
            return True
    return False


def benchmark_function(
    setup_code: Callable,
    single_test_code: Callable,
    R: int,
    N: int,
    loop_test_code: Callable = None,
) -> NDArray:
    """Run benchmark with a timeout guard.

    Optionally use a custom loop_test_code for repeated steps.

    Args:
        setup_code: Callable executed before each repeat (warmup / JIT trigger).
        single_test_code: Callable whose wall-clock time is measured (single step).
        R: Number of repeats (outer loop).
        N: Number of executions per repeat (inner loop).
        timeout: Abort if a single run would cause total time to exceed this (seconds).
        loop_test_code: Optional. If provided, used for timing test (N steps in one call).

    Returns:
        Array of per-execution wall-clock times (seconds). Empty if timeout exceeded.
    """
    setup_code()
    if loop_test_code is not None:
        timer = timeit.Timer(stmt=loop_test_code, setup=setup_code)
        return np.array(timer.repeat(repeat=R, number=1)) / N
    timer = timeit.Timer(stmt=single_test_code, setup=setup_code)
    return np.array(timer.repeat(repeat=R, number=N)) / N


def run_sweep(configs: list[dict], benchmark_fn: Callable[[dict], NDArray]) -> list[dict]:
    """Iterate pre-generated configs and collect benchmark timings.

    Calls ``benchmark_fn(config)`` for each config dict. On resource exhaustion
    (caught exception or empty timings), records the config and skips any future
    config dominated by it — same device, at least as many worlds and drones, and
    identical values for all other keys.

    Args:
        configs: Ordered list of config dicts. Each dict is stored (with a
            ``"timings"`` key added) in the returned results.
        benchmark_fn: Callable that receives a config dict and returns per-step
            timing measurements. An empty array signals self-detected infeasibility.

    Returns:
        List of ``{**config, "timings": list}`` dicts for every config that
        produced at least one timing measurement.
    """
    results = []
    exhausted: list[dict] = []
    for config in configs:
        if _is_dominated(config, exhausted):
            logger.info("Skipping config %s due to prior resource exhaustion.", config)
            continue
        try:
            timings = benchmark_fn(config)
        except WorkerError as e:
            logger.warning(f"Error for config {config}, skipping dominated configs.")
            logger.warning(f"Exception: {e}")
            exhausted.append(config)
            continue
        if len(timings) > 0:
            results.append({**config, "timings": timings.tolist()})
        else:
            exhausted.append(config)
    return results


def _result_key(row: dict, key_cols: list[str]) -> frozenset:
    """Return a hashable key for one result row."""
    return frozenset((k, row[k]) for k in key_cols)


def load_result_keys(name: str) -> set[frozenset]:
    """Return the set of existing result keys from a JSON file for skip-checking.

    Each key is a ``frozenset`` of ``(column, value)`` pairs covering all non-timings
    columns, so it is independent of column order.

    Args:
        name: Benchmark name used as the result filename stem (e.g. ``"crazyflow"``).

    Returns:
        Set of frozensets, one per existing row. Empty if the file does not exist.
    """
    path = _RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return set()
    with path.open() as f:
        rows = json.load(f)
    if not rows:
        return set()
    key_cols = [c for c in rows[0] if c != "timings"]
    return {_result_key(row, key_cols) for row in rows}


def save_results(rows: list[dict], name: str) -> None:
    """Save benchmark results, merging with (not overwriting) any existing JSON file.

    Rows in the existing file whose keys match a row in *rows* are replaced; all
    other existing rows are preserved.

    Args:
        rows: Benchmark results as dictionaries.
        name: Benchmark name used as the result filename stem (e.g. ``"crazyflow"``).
    """
    if not rows:
        return
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _RESULTS_DIR / f"{name}.json"
    key_cols = [c for c in rows[0] if c != "timings"]
    if path.exists():
        with path.open() as f:
            existing = json.load(f)
        new_keys = {_result_key(row, key_cols) for row in rows}
        existing = [row for row in existing if _result_key(row, key_cols) not in new_keys]
        rows = existing + rows
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def check_parameters(n_worlds: int, n_drones: int, framework: str) -> bool:
    """Check if the given combination of parameters is valid for benchmarking."""
    if framework == "gym-pybullet-drones":
        return False
    elif framework == "crazyflow":
        return False
    elif framework == "diff-aero":
        return False
    elif framework == "aerial-gym":
        # Aerial-gym uses single drone per environment
        return n_drones > 1
    elif framework == "flightning":
        # Flightning uses single drone per environment (VecEnv vmaps over worlds)
        return n_drones > 1
    elif framework == "crazyflow_grad":
        return False
    elif framework == "diff_aero_grad":
        return False
    elif framework == "flightning_grad":
        # Flightning uses single drone per environment (VecEnv vmaps over worlds)
        return n_drones > 1
    raise ValueError(f"Unknown framework: {framework}")
