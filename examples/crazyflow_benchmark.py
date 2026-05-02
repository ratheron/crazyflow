import logging

import jax
import jax.numpy as jnp
from common import (
    CONTROL_FREQ,
    DRONE_RANGE,
    NUMBER,
    REPEAT,
    SIM_FREQ,
    WORLD_RANGE,
    benchmark_function,
    check_parameters,
    load_result_keys,
    run_isolated,
    run_sweep,
    save_results,
)
from numpy.typing import NDArray
from uwb_estimation import (
    SensorConfig,
    configure_uwb_estimation,
    figure_eight_control,
    set_initial_state,
)

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.sim import Sim

logger = logging.getLogger(__name__)

PIPELINES = ("baseline", "uwb_imu_ekf")
BENCHMARK_SENSOR_CONFIG = SensorConfig(0.01, 0.02, 0.02, 0.03, 0.002, 1e-5, 1e-6)


def benchmark_crazyflow(n_worlds: int, n_drones: int, device: str, pipeline: str) -> NDArray:
    """Benchmark Crazyflow simulation throughput.

    Args:
        n_worlds: Number of parallel worlds.
        n_drones: Number of drones per world.
        device: JAX device string ("cpu" or "gpu").
        pipeline: Benchmark variant, either baseline or UWB/IMU/EKF-augmented.

    Returns:
        Array of timing measurements (seconds per step).
    """
    logger.info(
        "Benchmarking crazyflow on %s with %s worlds, %s drones, pipeline=%s",
        device,
        n_worlds,
        n_drones,
        pipeline,
    )
    sim = Sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        device=device,
        control=Control.state,
        freq=SIM_FREQ,
        state_freq=CONTROL_FREQ,
        physics="first_principles",
        drone_model="cf21B_500",
        integrator="rk4",
    )
    set_initial_state(sim)
    if pipeline == "uwb_imu_ekf":
        configure_uwb_estimation(sim, BENCHMARK_SENSOR_CONFIG)

    sim_data = sim.data
    sim_step = sim.build_step_fn()
    sim_reset = sim.build_reset_fn()
    jax_device = jax.devices(device)[0]
    cmd = jnp.asarray(figure_eight_control(0.0, n_worlds, n_drones), device=jax_device)

    @jax.jit
    def fori_loop(data, cmd, num_steps: int):
        def single_step(_: int, data):
            data = F.state_control(data, cmd)
            return sim_step(data, sim.freq // sim.control_freq)

        return jax.lax.fori_loop(0, num_steps, single_step, data)

    def setup() -> None:
        nonlocal sim_data, cmd
        sim_data = sim_reset(sim_data, sim.default_data)
        jax.block_until_ready(fori_loop(sim_data, cmd, 1))  # Ensure JIT compiled dynamics

    def single_test_code() -> None:
        nonlocal sim_data, cmd
        jax.block_until_ready(fori_loop(sim_data, cmd, 1))

    def loop_test_code() -> None:
        nonlocal sim_data, cmd
        jax.block_until_ready(fori_loop(sim_data, cmd, NUMBER))

    return benchmark_function(
        setup, single_test_code, REPEAT, NUMBER, loop_test_code=loop_test_code
    )


def run_benchmarks(
    devices: list[str], world_range: list[int], drone_range: list[int], isolated: bool
) -> None:
    """Sweep over devices x world_range x drone_range and collect timing data.

    Each configuration is run in an isolated spawned process so that a crash (e.g. OOM)
    does not terminate the parent. Loads existing results from disk and skips already-completed
    configs, merging new results before saving.

    Args:
        devices: List of JAX device strings to benchmark (e.g. ["cpu", "gpu"]).
        world_range: List of world counts to sweep.
        drone_range: List of drone counts to sweep.
        isolated: Whether to run each configuration in an isolated spawned worker.
    """
    existing_keys = load_result_keys("crazyflow")
    configs = [
        cfg
        for cfg in (
            {"n_worlds": w, "n_drones": d, "device": dev, "pipeline": pipeline}
            for dev in devices
            for w in world_range
            for d in drone_range
            for pipeline in PIPELINES
        )
        if frozenset(cfg.items()) not in existing_keys
        and not check_parameters(cfg["n_worlds"], cfg["n_drones"], "crazyflow")
    ]
    benchmark_fn = (
        (lambda cfg: run_isolated(benchmark_crazyflow, cfg))
        if isolated
        else (lambda cfg: benchmark_crazyflow(**cfg))
    )
    data = run_sweep(configs, benchmark_fn)
    if not data:
        logger.info("No new benchmark data collected.")
        return
    save_results(data, "crazyflow")
    logger.info("Saved results.")


def main(
    device: str | None = None,
    n_worlds: int | None = None,
    n_drones: int | None = None,
    sweep: bool = False,
    isolated: bool | None = None,
) -> None:
    """Run Crazyflow benchmark.

    Args:
        device: JAX device to benchmark ("cpu" or "gpu").
        n_worlds: Single world count to benchmark, or None to use the default quick case.
        n_drones: Single drone count to benchmark, or None to use DRONE_RANGE.
        sweep: If True and n_worlds is None, benchmark the full WORLD_RANGE sweep.
        isolated: Whether to benchmark each config in a spawned worker. Defaults to True only for
            direct CLI execution.
    """
    world_range = (
        [n_worlds] if n_worlds is not None else (WORLD_RANGE if sweep else [WORLD_RANGE[0]])
    )
    drone_range = [n_drones] if n_drones is not None else DRONE_RANGE
    devices = [device] if device is not None else ["cpu"]
    if isolated is None:
        isolated = benchmark_crazyflow.__module__ == "__main__"
    run_benchmarks(
        devices=devices, world_range=world_range, drone_range=drone_range, isolated=isolated
    )


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
