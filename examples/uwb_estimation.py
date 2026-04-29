from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from drone_models.core import load_params
from drone_models.transform import motor_force2rotor_vel

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.uwb import estimate_uwb_state, simulate_uwb

if TYPE_CHECKING:
    from numpy.typing import NDArray


def figure_eight_control(t: float, n_worlds: int, n_drones: int) -> NDArray:
    period = 10.0
    omega = 2.0 * np.pi / period
    x_amp, y_amp, z = 1.5, 1.0, 1.5

    cmd = np.zeros((n_worlds, n_drones, 13))
    # pos
    cmd[..., 0] = x_amp * np.sin(omega * t)
    cmd[..., 1] = y_amp * np.sin(2.0 * omega * t)
    cmd[..., 2] = z
    # vel
    cmd[..., 3] = x_amp * omega * np.cos(omega * t)
    cmd[..., 4] = 2.0 * y_amp * omega * np.cos(2.0 * omega * t)
    # acc
    cmd[..., 6] = -x_amp * omega**2 * np.sin(omega * t)
    cmd[..., 7] = -4.0 * y_amp * omega**2 * np.sin(2.0 * omega * t)
    return cmd


def configure_uwb_estimation(sim: Sim, range_std: float):
    range_std_arr = jnp.full_like(sim.data.uwb.range_std, range_std)
    use_estimate = jnp.ones_like(sim.data.estimates.use_for_control)
    sim.data = sim.data.replace(
        uwb=sim.data.uwb.replace(range_std=range_std_arr),
        estimates=sim.data.estimates.replace(use_for_control=use_estimate),
    )

    # Insert UWB sensing and estimation before the controller consumes feedback state.
    sim.step_pipeline = (simulate_uwb, estimate_uwb_state) + sim.step_pipeline
    sim.build_step_fn()


def set_initial_state(sim: Sim):
    cmd = figure_eight_control(0.0, sim.n_worlds, sim.n_drones)
    pos = jnp.asarray(cmd[..., :3], device=sim.device)
    vel = jnp.asarray(cmd[..., 3:6], device=sim.device)
    hover_thrust = -sim.data.params.mass * sim.data.params.gravity_vec[2] / 4
    params = load_params("first_principles", "cf21B_500")
    hover_rpm = motor_force2rotor_vel(hover_thrust, params["rpm2thrust"])
    rotor_vel = jnp.ones_like(sim.data.states.rotor_vel, device=sim.device) * hover_rpm
    states = sim.data.states.replace(pos=pos, vel=vel, rotor_vel=rotor_vel)
    estimates = sim.data.estimates.replace(
        pos=pos,
        vel=vel,
        quat=states.quat,
        ang_vel=states.ang_vel,
        covariance=sim.data.estimates.covariance * 0.01,
    )
    sim.data = sim.data.replace(states=states, estimates=estimates)


def run_trial(
    name: str, range_std: float | None, render: bool = False, duration: float = 20.0
) -> dict[str, NDArray]:
    sim = Sim(
        control=Control.state,
        physics="first_principles",
        drone_model="cf21B_500",  # cf21B_500, cf2x_L250
        integrator="rk4",
        freq=500,
        state_freq=100,
        rng_key=42,
    )
    set_initial_state(sim)
    if range_std is not None:
        configure_uwb_estimation(sim, range_std)

    truth, estimate, target = [], [], []
    steps_per_control = sim.freq // sim.control_freq
    for i in range(int(duration * sim.control_freq)):
        t = i / sim.control_freq
        cmd = figure_eight_control(t, sim.n_worlds, sim.n_drones)
        sim.state_control(cmd)
        sim.step(steps_per_control)
        truth_pos = np.asarray(jax.device_get(sim.data.states.pos[0, 0]))
        truth.append(truth_pos)
        if range_std is None:
            estimate.append(truth_pos)
        else:
            estimate.append(np.asarray(jax.device_get(sim.data.estimates.pos[0, 0])))
        target.append(cmd[0, 0, :3])

        if render:
            sim.render()

    sim.close()
    truth = np.array(truth)
    estimate = np.array(estimate)
    target = np.array(target)
    tracking_error = np.linalg.norm(truth - target, axis=-1)
    estimate_error = np.linalg.norm(estimate - truth, axis=-1)
    skip = sim.control_freq
    return {
        "name": name,
        "truth": truth,
        "estimate": estimate,
        "target": target,
        "tracking_rms": np.sqrt(np.mean(tracking_error[skip:] ** 2)),
        "estimate_rms": np.sqrt(np.mean(estimate_error[skip:] ** 2)),
    }


def main(plot: bool = False, render: bool = False):
    trials = [("Perfect knowledge", None), ("UWB high quality", 0.02), ("UWB low quality", 0.30)]
    results = [run_trial(name, range_std, render=render) for name, range_std in trials]
    for result in results:
        print(
            f"{result['name']}: tracking RMS {result['tracking_rms']:.3f} m, "
            f"estimate RMS {result['estimate_rms']:.3f} m"
        )

    if plot:
        plot_results(results)


def plot_results(results: list[dict[str, NDArray]]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 5))
    ax_traj = fig.add_subplot(1, 2, 1, projection="3d")
    ax_err = fig.add_subplot(1, 2, 2)
    target = results[0]["target"]
    ax_traj.plot(target[:, 0], target[:, 1], target[:, 2], color="k", label="target")

    xs = [target[:, 0]]
    ys = [target[:, 1]]
    zs = [target[:, 2]]
    for result in results:
        truth = result["truth"]
        ax_traj.plot(truth[:, 0], truth[:, 1], truth[:, 2], label=result["name"])
        err = np.linalg.norm(truth - target, axis=-1)
        ax_err.plot(err, label=result["name"])
        xs.append(truth[:, 0])
        ys.append(truth[:, 1])
        zs.append(truth[:, 2])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()

    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half = 0.5 * max_range

    # Make x, y, z axes use the same scale by setting equal ranges centered on their midpoints.
    ax_traj.set_xlim(x_mid - half, x_mid + half)
    ax_traj.set_ylim(y_mid - half, y_mid + half)
    ax_traj.set_zlim(z_mid - half, z_mid + half)

    ax_traj.set_title("Figure-eight tracking")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_zlabel("z [m]")
    ax_err.set_title("Tracking error")
    ax_err.set_xlabel("Control step")
    ax_err.set_ylabel("Error [m]")
    ax_traj.legend()
    ax_err.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True, render=False)
