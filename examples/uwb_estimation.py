from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from drone_models.core import load_params
from drone_models.transform import motor_force2rotor_vel
from numpy.typing import NDArray

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.uwb import estimate_uwb_imu_state, reset_uwb_bias, simulate_imu, simulate_uwb


@dataclass(frozen=True)
class SensorConfig:
    """Sensor noise settings for one UWB+IMU trial."""

    range_std: float
    range_bias_max: float
    estimator_range_std: float
    accel_std: float
    gyro_std: float
    accel_bias_walk_std: float
    gyro_bias_walk_std: float


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


def configure_uwb_estimation(sim: Sim, sensors: SensorConfig):
    range_std = jnp.full_like(sim.data.uwb.range_std, sensors.range_std)
    range_bias_max = jnp.full_like(sim.data.uwb.range_bias_max, sensors.range_bias_max)
    estimator_range_std = jnp.full_like(
        sim.data.uwb.estimator_range_std, sensors.estimator_range_std
    )
    accel_std = jnp.full_like(sim.data.imu.accel_std, sensors.accel_std)
    gyro_std = jnp.full_like(sim.data.imu.gyro_std, sensors.gyro_std)
    accel_bias_walk_std = jnp.full_like(
        sim.data.imu.accel_bias_walk_std, sensors.accel_bias_walk_std
    )
    gyro_bias_walk_std = jnp.full_like(sim.data.imu.gyro_bias_walk_std, sensors.gyro_bias_walk_std)
    use_estimate = jnp.ones_like(sim.data.estimates.use_for_control)
    sim.data = sim.data.replace(
        uwb=sim.data.uwb.replace(
            range_std=range_std,
            range_bias_max=range_bias_max,
            estimator_range_std=estimator_range_std,
        ),
        imu=sim.data.imu.replace(
            accel_std=accel_std,
            gyro_std=gyro_std,
            accel_bias_walk_std=accel_bias_walk_std,
            gyro_bias_walk_std=gyro_bias_walk_std,
        ),
        estimates=sim.data.estimates.replace(use_for_control=use_estimate),
    )
    sim.default_data = sim.default_data.replace(
        uwb=sim.default_data.uwb.replace(
            range_std=range_std,
            range_bias_max=range_bias_max,
            estimator_range_std=estimator_range_std,
        ),
        imu=sim.default_data.imu.replace(
            accel_std=accel_std,
            gyro_std=gyro_std,
            accel_bias_walk_std=accel_bias_walk_std,
            gyro_bias_walk_std=gyro_bias_walk_std,
        ),
        estimates=sim.default_data.estimates.replace(use_for_control=use_estimate),
    )

    # Insert UWB/IMU sensing and estimation before the controller consumes feedback state.
    if reset_uwb_bias not in sim.reset_pipeline:
        sim.reset_pipeline += (reset_uwb_bias,)
    sensor_pipeline = (simulate_uwb, simulate_imu, estimate_uwb_imu_state)
    if sim.step_pipeline[: len(sensor_pipeline)] != sensor_pipeline:
        sim.step_pipeline = sensor_pipeline + sim.step_pipeline
    sim.build_reset_fn()
    sim.build_step_fn()
    sim.data = reset_uwb_bias(sim.data)


def set_initial_state(sim: Sim):
    cmd = figure_eight_control(0.0, sim.n_worlds, sim.n_drones)
    pos = jnp.asarray(cmd[..., :3], device=sim.device)
    vel = jnp.asarray(cmd[..., 3:6], device=sim.device)
    hover_thrust = -sim.data.params.mass * sim.data.params.gravity_vec[2] / 4
    params = load_params("first_principles", "cf21B_500")
    hover_rpm = motor_force2rotor_vel(hover_thrust, params["rpm2thrust"])
    rotor_vel = jnp.ones_like(sim.data.states.rotor_vel, device=sim.device) * hover_rpm
    states = sim.data.states.replace(pos=pos, vel=vel, rotor_vel=rotor_vel)
    imu = sim.data.imu.replace(prev_vel=vel)
    estimates = sim.data.estimates.replace(
        pos=pos,
        vel=vel,
        quat=states.quat,
        ang_vel=states.ang_vel,
        covariance=sim.data.estimates.covariance,
    )
    sim.data = sim.data.replace(states=states, estimates=estimates, imu=imu)
    sim.default_data = sim.default_data.replace(
        states=sim.default_data.states.replace(pos=pos, vel=vel, rotor_vel=rotor_vel),
        estimates=sim.default_data.estimates.replace(
            pos=pos,
            vel=vel,
            quat=states.quat,
            ang_vel=states.ang_vel,
            covariance=sim.default_data.estimates.covariance,
        ),
        imu=sim.default_data.imu.replace(prev_vel=vel),
    )


def run_trial(
    name: str, sensors: SensorConfig | None, render: bool = False, duration: float = 120.0
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
    if sensors is not None:
        configure_uwb_estimation(sim, sensors)

    time, truth, estimate, target = [], [], [], []
    steps_per_control = sim.freq // sim.control_freq
    for i in range(int(duration * sim.control_freq)):
        t = i / sim.control_freq
        cmd = figure_eight_control(t, sim.n_worlds, sim.n_drones)
        sim.state_control(cmd)
        sim.step(steps_per_control)
        truth_pos = np.asarray(jax.device_get(sim.data.states.pos[0, 0]))
        time.append(t)
        truth.append(truth_pos)
        if sensors is None:
            estimate.append(truth_pos)
        else:
            estimate.append(np.asarray(jax.device_get(sim.data.estimates.pos[0, 0])))
        target.append(cmd[0, 0, :3])

        if render:
            sim.render()

    sim.close()
    time = np.array(time)
    truth = np.array(truth)
    estimate = np.array(estimate)
    target = np.array(target)
    tracking_error = np.linalg.norm(truth - target, axis=-1)
    estimate_error = np.linalg.norm(estimate - truth, axis=-1)
    skip = sim.control_freq
    return {
        "name": name,
        "time": time,
        "truth": truth,
        "estimate": estimate,
        "target": target,
        "tracking_rms": np.sqrt(np.mean(tracking_error[skip:] ** 2)),
        "estimate_rms": np.sqrt(np.mean(estimate_error[skip:] ** 2)),
    }


def main(plot: bool = False, render: bool = False):
    # larger estimator range std for better closed-loop stability
    trials = [
        ("Perfect knowledge", None),
        ("UWB+IMU high quality", SensorConfig(0.01, 0.02, 0.02, 0.03, 0.002, 1e-5, 1e-6)),
        ("UWB+IMU low quality", SensorConfig(0.10, 0.15, 0.15, 0.03, 0.0025, 0.001, 0.00025)),
        # Abhi values: accel bias 1e-5, gyro bias 1e-6
    ]
    results = [run_trial(name, sensors, render=render) for name, sensors in trials]
    for result in results:
        print(
            f"{result['name']}: tracking RMS {result['tracking_rms']:.3f} m, "
            f"estimate RMS {result['estimate_rms']:.3f} m"
        )

    if plot:
        plot_results(results)


def plot_results(results: list[dict[str, NDArray]]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5))
    ax_traj = fig.add_subplot(1, 3, 1, projection="3d")
    ax_err = fig.add_subplot(1, 3, 2)
    ax_est_err = fig.add_subplot(1, 3, 3)
    target = results[0]["target"]
    ax_traj.plot(target[:, 0], target[:, 1], target[:, 2], color="k", label="target")

    xs = [target[:, 0]]
    ys = [target[:, 1]]
    zs = [target[:, 2]]
    for result in results:
        time = result["time"]
        truth = result["truth"]
        estimate = result["estimate"]
        ax_traj.plot(truth[:, 0], truth[:, 1], truth[:, 2], label=result["name"])
        err = np.linalg.norm(truth - target, axis=-1)
        ax_err.plot(time, err, label=result["name"])
        estimate_err = np.linalg.norm(estimate - truth, axis=-1)
        ax_est_err.plot(time, estimate_err, label=result["name"])
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
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [m]")
    ax_est_err.set_title("Position estimate error")
    ax_est_err.set_xlabel("Time [s]")
    ax_est_err.set_ylabel("Error [m]")
    ax_traj.legend()
    ax_err.legend()
    ax_est_err.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True, render=False)
