import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_beam(beam, w_range, M_range, trail_positions):
    """Create an animation of the beam deflection and moment."""
    print("Computing full Newmark solution...")
    # Precompute w(x, t) over all x_vals and all time steps
    w_sol = beam.calculate_w(beam.x_vals)
    print("Solution computed.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Displacement plot
    line_disp, = ax1.plot([], [], lw=2, label="Beam Deflection")
    force_line_disp, = ax1.plot([], [], 'r--', lw=1.5, label="Force location")
    # Trailing force lines
    trail_lines_disp = [ax1.plot([], [], 'g--', lw=1)[0] for _ in trail_positions]


    ax1.set_xlim(0, beam.L)
    ax1.set_ylim(w_range[0] * 1.2, w_range[1] * 1.2)
    ax1.set_ylabel("Deflection $w(x,t)$")
    ax1.grid(True)
    ax1.legend()

    # Moment plot
    line_moment, = ax2.plot([], [], lw=2, color='orange', label="Bending Moment")
    force_line_moment, = ax2.plot([], [], 'r--', lw=1.5)
    trail_lines_moment = [ax2.plot([], [], 'g--', lw=1)[0] for _ in trail_positions]
    ax2.set_xlim(0, beam.L)
    ax2.set_ylim(M_range[0] * 1.2, M_range[1] * 1.2)
    ax2.set_xlabel("Beam position $x$")
    ax2.set_ylabel("Moment $M(x,t)$")
    ax2.grid(True)
    ax2.legend()

    def init():
        line_disp.set_data([], [])
        force_line_disp.set_data([], [])
        for line in trail_lines_disp:
            line.set_data([], [])
        line_moment.set_data([], [])
        force_line_moment.set_data([], [])
        for line in trail_lines_moment:
            line.set_data([], [])
        return [line_disp, force_line_disp, *trail_lines_disp,
                line_moment, force_line_moment, *trail_lines_moment]

    def update(frame):
        t = beam.t_vals[frame]
        w = w_sol[frame]
        M = beam.calculate_moment(w)

        x_force = beam.v * t

        # Update main force lines
        if 0 <= x_force <= beam.L:
            force_line_disp.set_data([x_force, x_force], w_range)
            force_line_moment.set_data([x_force, x_force], M_range)
        else:
            force_line_disp.set_data([], [])
            force_line_moment.set_data([], [])

        # Update trailing lines
        for i, d in enumerate(trail_positions):
            x_trail = x_force - d
            if 0 <= x_trail <= beam.L:
                trail_lines_disp[i].set_data([x_trail, x_trail], w_range)
                trail_lines_moment[i].set_data([x_trail, x_trail], M_range)
            else:
                trail_lines_disp[i].set_data([], [])
                trail_lines_moment[i].set_data([], [])

        # Update plots
        line_disp.set_data(beam.x_vals, w)
        line_moment.set_data(beam.x_vals, M)

        ax1.set_title(f"Beam Deflection and Moment \nTime = {t:.2f} s, Force at x = {min(x_force, beam.L):.2f}" +
                    (" (active)" if x_force <= beam.L else " (exited)"))

        return [line_disp, force_line_disp, *trail_lines_disp,
                line_moment, force_line_moment, *trail_lines_moment]



    N_FRAMES = 150
    frame_indices = np.linspace(0, len(beam.t_vals) - 1, N_FRAMES, dtype=int)
    ani = FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=True, interval=1000 / 30)

    ani.save(f"results/{beam.name}.gif", writer="ffmpeg", fps=30)
    plt.close(fig)
    print(f"Animation saved as {beam.name}'.gif'")
