import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation

def beam_element_stiffness_matrix(EI, h):
    factor = EI / h**3
    Ke = factor * np.array([
        [12,        6*h,       -12,       6*h],
        [6*h,    4*h**2,    -6*h,    2*h**2],
        [-12,     -6*h,       12,       -6*h],
        [6*h,    2*h**2,    -6*h,    4*h**2]
    ])
    return Ke


def beam_element_mass_matrix(rho, h):
    factor = (rho * h) / 420.0
    Me = factor * np.array([
        [156, 22 * h, 54, -13 * h],
        [22 * h, 4 * h**2, 13 * h, -3 * h**2],
        [54, 13 * h, 156, -22 * h],
        [-13 * h, -3 * h**2, -22 * h, 4 * h**2]
    ])
    return Me


def shape_function_interpolate(dofs, x_left, x_right, x_local_array):
    a = x_left
    b = x_right
    h_element = b - a
    dofs_reshaped = np.array(dofs).reshape(-1, 1)
    s = x_local_array / h_element
    s2 = s * s
    s3 = s * s2
    N1 = 1 - 3*s2 + 2*s3
    N2 = h_element * (s - 2*s2 + s3)
    N3 = 3*s2 - 2*s3
    N4 = h_element * (-s2 + s3)
    interpolated_values = (
        dofs_reshaped[0] * N1 +
        dofs_reshaped[1] * N2 +
        dofs_reshaped[2] * N3 +
        dofs_reshaped[3] * N4
    )
    return interpolated_values


def shape_function_values(xF, h):
    s = xF / h
    s2 = s * s
    s3 = s * s2
    N1 = 1 - 3*s2 + 2*s3
    N2 = h * (s - 2*s2 + s3)
    N3 = 3*s2 - 2*s3
    N4 = h * (-s2 + s3)
    return (N1, N2, N3, N4)


def newmark_beta_timestep(
    u_cur, v_cur, a_cur,
    M_red, C_red, K_red, f_next_red, dt,
    beta=0.25, gamma=0.5
):
    c1 = 1 / (beta * dt**2)
    c2 = gamma / (beta * dt)
    c3 = 1 / (beta * dt)
    c4 = 1 / (2 * beta) - 1
    c5 = 1 - (gamma / beta)
    c6 = dt * (1 - (gamma / (2 * beta)))
    lhs_red = M_red * c1 + C_red * c2 + K_red
    rhs_red = f_next_red + M_red @ (u_cur * c1 + v_cur * c3 + a_cur * c4) + C_red @ (u_cur * c2 + v_cur * c5 + a_cur * c6)
    u_next_red = np.linalg.solve(lhs_red, rhs_red)
    a_next_red = c1 * (u_next_red - u_cur) - v_cur * c3  - a_cur * c4
    v_next_red = c2 * (u_next_red - u_cur) + v_cur * c5 + a_cur * c6
    return u_next_red, v_next_red, a_next_red


# --- Models--

def simulate_force_model(n, dt, animate=False, cond="const"):
    # Parameters
    L = 36
    EI = 2.1e11 * 0.41
    v = 27.78
    h = L / n
    rho = 17000
    Mc = 42.11e3
    Mb = 68.20e2
    Ma = 18.13e2
    Mt = Mc + 2 * Mb + 4 * Ma
    Ndof = 2 * (n + 1)

    # Initializing matrices
    K = np.zeros((Ndof, Ndof))
    C = np.zeros((Ndof, Ndof))
    M = np.zeros((Ndof, Ndof))
    fixed_dofs = [0, 2 * n]
    for el in range(n):
        Ke = beam_element_stiffness_matrix(EI, h)
        Me = beam_element_mass_matrix(rho, h)
        dof_map = [2*el, 2*el+1, 2*el+2, 2*el+3]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += Ke[i, j]
                M[dof_map[i], dof_map[j]] += Me[i, j]


    # Removing fixed DOFs for easier time integration
    all_dofs = np.arange(Ndof)
    active_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    K_red = K[np.ix_(active_dofs, active_dofs)]
    M_red = M[np.ix_(active_dofs, active_dofs)]
    C_red = C[np.ix_(active_dofs, active_dofs)]

    # Initial conditions: (w0, phi0, w1, phi1, ... w_n+1, phi_n+1) and derivatives
    u_cur_red = np.zeros(len(active_dofs))
    v_cur_red = np.zeros(len(active_dofs))
    a_cur_red = np.zeros(len(active_dofs))

    displacement_history = []
    t_plot_vals = []
    t_vals = np.arange(0, L/v, dt)
    count = 0

    for t in tqdm(t_vals, desc="Simulating Beam Deflection"):
        # Calculate the force position
        Fx = v * t
        if cond == "const":
            F_val = Mt * 9.81
        elif cond == "harm":
            F_val = Mt * 9.81 * (1 + 0.2 * np.sin(17.13846251305491*9 * t))
        f_global = np.zeros(Ndof)
        element_idx_force = int(Fx // h)
        if element_idx_force == n:
            element_idx_force = n - 1
        x_local_force = Fx - element_idx_force * h

        # Creating the force vector
        N1f, N2f, N3f, N4f = shape_function_values(x_local_force, h)
        f_el = np.array([-F_val * N1f, -F_val * N2f, -F_val * N3f, -F_val * N4f])
        dof_map_force = [
            2 * element_idx_force,
            2 * element_idx_force + 1,
            2 * element_idx_force + 2,
            2 * element_idx_force + 3
        ]
        for i in range(4):
            f_global[dof_map_force[i]] += f_el[i]
        f_next_red = f_global[active_dofs]

        # Time integration step
        u_next_red, v_next_red, a_next_red = newmark_beta_timestep(
            u_cur_red, v_cur_red, a_cur_red,
            M_red, C_red, K_red, f_next_red, dt
        )
        u_cur_red = u_next_red
        v_cur_red = v_next_red
        a_cur_red = a_next_red

        # Storing results including fixed dofs
        u_full = np.zeros(Ndof)
        u_full[active_dofs] = u_next_red
        u_full[fixed_dofs] = 0.0
        if count % 10 == 0:
            displacement_history.append(u_full)
            t_plot_vals.append(t)
        count += 1

    if animate:
        animate_beam(displacement_history, t_plot_vals, n, h, L, v, shape_function_interpolate)
    return extract_midpoint_deflection(displacement_history, t_plot_vals, n, h, shape_function_interpolate)


def simulate_mass_model(n, dt, animate=False):
    # Parameters
    L = 36
    EI = 2.1e11 * 0.41
    v = 27.78
    h = L / n
    rho = 17000
    g = 9.81
    Mc = 42.11e3
    Mb = 68.20e2
    Ma = 18.13e2
    Mt = Mc + 2 * Mb + 4 * Ma
    Ndof = 2 * (n + 1)

    # Initializing matrices
    K = np.zeros((Ndof, Ndof))
    C = np.zeros((Ndof, Ndof))
    M = np.zeros((Ndof, Ndof))
    fixed_dofs = [0, 2 * n]
    for el in range(n):
        Ke = beam_element_stiffness_matrix(EI, h)
        Me = beam_element_mass_matrix(rho, h)
        dof_map = [2*el, 2*el+1, 2*el+2, 2*el+3]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += Ke[i, j]
                M[dof_map[i], dof_map[j]] += Me[i, j]

    all_dofs = np.arange(Ndof)
    active_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    K_red = K[np.ix_(active_dofs, active_dofs)]
    M_red_base = M[np.ix_(active_dofs, active_dofs)]
    C_red = C[np.ix_(active_dofs, active_dofs)]

    u_cur_red = np.zeros(len(active_dofs))
    v_cur_red = np.zeros(len(active_dofs))
    a_cur_red = np.zeros(len(active_dofs))

    displacement_history = []
    t_plot_vals = []
    t_vals = np.arange(0, L/v, dt)
    count = 0

    for t in tqdm(t_vals, desc="Simulating Beam Deflection"):
        Fx = v * t
        f_global_grav_term = np.zeros(Ndof)
        M_added_global = np.zeros((Ndof, Ndof))
        element_idx_force = int(Fx // h)
        if element_idx_force == n:
            element_idx_force = n - 1
        x_local_force = Fx - element_idx_force * h

        # Adding Mt * g to the force vector
        N1f, N2f, N3f, N4f = shape_function_values(x_local_force, h)
        N_vector = np.array([N1f, N2f, N3f, N4f])
        f_el_grav = -Mt * g * N_vector
        dof_map_force = [
            2 * element_idx_force,
            2 * element_idx_force + 1,
            2 * element_idx_force + 2,
            2 * element_idx_force + 3
        ]
        for i in range(4):
            f_global_grav_term[dof_map_force[i]] += f_el_grav[i]

        # Adding Mt * ü to the mass matrix
        M_load_local = Mt * (N_vector.reshape(-1, 1) @ N_vector.reshape(1, -1))
        for i in range(4):
            for j in range(4):
                M_added_global[dof_map_force[i], dof_map_force[j]] += M_load_local[i, j]

        f_next_red = f_global_grav_term[active_dofs]
        M_eff_red = M_red_base + M_added_global[np.ix_(active_dofs, active_dofs)]

        u_next_red, v_next_red, a_next_red = newmark_beta_timestep(
            u_cur_red, v_cur_red, a_cur_red,
            M_eff_red, C_red, K_red, f_next_red, dt
        )

        u_cur_red = u_next_red
        v_cur_red = v_next_red
        a_cur_red = a_next_red

        u_full = np.zeros(Ndof)
        u_full[active_dofs] = u_next_red
        u_full[fixed_dofs] = 0.0

        if count % 10 == 0:
            displacement_history.append(u_full)
            t_plot_vals.append(t)
        count += 1

    if animate:
        animate_beam(displacement_history, t_plot_vals, n, h, L, v, shape_function_interpolate)
    return extract_midpoint_deflection(displacement_history, t_plot_vals, n, h, shape_function_interpolate)

def simulate_two_mass_model(n, dt, k_suspension, c_suspension, animate=False):
    # Parameters
    L = 36
    EI = 2.1e11 * 0.41
    v = 27.78
    h = L / n
    rho = 17000
    g = 9.81
    
    # Vehicle masses
    Mc = 42.11e3
    Mb = 68.20e2
    Ma = 18.13e2
    
    M_bottom = 4 * Ma + 2 * Mb 

    Ndof_beam = 2 * (n + 1) # Number of DOFs for the beam
    Ndof_total_system = Ndof_beam + 1 # Total DOFs including 'z'

    # Initializing global matrices for the beam only
    K_beam = np.zeros((Ndof_beam, Ndof_beam))
    C_beam = np.zeros((Ndof_beam, Ndof_beam))
    M_beam = np.zeros((Ndof_beam, Ndof_beam))

    fixed_dofs_beam = [0, 2 * n]
    for el in range(n):
        Ke = beam_element_stiffness_matrix(EI, h)
        Me = beam_element_mass_matrix(rho, h)
        dof_map = [2*el, 2*el+1, 2*el+2, 2*el+3]
        for i in range(4):
            for j in range(4):
                K_beam[dof_map[i], dof_map[j]] += Ke[i, j]
                M_beam[dof_map[i], dof_map[j]] += Me[i, j]

    all_dofs_beam = np.arange(Ndof_beam)
    active_dofs_beam = np.setdiff1d(all_dofs_beam, fixed_dofs_beam)
    
    idx_beam_active = np.arange(len(active_dofs_beam))
    idx_z = len(active_dofs_beam) 

    K_beam_red_base = K_beam[np.ix_(active_dofs_beam, active_dofs_beam)]
    M_beam_red_base = M_beam[np.ix_(active_dofs_beam, active_dofs_beam)]
    C_beam_red_base = C_beam[np.ix_(active_dofs_beam, active_dofs_beam)]

    u_cur_aug = np.zeros(len(active_dofs_beam) + 1)
    v_cur_aug = np.zeros(len(active_dofs_beam) + 1)
    a_cur_aug = np.zeros(len(active_dofs_beam) + 1)

    displacement_history = [] # Stores only beam deflections for plotting
    t_plot_vals = []
    t_vals = np.arange(0, L/v, dt)
    count = 0

    for t in tqdm(t_vals, desc="Simulating Beam Deflection (Two-Mass Model)"):
        Fx = v * t
        
        # Determine current element under load and local position
        element_idx_force = int(Fx // h)
        if element_idx_force == n:
            element_idx_force = n - 1
        x_local_force = Fx - element_idx_force * h

        N1f, N2f, N3f, N4f = shape_function_values(x_local_force, h)
        N_vector_element = np.array([N1f, N2f, N3f, N4f])

        N_red = np.zeros(len(active_dofs_beam))
        dof_map_force = [
            2 * element_idx_force,
            2 * element_idx_force + 1,
            2 * element_idx_force + 2,
            2 * element_idx_force + 3
        ]
        for i, dof_idx in enumerate(dof_map_force):
            if dof_idx in active_dofs_beam:
                local_active_idx = np.where(active_dofs_beam == dof_idx)[0][0]
                N_red[local_active_idx] = N_vector_element[i]

        M_eff_aug = np.zeros((len(active_dofs_beam) + 1, len(active_dofs_beam) + 1))
        M_eff_aug[np.ix_(idx_beam_active, idx_beam_active)] = M_beam_red_base + M_bottom * (N_red.reshape(-1, 1) @ N_red.reshape(1, -1))
        M_eff_aug[np.ix_(idx_beam_active, [idx_z])] = Mc * N_red.reshape(-1, 1)
        M_eff_aug[idx_z, idx_z] = Mc

        C_eff_aug = np.zeros((len(active_dofs_beam) + 1, len(active_dofs_beam) + 1))
        C_eff_aug[np.ix_(idx_beam_active, idx_beam_active)] = C_beam_red_base
        C_eff_aug[np.ix_([idx_z], idx_beam_active)] = -c_suspension * N_red
        C_eff_aug[idx_z, idx_z] = c_suspension

        K_eff_aug = np.zeros((len(active_dofs_beam) + 1, len(active_dofs_beam) + 1))
        K_eff_aug[np.ix_(idx_beam_active, idx_beam_active)] = K_beam_red_base
        K_eff_aug[np.ix_([idx_z], idx_beam_active)] = -k_suspension * N_red
        K_eff_aug[idx_z, idx_z] = k_suspension
        
        f_next_aug = np.zeros(len(active_dofs_beam) + 1)
        f_next_aug[idx_beam_active] = -(M_bottom + Mc) * g * N_red.reshape(-1, 1).flatten()

        u_next_aug, v_next_aug, a_next_aug = newmark_beta_timestep(
            u_cur_aug, v_cur_aug, a_cur_aug,
            M_eff_aug, C_eff_aug, K_eff_aug, f_next_aug, dt
        )
        
        u_cur_aug = u_next_aug
        v_cur_aug = v_next_aug
        a_cur_aug = a_next_aug

        # Store only beam DOFs for plotting and animation
        u_full_beam_part = np.zeros(Ndof_beam)
        u_full_beam_part[active_dofs_beam] = u_next_aug[idx_beam_active]
        u_full_beam_part[fixed_dofs_beam] = 0.0

        if count % 10 == 0:
            displacement_history.append(u_full_beam_part)
            t_plot_vals.append(t)
        count += 1
    
    if animate:
        animate_beam(displacement_history, t_plot_vals, n, h, L, v, shape_function_interpolate)
    return extract_midpoint_deflection(displacement_history, t_plot_vals, n, h, shape_function_interpolate)


# --- Animation and Plotting ---

def animate_beam(displacement_history, t_plot_vals, n, h, L, v, shape_function_interpolate):
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'b-', label="Deflected Beam")
    original_beam, = ax.plot([0, L], [0, 0], 'k--', alpha=0.7, label="Original Beam Position")
    point_load_marker, = ax.plot([], [], 'ro', markersize=8, label="Point Load")

    max_deflection_global = 0.0
    for u_global_at_time_step in displacement_history:
        max_deflection_global = max(max_deflection_global, np.max(np.abs(u_global_at_time_step[::2])))
    y_margin = max(0.005, 1.5 * max_deflection_global)

    ax.set_ylim(-y_margin, y_margin)
    ax.set_xlim(0, L)
    ax.set_xlabel("Beam Length (m)")
    ax.set_ylabel("Deflection (m)")
    ax.set_title("Beam Deflection Under Moving Load")
    ax.grid(True)
    ax.legend()

    num_points_per_element = 20
    x_beam_plot_fine = np.linspace(0, L, n * num_points_per_element + 1)

    def animate(i):
        u_global_at_time_step = displacement_history[i]
        t_current = t_plot_vals[i]
        y_deflected = np.zeros_like(x_beam_plot_fine, dtype=float)
        for el_idx in range(n):
            element_dofs = u_global_at_time_step[2 * el_idx : 2 * el_idx + 4]
            x_left_el = el_idx * h
            x_right_el = (el_idx + 1) * h
            x_pts_in_element_mask = (x_beam_plot_fine >= x_left_el - 1e-9) & (x_beam_plot_fine <= x_right_el + 1e-9)
            x_pts_in_element = x_beam_plot_fine[x_pts_in_element_mask]
            if len(x_pts_in_element) > 0:
                x_local_pts = x_pts_in_element - x_left_el
                interpolated_deflections_for_element = np.array([
                    shape_function_interpolate(element_dofs, x_left_el, x_right_el, np.array([xi_val]))
                    for xi_val in x_local_pts
                ]).flatten()
                y_deflected[x_pts_in_element_mask] = interpolated_deflections_for_element
        line.set_data(x_beam_plot_fine, y_deflected)

        load_x_pos = v * t_current
        if 0 <= load_x_pos <= L:
            load_element_idx = int(load_x_pos // h)
            load_element_idx = np.clip(load_element_idx, 0, n - 1)
            load_x_local = load_x_pos - load_element_idx * h
            load_x_local = np.clip(load_x_local, 0, h)
            load_element_dofs = u_global_at_time_step[2 * load_element_idx : 2 * load_element_idx + 4]
            load_y_pos = shape_function_interpolate(load_element_dofs, load_element_idx * h, (load_element_idx + 1) * h, np.array([load_x_local]))
            point_load_marker.set_data([load_x_pos], [load_y_pos])
        else:
            point_load_marker.set_data([], [])

        ax.set_title(f"Beam Deflection Under Moving Load (Time: {t_current:.3f} s)")
        return line, original_beam, point_load_marker

    print("Creating animation...")
    ani = animation.FuncAnimation(fig, animate, frames=len(displacement_history), interval=50, blit=True)
    gif_filename = "beam_deflection.gif"
    ani.save(gif_filename, writer='pillow', fps=20)
    print(f"Animation saved to {gif_filename}")
    plt.show()


def extract_midpoint_deflection(displacement_history, t_plot_vals, n, h, shape_function_interpolate):
    midpoint_idx = n // 2
    midpoint_x_left = midpoint_idx * h
    midpoint_x_right = (midpoint_idx + 1) * h
    displacement_over_time = []
    for u_global_at_time_step in displacement_history:
        midpoint_dofs = u_global_at_time_step[2 * midpoint_idx : 2 * midpoint_idx + 4]
        midpoint_x_local = (midpoint_x_left + midpoint_x_right) / 2 - midpoint_x_left
        midpoint_deflection = shape_function_interpolate(midpoint_dofs, midpoint_x_left, midpoint_x_right, np.array([midpoint_x_local]))
        displacement_over_time.append(midpoint_deflection[0])

    # plt.figure(figsize=(10, 5))
    # plt.plot(t_plot_vals, displacement_over_time, label='Midpoint Deflection', color='orange')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Deflection (m)")
    # plt.title("Midpoint Deflection Over Time")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    return displacement_over_time, t_plot_vals


if __name__ == "__main__":
    n = 20
    dt = 0.001
    print("Running force model simulation...")
    simulate_force_model(n, dt)
    print("Running mass model simulation...")
    simulate_mass_model(n, dt)
