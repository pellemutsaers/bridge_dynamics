import numpy as np
import matplotlib.pyplot as plt


class BaseBeam:
    def __init__(self, v, L, C, EI, rho, t_end, N, num_t):
        """
        Initialize the base beam with standard properties.
        :param v: Velocity of the moving load
        :param L: Length of the beam
        :param C: Damping coefficient
        :param EI: Bending stiffness
        :param rho: Mass per unit length
        :param t_end: Simulation end time
        :param N: Number of modes
        :param num_t: Number of time steps
        """
        self.v = v
        self.L = L
        self.C = C
        self.EI = EI
        self.rho = rho
        self.t_end = t_end
        self.N = N
        self.num_t = num_t
        self.t_vals = np.linspace(0, t_end, num_t)

    def plot_displacement_and_moment(self, w_mid_vals, M_mid_vals):
        """Plot displacement and moment at the midpoint over time."""
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Displacement plot
        ax[0].plot(self.t_vals, w_mid_vals, label='Displacement at x=L/2')
        ax[0].set_ylabel('Deflection $w(L/2,t)$')
        ax[0].legend()
        ax[0].grid(True)

        # Moment plot
        ax[1].plot(self.t_vals, M_mid_vals, color='orange', label='Moment at x=L/2')
        ax[1].set_ylabel('Moment $M(L/2,t)$')
        ax[1].set_xlabel('Time [s]')
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig("w_and_M_at_midpoint.png", dpi=300)
        print("Midpoint graph saved as 'w_and_M_at_midpoint.png'")

class MCFMBeam(BaseBeam):
    def __init__(self, v, L, C, EI, rho, t_end, N, num_t, P0, num_x=10000):
        super().__init__(v, L, C, EI, rho, t_end, N, num_t)
        self.P0 = P0
        self.x_vals = np.linspace(0, L, num_x)

    def get_omega_n(self, n):
        """Natural frequency of mode n."""
        return (n * np.pi / self.L)**2 * np.sqrt(self.EI / self.rho)

    def get_xi_n(self, omega_n):
        """Damping ratio."""
        return self.C / (2 * self.rho * omega_n)

    def get_sum_element(self, t, n):
        """Calculate the summation element for mode n."""
        omega_n = self.get_omega_n(n)
        xi_n = self.get_xi_n(omega_n)
        beta_n = xi_n * omega_n  # beta_n is the damping coefficient
        Omega_n = n * self.v * np.pi / self.L

        # Denominator of the summation term
        denom = (omega_n**2 - Omega_n**2)**2 + (4 * beta_n**2 * Omega_n**2)

        # Components of the formula
        term1 = (omega_n**2 - Omega_n**2) * np.sin(Omega_n * t)
        term2 = -2 * beta_n * Omega_n * np.cos(Omega_n * t)
        term3 = np.exp(-beta_n * t) * (
            2 * beta_n * Omega_n * np.cos(omega_n * t) +
            (Omega_n / omega_n) * ((2 * beta_n**2 - Omega_n**2 - omega_n**2) * np.sin(omega_n * t))
        )

        # Full summation element
        return (term1 + term2 + term3) / denom

    def calculate_w(self, t):
        """Calculate displacement w(x, t)."""
        w_vals = np.zeros_like(self.x_vals)
        for n in range(1, self.N + 1):
            elem = self.get_sum_element(t, n)
            w_vals += elem * np.sin(n * np.pi * self.x_vals / self.L)
        return w_vals * 2 * self.P0 / (self.rho * self.L)

    def calculate_moment(self, w_vals):
        """Calculate bending moment M(x, t)."""
        return self.EI * np.gradient(np.gradient(w_vals, self.x_vals), self.x_vals)

    def calculate_midpoint_results(self):
        """Calculate displacement and moment at the midpoint over time."""
        x_mid = self.L / 2
        mid_idx = len(self.x_vals) // 2
        w_mid_vals = []
        M_mid_vals = []
        for t in self.t_vals:
            w_t = -self.calculate_w(t)
            M_t = self.calculate_moment(w_t)
            w_mid_vals.append(w_t[mid_idx])
            M_mid_vals.append(M_t[mid_idx])
        return np.array(w_mid_vals), np.array(M_mid_vals)


class MHFMBeam(BaseBeam):
    def __init__(self, v, L, C, EI, rho, t_end, N, num_t, P0, num_x=10000):
        super().__init__(v, L, C, EI, rho, t_end, N, num_t)
        self.P0 = P0
        self.x_vals = np.linspace(0, L, num_x)

    def P(self, tau):
        """Force function."""
        return self.P0

    def get_omega_n(self, n):
        """Natural frequency of mode n."""
        return (n * np.pi / self.L)**2 * np.sqrt(self.EI / self.rho)

    def get_xi_n(self, omega_n):
        """Damping ratio."""
        return self.C / (2 * self.rho * omega_n)

    def get_omega_np(self, omega_n, xi_n):
        """Damped natural frequency."""
        return omega_n * np.sqrt(1 - xi_n**2)

    def get_q_n(self, t, n, num_steps=1000):
        """Solve for modal amplitude q_n."""
        omega_n = self.get_omega_n(n)
        xi_n = self.get_xi_n(omega_n)
        omega_np = self.get_omega_np(omega_n, xi_n)

        tau_vals = np.linspace(0, t, num_steps)
        dtau = t / num_steps
        on_beam_mask = (self.v * tau_vals <= self.L)

        integrand_vals = (
            np.exp(-xi_n * omega_n * (t - tau_vals)) *
            np.sin(omega_np * (t - tau_vals)) *
            np.sin(n * np.pi * self.v * tau_vals / self.L) *
            self.P(tau_vals) * on_beam_mask
        )

        integral = np.trapz(integrand_vals, dx=dtau)
        return 2 / (self.rho * self.L * omega_np) * integral

    def calculate_w(self, t):
        """Calculate displacement w(x, t)."""
        w_vals = np.zeros_like(self.x_vals)
        for n in range(1, self.N + 1):
            q_n = self.get_q_n(t, n)
            w_vals += q_n * np.sin(n * np.pi * self.x_vals / self.L)
        return w_vals

    def calculate_moment(self, w_vals):
        """Calculate bending moment M(x, t)."""
        return self.EI * np.gradient(np.gradient(w_vals, self.x_vals), self.x_vals)

    def calculate_midpoint_results(self):
        """Calculate displacement and moment at the midpoint over time."""
        x_mid = self.L / 2
        mid_idx = mid_idx = len(self.x_vals) // 2
        w_mid_vals = []
        M_mid_vals = []
        for t in self.t_vals:
            w_t = -self.calculate_w(t)
            M_t = self.calculate_moment(w_t)
            w_mid_vals.append(w_t[mid_idx])
            M_mid_vals.append(M_t[mid_idx])
        return np.array(w_mid_vals), np.array(M_mid_vals)

    def calculate_deflection_and_moment(self):
        """Calculate deflection and moment over the entire beam and time."""
        w_vals = []
        M_vals = []
        for t in self.t_vals:
            w_t = -self.calculate_w(t)
            M_t = self.calculate_moment(w_t)
            w_vals.append(w_t)
            M_vals.append(M_t)
        return np.array(w_vals), np.array(M_vals)

class MMMBeam(BaseBeam):
    def __init__(self, v, L, C, EI, rho, t_end, N, num_t, M, beta, Gamma):
        super().__init__(v, L, C, EI, rho, t_end, N, num_t)
        self.M = M
        self.dt = t_end / num_t
        self.beta = beta
        self.Gamma = Gamma
        self.g = 9.81
        self.x_vals = np.linspace(0, L, 1000)

    def ODE1_matrix(self, N):
        Mmat = np.zeros((N, N))
        Cmat = np.zeros((N, N))
        Kmat = np.zeros((N, N))

        for i in range(N):
            # Mass matrix
            Mmat[i, i] = 1.0

            # Damping matrix
            Cmat[i, i] = 0.0

            # Stiffness matrix
            Kmat[i, i] = ((i + 1)**4 * np.pi**4 * self.EI) / (self.L**4 * self.rho)

        return Mmat, Cmat, Kmat

    def newmark_solution(self, x_location):
        Mmat, Cmat, Kmat = self.ODE1_matrix(self.N)

        u_cur = np.zeros((self.N, 1))
        v_cur = np.zeros((self.N, 1))
        a_cur = np.zeros((self.N, 1))
        y_array = []

        c1 = 1 / (self.beta * self.dt**2)
        c2 = self.Gamma / (self.beta * self.dt)
        c3 = 1 / (self.beta * self.dt)
        c4 = 1 / (2 * self.beta) - 1
        c5 = 1 - (self.Gamma / self.beta)
        c6 = self.dt * (1 - (self.Gamma / (2 * self.beta)))

        for t_idx, t in enumerate(self.t_vals):
            # Calculate the acceleration of the beam (a_beam) based on current accelerations
            a_beam_t = np.sum(a_cur[k, 0] * np.sin(np.pi * (k + 1) * self.v * (t + self.dt) / self.L) for k in range(self.N))

            # Calculate the full force vector at the current time step
            Fmat = np.array([[(2 * self.M * (self.g - a_beam_t) * np.sin(np.pi * (i + 1) * self.v * t / self.L) / (self.L * self.rho))] for i in range(self.N)])

            lhs = Mmat * c1 + Cmat * c2 + Kmat
            rhs = Fmat + Mmat @ (u_cur * c1 + v_cur * c3 + a_cur * c4) + Cmat @ (u_cur * c2 + v_cur * c5 + a_cur * c6)

            u_next = np.linalg.solve(lhs, rhs)
            a_next = c1 * (u_next - u_cur) - v_cur * c3  - a_cur * c4
            v_next = c2 * (u_next - u_cur) + v_cur * c5 + a_cur * c6

            y_sol = 0
            x_val = x_location
            for k in range(self.N):
                y_sol += u_cur[k, 0] * np.sin(np.pi * x_val * (k + 1) / self.L)
            y_array.append(y_sol)

            u_cur = u_next
            v_cur = v_next
            a_cur = a_next

        return -np.array(y_array)

    def calculate_moment(self, w_vals):
        """Calculate bending moment M(x, t)."""
        return self.EI * np.gradient(np.gradient(w_vals, self.x_vals), self.x_vals)

    def calculate_midpoint_results(self, do_moment=False):
            """Calculate displacement and moment at the midpoint over time."""
            x_mid = self.L / 2
            w_mid_vals = self.newmark_solution(x_mid)  # Calculate midpoint displacement directly

            if do_moment:
                M_vals = []
                w_vals_full = self.newmark_solution(self.x_vals) # Calculate displacement over the entire beam
                x_mid_idx = len(self.x_vals) // 2
                for i in range(len(self.t_vals)):
                    M_vals.append(self.calculate_moment(w_vals_full[i, :])) # Assuming w_vals_full is time x space
                M_mid_vals = np.array(M_vals)[:, x_mid_idx] # Extract midpoint moment over time
                return w_mid_vals, M_mid_vals
            else:
                return w_mid_vals

    def calculate_deflection_and_moment(self):
        """Calculate deflection and moment over the entire beam and time."""
        w_vals = []
        M_vals = []
        for t in self.t_vals:
            w_t = self.calculate_w(t)
            M_t = self.calculate_moment(w_t)
            w_vals.append(w_t)
            M_vals.append(M_t)
        return np.array(w_vals), np.array(M_vals)



