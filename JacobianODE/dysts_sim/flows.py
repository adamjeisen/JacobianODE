"""
Various low-dimensional dynamical systems in Python.
For flows that occur on unbounded intervals (eg non-autonomous systems),
coordinates are transformed to a basis where the domain remains bounded

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

Adapted from https://github.com/williamgilpin/dysts
Huge shoutout to William Gilpin for a fantastic repo, check out his work!
"""
from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import ListConfig
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .base import data_default, DynSys, DynSysDelay, staticjit
from .utils import find_significant_frequencies

class Arneodo(DynSys):
    @staticjit
    def _rhs(x, y, z, t, a, b, c, d):
        xdot = y
        ydot = z
        zdot = -a * x - b * y - c * z + d * x ** 3
        return xdot, ydot, zdot

    # @staticjit
    # def _jac(x, y, z, t, a, b, c, d):
    #     row1 = [0, 1, 0]
    #     row2 = [0, 0, 1]
    #     row3 = [-a + 3 * d * x ** 2, -b, -c]
    #     return [row1, row2, row3]
    @staticjit
    def _jac(x, y, z, t, a, b, c, d):
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-a + 3 * d * x**2, -b, -c]
        ], dtype=np.float64)

class Lorenz(DynSys):
    @staticjit
    def _rhs(x, y, z, t, beta, rho, sigma):
        xdot = sigma * (y - x)
        ydot = rho * x - x * z - y 
        zdot = x * y - beta * z
        return np.array([xdot, ydot, zdot])

    @staticjit
    def _jac(x, y, z, t, beta, rho, sigma):
        # Return a numpy array with float64 dtype to ensure consistent types
        return np.array([
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta]
        ], dtype=np.float64)
    # @staticjit
    # def _jac(x, y, z, t, beta, rho, sigma):
    #     return np.array([
    #         [-sigma, sigma, 0],
    #         [rho - z, -1, -x],
    #         [y, x, -beta]
    #     ])

class Lorenz96(DynSys):
    @staticjit
    def _rhs(x, t, F):
        N = len(x)
        xdot = np.zeros(N)
        for i in range(N):
            xdot[i] = (x[(i-1)%N] * (x[(i+1)%N] - x[i-2])) - x[i] + F
        return xdot

    @staticjit
    def _jac(x, t, F):
        N = len(x)
        jac = [[0 for j in range(N)] for i in range(N)]
        for i in range(N):
            jac[i][i] = -1
            jac[i][(i+1)%N] = x[(i-1)%N]
            jac[i][(i-2)%N] = -x[(i+1)%N]
            jac[i][(i-1)%N] = x[(i+1)%N] - x[i-2]
        return jac

class LotkaVolterra(DynSys):
    @staticjit
    def _rhs(x1, x2, t, alpha, beta, gamma, delta):
        x1dot = x1 * (alpha - beta * x2)
        x2dot = -x2 * (gamma - delta * x1)
        return x1dot, x2dot
    
    @staticjit
    def _jac(x1, x2, t, alpha, beta, gamma, delta):
        row1 = [alpha - beta * x2, -beta * x1]
        row2 = [delta * x2, -gamma + delta * x1]
        return [row1, row2]

class RNNOscillator(DynSys):
    @staticjit
    def _rhs(x, t, W, tau):
        return (1/tau)*(-x + W @ np.tanh(x))
    
    @staticjit
    def _jac(x, t, W, tau):
        return (1/tau)*(-np.eye(W.shape[-1]).astype(W.dtype) + W @ np.diag(1 - np.tanh(x)**2).astype(W.dtype))

class RNNChaotic(DynSys):
    @staticjit
    def _rhs(x, t, W, tau):
        x = x.astype(W.dtype)
        return (1/tau)*(-x + W @ np.tanh(x))
    
    @staticjit
    def _jac(x, t, W, tau):
        return (1/tau)*(-np.eye(W.shape[-1]).astype(W.dtype) + W @ np.diag(1 - np.tanh(x)**2).astype(W.dtype))
    
class RNNStableSmall(DynSys):
    @staticjit
    def _rhs(x, t, W, tau):
        x = x.astype(W.dtype)
        return (1/tau)*(-x + W @ np.tanh(x))
    
    @staticjit
    def _jac(x, t, W, tau):
        return (1/tau)*(-np.eye(W.shape[-1]).astype(W.dtype) + W @ np.diag(1 - np.tanh(x)**2).astype(W.dtype))

class Lorenz96(DynSys):
    @staticjit
    def _rhs(X, t, F):
        N = X.shape[0]
        Xdot = np.zeros_like(X)
        for i in range(N):
            Xdot[i] = (X[(i + 1) % N] - X[i - 2]) * X[i - 1] - X[i] + F
        return Xdot

    @staticjit
    def _jac(X, t, F):
        N = X.shape[0]
        J = np.zeros((N, N))
        for i in range(N):
            J[i, i] = -1
            J[i, (i + 1) % N] = X[i - 1]
            J[i, i - 1] = X[(i + 1) % N] - X[i - 2]
            J[i, i - 2] = -X[i - 1]
        return J

class Lorenz96Small(DynSys):
    @staticjit
    def _rhs(X, t, F):
        N = X.shape[0]
        Xdot = np.zeros_like(X)
        for i in range(N):
            Xdot[i] = (X[(i + 1) % N] - X[i - 2]) * X[i - 1] - X[i] + F
        return Xdot

    @staticjit
    def _jac(X, t, F):
        N = X.shape[0]
        J = np.zeros((N, N))
        for i in range(N):
            J[i, i] = -1
            J[i, (i + 1) % N] = X[i - 1]
            J[i, i - 1] = X[(i + 1) % N] - X[i - 2]
            J[i, i - 2] = -X[i - 1]
        return J

class Lorenz96n32(Lorenz96):
    pass

class Lorenz96n64(Lorenz96):
    pass

# class VanDerPol(DynSys):
#     @staticjit
#     def _rhs(x, y, t, mu):
#         xdot = y
#         ydot = mu * (1 - x ** 2) * y - x
#         return xdot, ydot

#     @staticjit
#     def _jac(x, y, t, mu):
#         row1 = [0, 1]
#         row2 = [-2 * mu * x * y - 1, mu * (1 - x ** 2)]
#         return [row1, row2]

class VanDerPol(DynSys):
    @staticjit
    def _rhs(x, y, t, mu):
        xdot = y
        ydot = mu * (1 - x ** 2) * y - x
        return xdot, ydot

    @staticjit
    def _jac(x, y, t, mu):
        # Return a 2x2 numpy array instead of nested lists
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * x * y - 1.0, mu * (1.0 - x ** 2)]
        ])

class KuramotoSivashinsky:
    @staticjit
    def _rhs(Y, t, mu, L, lambda_val):
        # Number of spatial points
        N = Y.shape[-1]
        # Spatial grid size
        delta_x = L / N
        # Initialize the derivative array
        Ydot = np.zeros_like(Y)

        # Compute second-order and fourth-order finite differences
        for i in range(N):
            Y_dot_1 = (Y[(i + 1) % N] - Y[(i - 1) % N]) / (2 * delta_x)
            Y_dot_2 = (Y[(i + 2) % N] - 2 * Y[i] + Y[(i - 2) % N]) / (delta_x**2)
            Y_dot_4 = (Y[(i + 2) % N] - 4 * Y[(i + 1) % N] + 6 * Y[i] - 4 * Y[(i - 1) % N] + Y[(i - 2) % N]) / (delta_x**4)
            
            # Compute RHS using the KS equation
            Ydot[i] = - Y[i] * Y_dot_1 - (1 + mu*np.cos(2 * np.pi * i * delta_x / lambda_val)) * Y_dot_2 - Y_dot_4
        
        return Ydot
    # def _rhs(Y, t, mu, L, lambda_val):
    #     # Number of spatial points
    #     N = Y.shape[0]
    #     # Spatial grid size
    #     delta_x = L / N
        
    #     # Wavenumber array for the Fourier modes
    #     k = np.fft.fftfreq(N, d=delta_x) * 2 * np.pi
        
    #     # Compute derivatives in Fourier space
    #     Y_hat = np.fft.fft(Y)  # Fourier transform of Y
        
    #     # First, second, and fourth derivatives in Fourier space
    #     Y_dot_1_hat = 1j * k * Y_hat
    #     Y_dot_2_hat = -(k ** 2) * Y_hat
    #     Y_dot_4_hat = (k ** 4) * Y_hat

    #     # Transform back to physical space
    #     Y_dot_1 = np.fft.ifft(Y_dot_1_hat).real
    #     Y_dot_2 = np.fft.ifft(Y_dot_2_hat).real
    #     Y_dot_4 = np.fft.ifft(Y_dot_4_hat).real

    #     # Compute RHS of KS equation
    #     Ydot = -Y * Y_dot_1 - (1 + mu * np.cos(2 * np.pi * np.arange(N) * delta_x / lambda_val)) * Y_dot_2 - Y_dot_4

    #     return Ydot
    # def _rhs(Y, t, mu, L, lambda_val):
    #     # Number of spatial points
    #     N = Y.shape[0]
    #     # Spatial grid size
    #     delta_x = L / N
        
    #     # Wavenumber array for the Fourier modes
    #     k = np.fft.fftfreq(N, d=delta_x) * 2 * np.pi
    #     k_squared = k**2
    #     k_fourth = k**4
        
    #     # Compute derivatives in Fourier space
    #     Y_hat = np.fft.fft(Y)  # Fourier transform of Y
        
    #     # Second and fourth derivatives in Fourier space
    #     Y_dot_2_hat = -k_squared * Y_hat
    #     Y_dot_4_hat = k_fourth * Y_hat

    #     # Transform back to physical space
    #     Y_dot_2 = np.fft.ifft(Y_dot_2_hat).real
    #     Y_dot_4 = np.fft.ifft(Y_dot_4_hat).real

    #     # Compute the nonlinear term
    #     nonlinear_term = -Y * np.fft.ifft(1j * k * Y_hat).real

    #     # Compute RHS of KS equation
    #     Ydot = nonlinear_term - (1 + mu * np.cos(2 * np.pi * np.arange(N) * delta_x / lambda_val)) * Y_dot_2 - Y_dot_4

    #     return Ydot

    @staticjit
    def _jac(Y, t, mu, L, lambda_val):
        # Number of spatial points
        N = Y.shape[-1]
        # Spatial grid size
        delta_x = L / N
        # Initialize Jacobian matrix
        J = np.zeros((N, N))

        # Compute the Jacobian matrix
        for i in range(N):
            # Diagonal terms
            J[i, i] = -1 * (Y[(i + 1) % N] - Y[(i - 1) % N]) / (2 * delta_x)
            J[i, i] -= (1 + mu*np.cos(2 * np.pi * i * delta_x / lambda_val)) * (-2 / delta_x**2)
            J[i, i] -= (6 / delta_x**4)

            # Off-diagonal terms
            J[i, (i + 1) % N] = -Y[i] / (2 * delta_x) + (4 / delta_x**4)
            J[i, (i - 1) % N] = Y[i] / (2 * delta_x) + (4 / delta_x**4)
            J[i, (i + 2) % N] = -(1 + mu*np.cos(2 * np.pi * i * delta_x / lambda_val)) / delta_x**2 - 1 / delta_x**4
            J[i, (i - 2) % N] = -(1 + mu*np.cos(2 * np.pi * i * delta_x / lambda_val)) / delta_x**2 - 1 / delta_x**4

        return J

class LorotkaVoltarneodo(DynSys):
    def __init__(self, comm_strengths, **kwargs):
        super().__init__(**kwargs)
        self.names = ['Arneodo', 'Lorenz', 'LotkaVolterra']
        self.eqs = {}
        for name in self.names:
            self.eqs[name] = eval(name)()

        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if isinstance(comm_strengths, list) or isinstance(comm_strengths, ListConfig):
            comm_strengths = np.array(comm_strengths)

        self.comm_strengths = comm_strengths
        self.comm_weights = {}
        for i, receiver in enumerate(self.names):
            for j, sender in enumerate(self.names):
                if comm_strengths[i, j] != 0:
                    if i == j:
                        self.comm_weights[(receiver, sender)] = comm_strengths[i, j]
                    else:
                        self.comm_weights[(receiver, sender)] = np.random.randn(self.eqs[receiver].embedding_dimension, self.eqs[sender].embedding_dimension)
                        self.comm_weights[(receiver, sender)] /= np.linalg.norm(self.comm_weights[(receiver, sender)], ord=2)
                        self.comm_weights[(receiver, sender)] *= comm_strengths[i, j]
    def _rhs(self, x_a, y_a, z_a, x_l, y_l, z_l, x1, x2, t, a, b, c, d, beta, rho, sigma, alpha_lv, beta_lv, gamma_lv, delta_lv):
        update_params = dict(
            Arneodo=dict(
                    X=np.array([x_a, y_a, z_a]),
                    t=t,
                    params=dict(a=a, b=b, c=c, d=d),
                ),
        
            Lorenz=dict(
                X=np.array([x_l, y_l, z_l]),
                t=t,
                params=dict(beta=beta, rho=rho, sigma=sigma)
            ),
        
            LotkaVolterra=dict(
                X=np.array([x1, x2]),
                t=t,
                params=dict(alpha=alpha_lv, beta=beta_lv, gamma=gamma_lv, delta=delta_lv)
            )
        )
        
        weight_updates = {}
        for name in self.names:
            weight_updates[name] = np.array(self.eqs[name]._rhs(*update_params[name]['X'], update_params[name]['t'], **update_params[name]['params']))
        
        for (receiver, sender), weight in self.comm_weights.items():
            if receiver != sender:
                weight_updates[receiver] += weight @ update_params[sender]['X']
            else:
                weight_updates[receiver] *= weight
        
        return [*weight_updates['Arneodo'], *weight_updates['Lorenz'], *weight_updates['LotkaVolterra']]

    # def _jac(self, x_a, y_a, z_a, x_l, y_l, z_l, x1, x2, t, a, b, c, d, beta, rho, sigma, alpha_lv, beta_lv, gamma_lv, delta_lv):
    #     update_params = dict(
    #         Arneodo=dict(
    #                 X=np.array([x_a, y_a, z_a]),
    #                 t=t,
    #                 params=dict(a=a, b=b, c=c, d=d),
    #             ),
        
    #         Lorenz=dict(
    #             X=np.array([x_l, y_l, z_l]),
    #             t=t,
    #             params=dict(beta=beta, rho=rho, sigma=sigma)
    #         ),
        
    #         LotkaVolterra=dict(
    #             X=np.array([x1, x2]),
    #             t=t,
    #             params=dict(alpha=alpha_lv, beta=beta_lv, gamma=gamma_lv, delta=delta_lv)
    #         )
    #     )

    #     jacs = np.zeros((*x_a.shape, 8, 8))

    #     for i, name in enumerate(self.names):
    #         jacs[..., i*3:min([(i+1)*3, 8]), :][..., i*3:min([(i+1)*3, 8])] = self.comm_weights[(name, name)]*np.array(self.eqs[name]._jac(*update_params[name]['X'], update_params[name]['t'], **update_params[name]['params']))

    #     for i, name1 in enumerate(self.names):
    #         for j, name2 in enumerate(self.names):
    #             if i != j and self.comm_strengths[i, j] != 0:
    #                 jacs[..., i*3:min([(i+1)*3, 8]), :][..., j*3:min([(j+1)*3, 8])] = self.comm_weights[(name1, name2)]
        
    #     return jacs

    def _jac(self, x_a, y_a, z_a, x_l, y_l, z_l, x1, x2, t, a, b, c, d,
            beta, rho, sigma, alpha_lv, beta_lv, gamma_lv, delta_lv):
        update_params = dict(
            Arneodo=dict(
                X=np.array([x_a, y_a, z_a]),
                t=t,
                params=dict(a=a, b=b, c=c, d=d),
            ),
            Lorenz=dict(
                X=np.array([x_l, y_l, z_l]),
                t=t,
                params=dict(beta=beta, rho=rho, sigma=sigma)
            ),
            LotkaVolterra=dict(
                X=np.array([x1, x2]),
                t=t,
                params=dict(alpha=alpha_lv, beta=beta_lv, gamma=gamma_lv, delta=delta_lv)
            )
        )

        # Create a zero array with the same dtype as the Jacobian blocks you expect
        jacs = np.zeros((*x_a.shape, 8, 8))

        # Update diagonal blocks in a single indexing call and ensure matching dtype
        for i, name in enumerate(self.names):
            jac_block = np.array(
                self.eqs[name]._jac(
                    *update_params[name]['X'],
                    update_params[name]['t'],
                    **update_params[name]['params']
                ),
                dtype=jacs.dtype
            )
            jacs[..., i*3:min((i+1)*3, 8), i*3:min((i+1)*3, 8)] = (
                self.comm_weights[(name, name)] * jac_block
            )

        # Update off-diagonal blocks in a single indexing call and cast to correct dtype
        for i, name1 in enumerate(self.names):
            for j, name2 in enumerate(self.names):
                if i != j and self.comm_strengths[i, j] != 0:
                    weight_block = np.array(self.comm_weights[(name1, name2)], dtype=jacs.dtype)
                    jacs[..., i*3:min((i+1)*3, 8), j*3:min((j+1)*3, 8)] = weight_block

        return jacs

def make_saveable(d):
    for key in d.keys():
        if isinstance(d[key], np.ndarray):
            d[key] = d[key].tolist()
        elif isinstance(d[key], dict):
            d[key] = make_saveable(d[key])
    
    return d

def make_metadata(eq_name,
                  embedding_dimension, 
                  parameters, 
                  initial_conditions, 
                  citation=None, 
                  delay=False, 
                  dt=0.001, 
                  nonautonomous=False,
                  positive_only=False,
                  vectorize=False,
                  sim_time=1, 
                  num_ics=25, 
                  new_ic_mode='reference',
                  standardize=True, 
                  transient_fraction=0.2, 
                  scale_by=2500, 
                  plot_stuff=True, 
                  override=False,
                  testing=False,
                  verbose=False,
                  PCA_dims=None,
                  traj_offset_sd=1,
                  plot_traj=False,
                  period=None,
                  **eq_kwargs):
    eq = eval(eq_name)(**eq_kwargs)
    with open(eq.data_path, "r") as read_file:
        data = json.load(read_file)

    if not override:
        if eq.name in data:
            print(f"Equation {eq.name} already in metdata. Nothing to be done!")
            return
    else:
        if not testing:
            print(f"Overriding: metadata will be made!")

    print(f"Making metadata for {eq_name}!")
    
    eq_data_dict = deepcopy(data_default)
    # leave bifurcation_parameter blank
    eq_data_dict['citation'] = citation
    # leave correlation_dimension blank
    eq_data_dict['delay'] = delay
    # leave description blank
    eq_data_dict['dt'] = dt
    eq_data_dict['embedding_dimension'] = embedding_dimension
    # leave hamiltonian blank
    eq_data_dict['initial_conditions'] = initial_conditions
    # leave kaplyan_yorke_dimension blank
    # leave lyapunov_spectrum_estimated blank
    # leave maximum_lyapunov_estimated blank
    # leave multiscale_entropy blank
    eq_data_dict['nonautonomous'] = nonautonomous
    eq_data_dict['parameters'] = parameters
    eq_data_dict['period'] = 1 # start period at 1 s for easier simulation
    # leave pesin_entropy blank
    # leave unbounded_indices blank
    eq_data_dict['positive_only'] = positive_only
    eq_data_dict['vectorize'] = vectorize
    
    if not testing:
        data[eq_name] = eq_data_dict
        sorted_data = {key: value for key, value in sorted(data.items())}
        
        sorted_data = make_saveable(sorted_data)
        
        # Writing data to JSON file
        with open(eq.data_path, "w") as json_file:
            json.dump(sorted_data, json_file)

        eq = eval(eq_name)(**eq_kwargs)
    else:
        eq.redo_init(eq_data_dict)

    print("Computing trajectory solutions...")

    # pts_per_period = int(eq.period/eq.dt)
    # make solution to compute period with
    sol = eq.make_trajectory(n_periods=sim_time, num_ics=num_ics, new_ic_mode=new_ic_mode, standardize=standardize, traj_offset_sd=traj_offset_sd, resample=False, verbose=verbose)
    print("Trajectory solutions identified!!")

    if plot_traj:
        plt.figure(figsize=(12, 5))
        for i in range(sol.shape[0]):
            plt.plot(np.arange(sol.shape[1])*dt, sol[i, :, 0])
        plt.xlabel('Time (s)')
        plt.title('Trajectory Solutions')
        plt.show()

    # evaluate period
    cutoff = int(transient_fraction*sol.shape[-2])
    sol = sol[:, cutoff:]

    sol = sol.transpose(1, 0, 2).reshape(sol.shape[1], -1) # reshape to (time, embedding_dimension*n_trials)
    dt = eq.dt
    print(dt)

    if PCA_dims is not None:
        pca = PCA(n_components=PCA_dims)
        pca.fit(sol)
        sol = pca.transform(sol)
    
    if period is not None:
        print(f"Period of {period} was provided by the user!")
    else:
        print("Finding period...")

        chosen_freqs = []
        smallest_freqs = []
        for comp in tqdm(sol.T, total=sol.T.shape[0]):
            freqs, amps = find_significant_frequencies(comp, surrogate_method='rs', fs=1/dt, return_amplitudes=True)
            chosen_freqs.append(freqs[np.argmax(np.abs(amps))])
            smallest_freqs.append(np.min(freqs))
        # print(chosen_freqs)
        period = 1/np.mean(chosen_freqs)

        print(f"Chosen period is {period} seconds")

        if plot_stuff:
            print("plotting stuff...")
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            freq_max = (1/period)*2
            freq_min = 0
            periods_all = None
            amps_all = None
            for comp in sol.T:
                amps = np.fft.rfft(comp)
                freqs = np.fft.rfftfreq(n=len(comp), d=dt)
                inds = (freqs < freq_max) & (freqs > freq_min)
                if periods_all is None:
                    periods_all = 1/freqs[inds]
                    amps_all = np.abs(amps)[inds]
                else:
                    periods_all += 1/freqs[inds]
                    amps_all += np.abs(amps)[inds]
            periods_all /= sol.T.shape[0]
            amps_all /= sol.T.shape[0]
            plt.plot(periods_all, amps_all, label='mean fft')
            plt.axvline(period, linestyle='--', c='pink', label='pred $t_{peak}$')
            plt.xscale('log')
            plt.xlabel('Log Period')
            plt.ylabel('Absolute Value FFT')
            plt.legend()

            plt.subplot(1, 2, 2)
            # sample_traj = eq.make_trajectory(n_periods=sim_time, num_ics=1, standardize=False, resample=False, verbose=verbose)
            # plt.plot(np.arange(sample_traj.shape[1])*dt, sample_traj[0, :, :])
            sample_traj = sol[:, :embedding_dimension]
            plt.plot(np.arange(sol.shape[0])*dt, sample_traj)
            plt.xlabel('Time (s)')
            t = period
            while t < sample_traj.shape[1]*dt:
                plt.axvline(t, linestyle='--', c='k')
                t += period
            plt.suptitle(f"{eq.name} Period Analysis")
            plt.show()
        
    eq_data_dict['period'] = period
    eq_data_dict['dt'] = period/scale_by
    
    if not testing:
        # Writing data to JSON file
        with open(eq.data_path, "w") as json_file:
            json.dump(sorted_data, json_file)
            json.dump(sorted_data, json_file)