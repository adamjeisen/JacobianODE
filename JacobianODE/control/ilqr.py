import hydra
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch import nn

from ..jacobians.jacobian_utils import make_wmtask_trajectories, generate_wmtask_data, load_run, load_checkpoint
from ..jacobians.jacobianODE import JacobianODE

class iLQR:
    """
    iLQR class that takes in continuous-time dynamics f_c(x,u) = dx/dt,
    as well as their Jacobians A_c(x,u), B_c(x,u).
    The discretization is done via a forward Euler scheme:
       x_{k+1} = x_k + dt * f_c(x_k, u_k)
       A_k = I + dt * A_c(x_k, u_k)
       B_k = dt * B_c(x_k, u_k)
    """
    def __init__(self, f_c, A_c_func, B_c_func,
                 Q, R, Qf,
                 x0, x_ref, u_init,
                 dt, T,
                 max_iter=100, tol=1e-6, reg_init=1.0, reg_min=1e-6, reg_max=1e10, _delta_0=2.0,
                #  alpha_list=[1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
                alpha_list = None,
                 ignore_first_n=0, max_failed_loops=20, verbose=False):
        """
        f_c: continuous-time dynamics function, f_c(x,u) -> dx/dt
        A_c_func: continuous-time Jacobian wrt x, A_c(x,u) = d f_c / d x
        B_c_func: continuous-time Jacobian wrt u, B_c(x,u) = d f_c / d u

        Q, R, Qf: cost weighting matrices for stage cost and terminal cost
        x0: initial state (n-dim tensor)
        x_ref: desired trajectory (length T+1). Each entry is n-dim
        u_init: initial control sequence (length T). Each entry is m-dim
        dt: discrete time step
        T: number of discrete steps
        max_iter, tol, reg_init, alpha_list: iLQR hyperparameters
        ignore_first_n: optionally ignore the stage cost for the first n steps
        reg_scale_factor: factor to scale regularization by when no cost reduction is found
        max_failed_loops: maximum number of failed loops before stopping
        """
        self.f_c = f_c
        self.A_c_func = A_c_func
        self.B_c_func = B_c_func

        self.Q = Q
        self.R = R
        self.Qf = Qf
        
        self.x0 = x0
        if isinstance(x_ref, torch.Tensor):
            self.x_ref = [x_ref[t] for t in range(x_ref.shape[0])]
        else:
            self.x_ref = x_ref

        if isinstance(u_init, torch.Tensor):
            self.u_nom = [u_init[t] for t in range(u_init.shape[0])]
        else:
            self.u_nom = u_init

        self.dt = dt
        self.T = T

        self.max_iter = max_iter
        self.tol = tol
        self.reg_init = reg_init
        self.reg = reg_init
        self.reg_min = reg_min
        self.reg_max = reg_max
        self._delta_0 = _delta_0
        self._delta = self._delta_0
        if alpha_list is None:
            self.alpha_list = 1.1**(-np.arange(10)**2)
        else:
            self.alpha_list = alpha_list
        self.ignore_first_n = ignore_first_n

        # To record convergence metrics and cost history
        self.cost_trace = []
        self.iterations = 0
        self.verbose = verbose
        self.max_failed_loops = max_failed_loops

    def forward_rollout(self, x0, u_seq):
        """
        Discrete forward rollout using forward Euler:
           x_{k+1} = x_k + dt * f_c(x_k, u_k).
        """
        x_seq = [x0.clone()]
        for t in range(self.T):
            x_current = x_seq[t]
            dxdt = self.f_c(x_current, u_seq[t])
            x_next = x_current + self.dt * dxdt
            x_seq.append(x_next)
        return x_seq
    
    def forward_pass(self, x_seq, u_seq, k_seq, K_seq, alpha):
        """
        Forward pass with line search parameter alpha.
        x_new[k+1] = x_new[k] + dt * f_c( x_new[k], u_new[k] )
        where u_new[k] = u_seq[k] + alpha*k_seq[k] + K_seq[k]*(x_new[k] - x_seq[k])
        """
        x_new = [self.x0.clone()]
        u_new = []
        for t in range(self.T):
            dx = x_new[t] - x_seq[t]
            du = u_seq[t] + alpha * k_seq[t] + K_seq[t] @ dx
            u_new.append(du)
            # Forward Euler
            x_next = x_new[t] + self.dt * self.f_c(x_new[t], du)
            x_new.append(x_next)
        cost_new = self.compute_cost(x_new, u_new)
        return x_new, u_new, cost_new

    def compute_cost(self, x_seq, u_seq):
        """
        Computes the total cost for a given trajectory:
           cost = sum_{t=0}^{T-1} [ (x - x_ref)^T Q (x - x_ref) + u^T R u ]
                 + (x_T - x_ref_T)^T Qf (x_T - x_ref_T)
        We optionally ignore the first 'ignore_first_n' steps in the stage cost.
        """
        cost = 0.0
        for t in range(self.T):
            if t < self.ignore_first_n:
                continue
            dx = x_seq[t] - self.x_ref[t]
            du = u_seq[t]
            stage_cost = dx @ self.Q @ dx + du @ self.R @ du
            cost += stage_cost
        # Terminal cost
        dx_terminal = x_seq[-1] - self.x_ref[-1]
        cost += dx_terminal @ self.Qf @ dx_terminal
        return cost

    def linearize_dynamics(self, x, u):
        """
        Discretize the Jacobians using forward Euler:
           A = I + dt * A_c(x, u)
           B = dt * B_c(x, u)
        """
        n = x.shape[0]
        I = torch.eye(n, dtype=x.dtype).to(x.device)
        A_c = self.A_c_func(x, u)
        B_c = self.B_c_func(x, u)
        A = I + self.dt * A_c
        B = self.dt * B_c
        return A, B

    def backward_pass(self, x_seq, u_seq):
        """
        Compute the backward pass to find feedforward (k) and feedback (K) gains.
        We approximate the cost and dynamics quadratically/linearly about (x_seq, u_seq).
        """
        n = self.x0.shape[0]
        m = u_seq[0].shape[0]
        T = self.T

        # Initialize value function at terminal time
        V_x = self.Qf @ (x_seq[-1] - self.x_ref[-1])
        V_xx = self.Qf.clone()

        k_seq = [None] * T
        K_seq = [None] * T

        delta_cost = 0.0

        # regu_I = self.reg * torch.eye(V_xx.shape[0]).to(V_xx.device)
        for t in reversed(range(T)):
            dx = x_seq[t] - self.x_ref[t]
            du = u_seq[t]
            l_x = self.Q @ dx
            l_u = self.R @ du
            l_xx = self.Q
            l_uu = self.R
            l_ux = torch.zeros((m, n), dtype=x_seq[0].dtype).to(x_seq[0].device)

            # Discretized Jacobians
            A, B = self.linearize_dynamics(x_seq[t], u_seq[t])

            # Q-function approximations
            Q_x = l_x + A.t() @ V_x
            Q_u = l_u + B.t() @ V_x
            Q_xx = l_xx + A.t() @ V_xx @ A
            Q_ux = l_ux + B.t() @ V_xx @ A
            Q_uu = l_uu + B.t() @ V_xx @ B

            # Regularization
            Q_ux_reg = Q_ux + B.t() @ (self.reg * torch.eye(n, dtype=Q_ux.dtype).to(Q_ux.device)) @ A
            Q_uu_reg = Q_uu + B.t() @ (self.reg * torch.eye(n, dtype=Q_uu.dtype).to(Q_uu.device)) @ B

            try:
                Q_uu_inv = torch.inverse(Q_uu_reg)
            except RuntimeError:
                if self.verbose:
                    print("Matrix inversion error in backward pass. Increasing regularization.")
                return None, None, None

            # Gains
            k = - Q_uu_inv @ Q_u
            # K = - Q_uu_inv @ Q_ux
            K = -Q_uu_inv @ Q_ux_reg

            k_seq[t] = k
            K_seq[t] = K

            # Update value function
            V_x = Q_x + K.t() @ Q_uu @ k + K.t() @ Q_u + Q_ux.t() @ k
            V_xx = Q_xx + K.t() @ Q_uu @ K + K.t() @ Q_ux + Q_ux.t() @ K
            # Symmetrize
            V_xx = 0.5 * (V_xx + V_xx.t())

            delta_cost += (k @ Q_u).item()
            # delta_cost += Q_u.t() @ k + 0.5 * k.t() @ Q_uu @ k

        return k_seq, K_seq, delta_cost
    

    def increase_regularization(self):
        self._delta = max(self._delta_0, self._delta*self._delta_0)
        self.reg = max(self.reg_min, self.reg*self._delta)
    
    def decrease_regularization(self):
        self._delta = min(1.0, self._delta) / self._delta_0
        self.reg *= self._delta
        if self.reg < self.reg_min:
            self.reg = 0.0
    def optimize(self):
        """
        Main iLQR optimization loop.
        Returns a dictionary with optimized control, states, gains, cost trace, etc.
        """
        self.reg = self.reg_init
        self._delta = self._delta_0
        # Initial rollout
        x_seq = self.forward_rollout(self.x0, self.u_nom)
        cost = self.compute_cost(x_seq, self.u_nom)
        self.cost_trace.append(cost.item())

        n_failed_loops = 0

        for it in range(self.max_iter):

            self.iterations = it + 1
            backward = self.backward_pass(x_seq, self.u_nom)
            if backward[0] is None:
                # self.reg *= self.reg_scale_factor
                self.increase_regularization()
                n_failed_loops += 1
                if n_failed_loops > self.max_failed_loops:
                    if self.verbose:
                        print("Too many failed loops. Exiting.")
                    break
                continue
            k_seq, K_seq, exp_red = backward

            found_alpha = False
            for alpha in self.alpha_list:
                x_new, u_new, cost_new = self.forward_pass(x_seq, self.u_nom, k_seq, K_seq, alpha)
                if cost_new < cost:
                    found_alpha = True
                    break

            if not found_alpha:
                # self.reg *= self.reg_scale_factor
                self.increase_regularization()
                if self.verbose:
                    print(f"Iteration {it}: no cost reduction found, increasing regularization to {self.reg}")
                n_failed_loops += 1
                if n_failed_loops > self.max_failed_loops:
                    if self.verbose:
                        print("Too many failed loops. Exiting.")
                    break
            else:
                n_failed_loops = 0
                # self.reg = max(self.reg / 1.6, 1e-6)
                self.decrease_regularization()
                cost_diff = cost - cost_new
                cost = cost_new
                x_seq = x_new
                self.u_nom = u_new
                self.cost_trace.append(cost.item())
                if self.verbose:
                    print(f"Iteration {it}: cost = {cost.item():.6f}, cost reduction = {cost_diff:.6f}")

                if abs(cost_diff) < self.tol:
                    if self.verbose:
                        print("Convergence reached!")
                    break
            if self.verbose:
                print(f"Iteration {it}: reg = {self.reg}, delta = {self._delta}")

        return {
            'u_opt': self.u_nom,
            'x_opt': x_seq,
            'k_seq': k_seq,
            'K_seq': K_seq,
            'V_trace': self.cost_trace,
            'iterations': self.iterations,
        }

def get_ilqr_rets(f_c, A_c, B_c, B, f_true, x0, x_ref, u_init, dt, T, Q, R, Qf, ignore_first_n, verbose=False):
    """Compute optimal control sequence and evaluate performance using iLQR.

    This function runs iLQR optimization to find the optimal control sequence for a given dynamics model,
    and evaluates its performance by comparing controlled and uncontrolled trajectories.

    Args:
        f_c (Callable): Continuous-time dynamics function, f_c(x, u, B) -> dx/dt
        A_c (Callable): Continuous-time Jacobian wrt x, A_c(x, u) = d f_c / d x
        B_c (Callable): Continuous-time Jacobian wrt u, B_c(x, u, B) = d f_c / d u
        B (torch.Tensor): Control input matrix
        f_true (Callable): True dynamics function for simulation, f_true(x, u, B)
        x0 (torch.Tensor): Initial state vector
        x_ref (torch.Tensor): Reference trajectory (T+1 x n)
        u_init (List[torch.Tensor]): Initial control sequence (length T)
        dt (float): Discrete time step
        T (int): Number of discrete time steps
        Q (torch.Tensor): State cost weighting matrix
        R (torch.Tensor): Control cost weighting matrix
        Qf (torch.Tensor): Terminal state cost weighting matrix
        ignore_first_n (int): Number of initial steps to ignore in cost calculation
        verbose (bool): Whether to print detailed optimization progress

    Returns:
        dict: A dictionary containing:
            - 'x_uncontrolled': Uncontrolled trajectory (T+1 x n)
            - 'x_controlled': Controlled trajectory (T+1 x n)
            - 'u_opt': Optimal control sequence (list of length T)
            - 'cost_trace': Trace of cost values during optimization
    """
    # Instantiate iLQR with continuous-time dynamics
    n = x0.shape[0]
    m = u_init[0].shape[0]
    ilqr_solver = iLQR(
        f_c=lambda x, u, B: f_c(x, u, B),
        A_c_func=lambda x, u: A_c(x, u),
        B_c_func=lambda x, u, B: B_c(x, u, B),
        Q=Q,
        R=R,
        Qf=Qf,
        x0=x0,
        x_ref=x_ref,
        u_init=u_init,
        dt=dt,
        T=T,
        max_iter=100,
        tol=1e-6,
        reg_init=1.0,
        ignore_first_n=ignore_first_n,
        verbose=verbose
    )

    # Run iLQR
    results = ilqr_solver.optimize()

    # Extract results
    u_opt = [u.cpu() for u in results['u_opt']]
    x_opt = [x.cpu() for x in results['x_opt']]
    cost_trace = results['V_trace']
    B = B.cpu()
    x_ref = x_ref.cpu()

    x_uncontrolled = torch.zeros(T + 1, n)
    x_controlled = torch.zeros(T + 1, n)
    x_uncontrolled[0] = x0
    x_controlled[0] = x0
    for t in range(1, T + 1):
        x_uncontrolled[t] = x_uncontrolled[t-1] + dt * f_true(x_uncontrolled[t-1], torch.zeros(m), B)
        x_controlled[t] = x_controlled[t-1] + dt * f_true(x_controlled[t-1], u_opt[t-1], B)
    
    return {'x_uncontrolled': x_uncontrolled, 'x_controlled': x_controlled, 'u_opt': u_opt, 'cost_trace': cost_trace}

def get_wmtask_data_and_models(cfg=None, load_models=True):
    """Load and prepare data and models for the working memory task analysis.

    This function loads the working memory task data, trajectories, and trained models (JacobianODE and NeuralODE).
    It handles the initialization of models, loading of checkpoints, and preparation of data loaders.

    Args:
        cfg (Optional[Dict]): Configuration dictionary. If None, loads default config from hydra.
        load_models (bool): Whether to load the trained models (JacobianODE and NeuralODE).
                           If False, returns None for model-related fields.

    Returns:
        dict: A dictionary containing:
            - 'eq': The equation object containing the true dynamics
            - 'sol': Solution dictionary containing trajectories and parameters
            - 'dt': Time step size
            - 'all_dataloader': DataLoader containing all trajectories
            - 'lit_model': Trained JacobianODE model (None if load_models=False)
            - 'lit_model_node': Trained NeuralODE model (None if load_models=False)
            - 'cfg_mlp': Configuration for the JacobianODE model
            - 'cfg_neuralode': Configuration for the NeuralODE model
            - 'trajs': Dictionary containing training and test trajectories
    """
    if cfg is None:
        # load the config
        with hydra.initialize(version_base="1.3", config_path="../jacobians/conf"):
            cfg = hydra.compose(config_name="config")

    eq, sol, dt = make_wmtask_trajectories(cfg, verbose=True)

    all_dataloader, train_dataloader, val_dataloader, test_dataloader = generate_wmtask_data(sol['params'])

    if load_models:
        # JacobianODE + NODE
        project = 'WMTask__JacobianEstimation'
        # id_val = '0z6qaja4' # plain old jacobianODE
        id_val = cfg.mlp_id
        run, cfg_mlp, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model = load_run(project, id_val, no_noise=False, verbose=True)
        epoch = None
        load_checkpoint(run, cfg_mlp, lit_model, save_dir=None, epoch=epoch, verbose=True)

        # id_val = '8g6vanvm' # neural ode
        id_val = cfg.neuralode_id
        run, cfg_neuralode, eq, dt, values, train_dataloader, val_dataloader, test_dataloader, trajs, lit_model_node = load_run(project, id_val, no_noise=False, verbose=True)
        epoch = None
        load_checkpoint(run, cfg_neuralode, lit_model_node, save_dir=None, epoch=epoch, verbose=True)
    else:
        lit_model = None
        lit_model_node = None
        cfg_mlp = None
        cfg_neuralode = None
        trajs = None

    return {'eq': eq, 'sol': sol, 'dt': dt, 'all_dataloader': all_dataloader, 'lit_model': lit_model, 'lit_model_node': lit_model_node, 'cfg_mlp': cfg_mlp, 'cfg_neuralode': cfg_neuralode, 'trajs': trajs}

def perform_wmtask_ilqr_analysis(cfg, wmtask_data_and_models={}, verbose=False):
    """Perform iLQR analysis on the working memory task using different dynamics models.

    This function runs iLQR optimization using different dynamics models (true dynamics, JacobianODE,
    NeuralODE, and a linear baseline) to control the system to reach different goal states.
    It evaluates the performance of each model in terms of MSE and classification accuracy.

    Args:
        cfg (Dict): Configuration dictionary containing:
            - traj_ind: Index of the test trajectory to use
            - Q_weight: Weight for state cost matrix
            - R_weight: Weight for control cost matrix
            - Qf_weight: Weight for terminal state cost matrix
            - extra_time_points: Number of extra time points to consider ('all' or int)
            - accuracy_inds: Whether to compute accuracy on 'final' state or 'all' response times
            - use_line_for_ref: Whether to use linear interpolation for reference trajectory
        wmtask_data_and_models (Dict): Dictionary containing data and models from get_wmtask_data_and_models.
                                      If empty, will call get_wmtask_data_and_models.
        verbose (bool): Whether to print detailed progress information.

    Returns:
        dict: A dictionary containing results for each model type and goal label:
            - 'true_rets': Results using true dynamics
            - 'jac_rets': Results using JacobianODE model
            - 'jac_f_est_rets': Results using JacobianODE with estimated dynamics
            - 'node_rets': Results using NeuralODE model
            - 'baseline_rets': Results using linear baseline model
            
            Each model's results (e.g., true_rets) is a dictionary keyed by goal_label containing:
            - 'x_uncontrolled': Uncontrolled trajectory
            - 'x_controlled': Controlled trajectory
            - 'u_opt': Optimal control sequence
            - 'cost_trace': Cost values during optimization
            - '*_uncontrolled_mse': MSE of uncontrolled trajectory
            - '*_controlled_mse': MSE of controlled trajectory
            - '*_uncontrolled_acc': Classification accuracy of uncontrolled trajectory
            - '*_controlled_acc': Classification accuracy of controlled trajectory
    """
    if not wmtask_data_and_models:
        wmtask_data_and_models = get_wmtask_data_and_models(cfg)

    eq = wmtask_data_and_models['eq']
    sol = wmtask_data_and_models['sol']
    dt = wmtask_data_and_models['dt']
    all_dataloader = wmtask_data_and_models['all_dataloader']
    lit_model = wmtask_data_and_models['lit_model']
    lit_model_node = wmtask_data_and_models['lit_model_node']
    
    def f_wmtask_true(x, u, B):
        """Compute the true dynamics of the working memory task.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input
            B (torch.Tensor): Control input matrix

        Returns:
            torch.Tensor: State derivative dx/dt
        """
        return eq.rhs(x) + B @ u

    def A_wmtask_true(x, u):
        """Compute the true Jacobian of the working memory task dynamics.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input (unused, kept for interface consistency)

        Returns:
            torch.Tensor: Jacobian matrix d f_c / d x
        """
        return eq.jac(x).squeeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def f_wmtask_jac(x, u, B, use_node=True, f_est=None):
        """Compute the dynamics using either NeuralODE or estimated dynamics from JacobianODE.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input
            B (torch.Tensor): Control input matrix
            use_node (bool): Whether to use NeuralODE (True) or estimated dynamics (False)
            f_est (Optional[Callable]): Function to estimate dynamics if use_node is False

        Returns:
            torch.Tensor: State derivative dx/dt

        Raises:
            ValueError: If use_node is False and f_est is not provided
        """
        if lit_model_node is not None and use_node:
            with torch.no_grad():
                return lit_model_node.model(torch.tensor(0.0).to(device), x).squeeze(0) + B @ u
        else:
            with torch.no_grad():
                if f_est is None:
                    raise ValueError("f_est must be provided if use_node is False")
                return f_est(torch.tensor(0.0).to(device), x).squeeze(0) + B @ u

    def f_wmtask_node(x, u, B):
        """Compute the dynamics using the NeuralODE model.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input
            B (torch.Tensor): Control input matrix

        Returns:
            torch.Tensor: State derivative dx/dt
        """
        with torch.no_grad():
            return lit_model_node.model(torch.tensor(0.0).to(device), x).squeeze(0) + B @ u

    def A_wmtask_jac(x, u):
        """Compute the Jacobian using the JacobianODE model.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input (unused, kept for interface consistency)

        Returns:
            torch.Tensor: Jacobian matrix d f_c / d x
        """
        with torch.no_grad():
            return lit_model.compute_jacobians(x).squeeze(0)

    def A_wmtask_node(x, u):
        """Compute the Jacobian using the NeuralODE model.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input (unused, kept for interface consistency)

        Returns:
            torch.Tensor: Jacobian matrix d f_c / d x
        """
        with torch.no_grad():
            return lit_model_node.compute_jacobians(x).squeeze(0)

    def B_wmtask(x, u, B):
        """Return the control input matrix (constant for this system).

        Args:
            x (torch.Tensor): Current state (unused, kept for interface consistency)
            u (torch.Tensor): Control input (unused, kept for interface consistency)
            B (torch.Tensor): Control input matrix

        Returns:
            torch.Tensor: Control input matrix B
        """
        return B

    # Compute linear baseline dynamics using least squares
    X_minus = sol['values'][:, :-1].reshape(-1, sol['values'].shape[-1])
    X_plus = sol['values'][:, 1:].reshape(-1, sol['values'].shape[-1])
    # linear regression
    A = torch.linalg.lstsq(X_minus.to(device), X_plus.to(device)).solution.T

    def f_wmtask_baseline(x, u, B):
        """Compute the dynamics using a linear model fit to the data.

        Args:
            x (torch.Tensor): Current state
            u (torch.Tensor): Control input
            B (torch.Tensor): Control input matrix

        Returns:
            torch.Tensor: State derivative dx/dt
        """
        return A @ x + B @ u

    def A_wmtask_baseline(x, u):
        """Return the constant Jacobian matrix for the linear baseline model.

        Args:
            x (torch.Tensor): Current state (unused, kept for interface consistency)
            u (torch.Tensor): Control input (unused, kept for interface consistency)

        Returns:
            torch.Tensor: Constant Jacobian matrix A
        """
        return A


    lit_model = lit_model.to(device)
    lit_model_node = lit_model_node.to(device)
    # sol['values'] = sol['values'].to(device)
    test_trajs = wmtask_data_and_models['trajs']['test_trajs'].sequence
    test_trajs = test_trajs.to(device)
    train_trajs = wmtask_data_and_models['trajs']['train_trajs'].sequence
    train_trajs = train_trajs.to(device)
    train_inds = wmtask_data_and_models['trajs']['train_inds']

    # Time parameters
    T = sol['values'].shape[-2] - 1      # number of time steps
    dt = sol['dt']       # time step size

    if cfg.extra_time_points == 'all':
        extra_time_points = T - all_dataloader.dataset.n_response_t
    else:
        extra_time_points = cfg.extra_time_points
    ignore_first_n = T - all_dataloader.dataset.n_response_t - extra_time_points
    if verbose: 
        print(f"ignore_first_n: {ignore_first_n}")

    total_time = T * dt
    time_points = torch.linspace(0, total_time, T+1).to(device)

    d = sol['values'].shape[-1]

    # B = torch.eye(d).to(device)
    B = torch.zeros(d, d).to(device)
    # B[torch.arange(sol['params']['N1'], sol['params']['N1'] + sol['params']['N2']), torch.arange(sol['params']['N1'], sol['params']['N1'] + sol['params']['N2'])] = 1.0
    B[torch.arange(sol['params']['N1']), torch.arange(sol['params']['N1'])] = 1.0

    # with torch.no_grad():   
    #     B = eq.model.W_hi[:, :4]
    #     B[torch.arange(sol['params']['N1'])] = 0.0
    #     B = B.to(device)

    # Dimensions
    n = d   # state dimension [position, velocity]
    m = B.shape[-1]   # control dimension [acceleration]

    # Define cost weights (quadratic cost)
    Q = cfg.Q_weight * torch.eye(n).to(device) # Q = 1.0 is good !!!!!!
    # Q[:sol['params']['N1'], :][:, :sol['params']['N1']] = 0.0
    # R = 0.0001 * torch.eye(m).to(device) # R = 0.01 is good !!!!!!
    R = cfg.R_weight * torch.eye(m).to(device) # R = 0.01 is good !!!!!!
    # R = 0.0 * torch.eye(m).to(device)
    # Qf = 10.0 * torch.eye(n).to(device) # Qf = 10.0 is good !!!!!!
    Qf = cfg.Qf_weight * torch.eye(n).to(device) # Qf = 10.0 is good !!!!!!
    # Qf[:sol['params']['N1'], :][:, :sol['params']['N1']] = 0.0

    accuracy_inds = cfg.accuracy_inds
    use_line_for_ref = cfg.use_line_for_ref
    if use_line_for_ref:
        ignore_first_n = 0

    true_rets = {}
    jac_rets = {}
    jac_f_est_rets = {}
    node_rets = {}
    baseline_rets = {}

    # labels_to_use = [i for i in range(4) if i != all_dataloader.dataset.labels[cfg.traj_ind]]
    all_dataloader_ind = wmtask_data_and_models['trajs']['test_inds'][cfg.traj_ind]
    labels_to_use = [i for i in range(4) if i != all_dataloader.dataset.labels[all_dataloader_ind].item()]
    for goal_label in labels_to_use:

        # Initial state
        # x0 = sol['values'][cfg.traj_ind, 0] + torch.randn(d).to(device) * (sol['values'].std()*0.05/np.sqrt(d))
        # x0 = test_trajs[cfg.traj_ind, 0] + torch.randn(d).to(device) * (sol['values'].std()*0.05/np.sqrt(d))
        x0 = test_trajs[cfg.traj_ind, 0]
        # x0 = sol['values'][traj_ind, 0] + torch.randn(d).to(device) * sol['values'].std()*0

        # final_state = sol['values'][:, -1, :][all_dataloader.dataset.labels == goal_label].mean(axis=0).to(device)
        final_state = train_trajs[:, -1, :][all_dataloader.dataset.labels[train_inds] == goal_label].mean(axis=0).to(device)
        x_ref = torch.zeros(T + 1, n).to(device)
        x_ref[-all_dataloader.dataset.n_response_t-extra_time_points:] = final_state
        if use_line_for_ref:
            line_t_vals = torch.linspace(0, 1, x_ref.shape[0] - all_dataloader.dataset.n_response_t-extra_time_points).to(device)
            x_ref[:-all_dataloader.dataset.n_response_t-extra_time_points] = x0.unsqueeze(0) + line_t_vals.unsqueeze(1) * (final_state.unsqueeze(0) - x0.unsqueeze(0))

        # Nominal control sequence
        u_init = [0.001 * torch.randn(m).to(device) for _ in range(T)]

        # jacobian_ode = JacobianODE(sol['values'][cfg.traj_ind].unsqueeze(0), jac_func=lambda x, t: lit_model.compute_jacobians(x), dt=sol['dt'])
        norms = torch.linalg.norm(wmtask_data_and_models['trajs']['train_trajs'].sequence[:, 0] - x0.cpu(), dim=-1)
        closest_traj_ind = norms.argmin()
        seeding_traj = wmtask_data_and_models['trajs']['train_trajs'][closest_traj_ind].unsqueeze(0)[:, :15].to(device)
        jacobian_ode = JacobianODE(seeding_traj, jac_func=lambda x, t: lit_model.compute_jacobians(x), dt=sol['dt'])
        f_est = jacobian_ode.get_deriv_func(interp_pts=4, inner_N=20)

        

        true_rets[goal_label] = get_ilqr_rets(f_wmtask_true, A_wmtask_true, B_wmtask, B, f_wmtask_true, x0, x_ref, u_init, dt, T, Q, R, Qf, ignore_first_n, verbose=verbose)
        jac_rets[goal_label] = get_ilqr_rets(f_wmtask_jac, A_wmtask_jac, B_wmtask, B, f_wmtask_true, x0, x_ref, u_init, dt, T, Q, R, Qf, ignore_first_n, verbose=verbose)
        jac_f_est_rets[goal_label] = get_ilqr_rets(lambda x, u, B: f_wmtask_jac(x, u, B, use_node=False, f_est=f_est), A_wmtask_jac, B_wmtask, B, f_wmtask_true, x0, x_ref, u_init, dt, T, Q, R, Qf, ignore_first_n, verbose=verbose)
        node_rets[goal_label] = get_ilqr_rets(f_wmtask_node, A_wmtask_node, B_wmtask, B, f_wmtask_true, x0, x_ref, u_init, dt, T, Q, R, Qf, ignore_first_n, verbose=verbose)
        baseline_rets[goal_label] = get_ilqr_rets(f_wmtask_baseline, A_wmtask_baseline, B_wmtask, B, f_wmtask_true, x0, x_ref, u_init, dt, T, Q, R, Qf, ignore_first_n, verbose=verbose)


        true_rets[goal_label]['true_uncontrolled_mse'] = torch.mean((true_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        jac_rets[goal_label]['jac_uncontrolled_mse'] = torch.mean((jac_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        jac_f_est_rets[goal_label]['jac_f_est_uncontrolled_mse'] = torch.mean((jac_f_est_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        node_rets[goal_label]['node_uncontrolled_mse'] = torch.mean((node_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        baseline_rets[goal_label]['baseline_uncontrolled_mse'] = torch.mean((baseline_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)

        true_rets[goal_label]['true_controlled_mse'] = torch.mean((true_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        jac_rets[goal_label]['jac_controlled_mse'] = torch.mean((jac_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        jac_f_est_rets[goal_label]['jac_f_est_controlled_mse'] = torch.mean((jac_f_est_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        node_rets[goal_label]['node_controlled_mse'] = torch.mean((node_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)
        baseline_rets[goal_label]['baseline_controlled_mse'] = torch.mean((baseline_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:] - x_ref[-all_dataloader.dataset.n_response_t:].cpu())**2)

        W_oh = eq.model.W_oh * eq.model.output_mask

        if accuracy_inds == 'final':
            true_uncontrolled_preds = nn.Softmax(dim=-1)(true_rets[goal_label]['x_uncontrolled'][-1]@ W_oh.T).argmax(dim=-1)
            true_controlled_preds = nn.Softmax(dim=-1)(true_rets[goal_label]['x_controlled'][-1]@ W_oh.T).argmax(dim=-1)

            jac_uncontrolled_preds = nn.Softmax(dim=-1)(jac_rets[goal_label]['x_uncontrolled'][-1]@ W_oh.T).argmax(dim=-1)
            jac_controlled_preds = nn.Softmax(dim=-1)(jac_rets[goal_label]['x_controlled'][-1]@ W_oh.T).argmax(dim=-1)

            jac_f_est_uncontrolled_preds = nn.Softmax(dim=-1)(jac_f_est_rets[goal_label]['x_uncontrolled'][-1]@ W_oh.T).argmax(dim=-1)
            jac_f_est_controlled_preds = nn.Softmax(dim=-1)(jac_f_est_rets[goal_label]['x_controlled'][-1]@ W_oh.T).argmax(dim=-1)

            node_uncontrolled_preds = nn.Softmax(dim=-1)(node_rets[goal_label]['x_uncontrolled'][-1]@ W_oh.T).argmax(dim=-1)
            node_controlled_preds = nn.Softmax(dim=-1)(node_rets[goal_label]['x_controlled'][-1]@ W_oh.T).argmax(dim=-1)

            baseline_uncontrolled_preds = nn.Softmax(dim=-1)(baseline_rets[goal_label]['x_uncontrolled'][-1]@ W_oh.T).argmax(dim=-1)
            baseline_controlled_preds = nn.Softmax(dim=-1)(baseline_rets[goal_label]['x_controlled'][-1]@ W_oh.T).argmax(dim=-1)

            n_preds = 1

        elif accuracy_inds == 'all':
            true_uncontrolled_preds = nn.Softmax(dim=-1)(true_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)
            true_controlled_preds = nn.Softmax(dim=-1)(true_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)

            jac_uncontrolled_preds = nn.Softmax(dim=-1)(jac_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)
            jac_controlled_preds = nn.Softmax(dim=-1)(jac_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)

            jac_f_est_uncontrolled_preds = nn.Softmax(dim=-1)(jac_f_est_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)
            jac_f_est_controlled_preds = nn.Softmax(dim=-1)(jac_f_est_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)

            node_uncontrolled_preds = nn.Softmax(dim=-1)(node_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)
            node_controlled_preds = nn.Softmax(dim=-1)(node_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)

            baseline_uncontrolled_preds = nn.Softmax(dim=-1)(baseline_rets[goal_label]['x_uncontrolled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)
            baseline_controlled_preds = nn.Softmax(dim=-1)(baseline_rets[goal_label]['x_controlled'][-all_dataloader.dataset.n_response_t:]@ W_oh.T).argmax(dim=-1)

            n_preds = all_dataloader.dataset.n_response_t
        else:
            raise ValueError(f"Invalid accuracy_inds: {accuracy_inds}")

        true_uncontrolled_acc = torch.sum(true_uncontrolled_preds == goal_label).cpu().item()/n_preds
        true_controlled_acc = torch.sum(true_controlled_preds == goal_label).cpu().item()/n_preds
        jac_uncontrolled_acc = torch.sum(jac_uncontrolled_preds == goal_label).cpu().item()/n_preds
        jac_controlled_acc = torch.sum(jac_controlled_preds == goal_label).cpu().item()/n_preds
        jac_f_est_uncontrolled_acc = torch.sum(jac_f_est_uncontrolled_preds == goal_label).cpu().item()/n_preds
        jac_f_est_controlled_acc = torch.sum(jac_f_est_controlled_preds == goal_label).cpu().item()/n_preds
        node_uncontrolled_acc = torch.sum(node_uncontrolled_preds == goal_label).cpu().item()/n_preds
        node_controlled_acc = torch.sum(node_controlled_preds == goal_label).cpu().item()/n_preds
        baseline_uncontrolled_acc = torch.sum(baseline_uncontrolled_preds == goal_label).cpu().item()/n_preds
        baseline_controlled_acc = torch.sum(baseline_controlled_preds == goal_label).cpu().item()/n_preds

        true_rets[goal_label]['true_uncontrolled_acc'] = true_uncontrolled_acc
        true_rets[goal_label]['true_controlled_acc'] = true_controlled_acc
        jac_rets[goal_label]['jac_uncontrolled_acc'] = jac_uncontrolled_acc
        jac_rets[goal_label]['jac_controlled_acc'] = jac_controlled_acc
        jac_f_est_rets[goal_label]['jac_f_est_uncontrolled_acc'] = jac_f_est_uncontrolled_acc
        jac_f_est_rets[goal_label]['jac_f_est_controlled_acc'] = jac_f_est_controlled_acc
        node_rets[goal_label]['node_uncontrolled_acc'] = node_uncontrolled_acc
        node_rets[goal_label]['node_controlled_acc'] = node_controlled_acc
        baseline_rets[goal_label]['baseline_uncontrolled_acc'] = baseline_uncontrolled_acc
        baseline_rets[goal_label]['baseline_controlled_acc'] = baseline_controlled_acc
    
    return {'true_rets': true_rets, 'jac_rets': jac_rets, 'jac_f_est_rets': jac_f_est_rets, 'node_rets': node_rets, 'baseline_rets': baseline_rets}