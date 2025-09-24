import numpy as np
import torch

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