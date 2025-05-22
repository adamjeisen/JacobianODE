"""
Implementation of Jacobian ODE models for learning and simulating dynamical systems.
This module provides tools for:
1. Computing Jacobian-based ODE solutions
2. Generating piecewise spline interpolations of trajectories
3. Integrating ODEs using various numerical methods
4. Simulating system dynamics with teacher forcing
"""

import numpy as np
import torch
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from tqdm.auto import tqdm

from ..torchquad import Simpson, Trapezoid, GaussLegendre
from .torchcubicspline_pw import PiecewiseCubicSpline
from .paths import c_line, c_prime_line

def generate_pw_spline(traj, time_vals=None, dt=None):
    """Generate piecewise cubic spline interpolation of a trajectory.
    
    Creates a continuous representation of a discrete trajectory using piecewise
    cubic splines, along with its derivative.
    
    Args:
        traj (torch.Tensor): Trajectory data of shape (..., time_steps, dims)
        time_vals (torch.Tensor, optional): Time points corresponding to trajectory. 
            If None, will be generated using dt. Defaults to None.
        dt (float, optional): Time step size. Required if time_vals is None. Defaults to None.
        
    Returns:
        tuple: (c_pw, c_prime_pw) where:
            - c_pw: Function that evaluates the spline at given time points
            - c_prime_pw: Function that evaluates the spline derivative at given time points
            
    Raises:
        ValueError: If neither time_vals nor dt is provided
    """
    if time_vals is None:
        if dt is None:
            raise ValueError("dt must be provided if time_vals is not provided")
        time_vals = torch.arange(traj.shape[-2]).type(traj.dtype).to(traj.device)*dt
    else:
        time_vals = time_vals

    spline_pw = PiecewiseCubicSpline(time_vals, traj, bc_type='natural')
    c_pw = lambda t: spline_pw.evaluate(t.squeeze(-1).type(traj.dtype).to(traj.device))
    c_prime_pw = lambda t: spline_pw.derivative(t.squeeze(-1).type(traj.dtype).to(traj.device))
    return c_pw, c_prime_pw

class JacobianODE:
    """Class for computing Jacobian-based ODE solutions.
    
    This class implements methods for computing solutions to ODEs using Jacobian
    information, with support for various integration methods and trajectory
    interpolation techniques.
    
    Attributes:
        traj (torch.Tensor): The trajectory data
        time_vals (torch.Tensor): Time points for the trajectory
        dt (float): Time step size
        jac_func (callable): Function that computes the Jacobian
        int_method (str): Integration method to use ('Simpson', 'Trapezoid', 'GaussLegendre', or 'odeint')
        int_func (object): Integration function object
        c_spline (callable): Spline interpolation function for trajectory
        c_prime_spline (callable): Spline interpolation function for trajectory derivative
        x_t (torch.Tensor): Current state
        x_t_0 (torch.Tensor): Initial state
        sigma (float): Scaling factor for the Jacobian
    """
    
    def __init__(self, traj, jac_func, time_vals=None, dt=None, int_method='Trapezoid', fit_spline=True, sigma=1, spline_skip_pts=1):
        """Initialize the JacobianODE object.
        
        Args:
            traj (torch.Tensor): Trajectory data of shape (..., time_steps, dims)
            jac_func (callable): Function that computes the Jacobian matrix
            time_vals (torch.Tensor, optional): Time points for trajectory. Defaults to None.
            dt (float, optional): Time step size. Required if time_vals is None. Defaults to None.
            int_method (str, optional): Integration method to use. Defaults to 'Trapezoid'.
            fit_spline (bool, optional): Whether to fit spline to trajectory. Defaults to True.
            sigma (float, optional): Scaling factor for Jacobian. Defaults to 1.
            spline_skip_pts (int, optional): Number of points to skip when fitting spline. Defaults to 1.
            
        Raises:
            ValueError: If neither time_vals nor dt is provided, or if invalid integration method
        """
        self.traj = traj
        if time_vals is None:
            if dt is None:
                raise ValueError("dt must be provided if time_vals is not provided")
            self.time_vals = torch.arange(traj.shape[-2]).type(traj.dtype).to(traj.device)*dt
        else:
            self.time_vals = time_vals
        self.dt = (self.time_vals[1] - self.time_vals[0]).item()
        self.jac_func = jac_func
        self.int_method = int_method
        if int_method == "Simpson":
            self.int_func = Simpson()
        elif int_method == "Trapezoid":
            self.int_func = Trapezoid()
        elif int_method == "GaussLegendre":
            self.int_func = GaussLegendre()
        elif int_method == "odeint":
            self.int_func = odeint
        else:
            raise ValueError("Invalid integration int_method: {}, must be 'Simpson', 'Trapezoid', 'GaussLegendre'".format(int_method))

        if fit_spline:
            if time_vals is None:
                time_vals = torch.arange(traj.shape[-2]).type(traj.dtype).to(traj.device)*dt
            self.c_spline, self.c_prime_spline = generate_pw_spline(traj[..., ::spline_skip_pts, :], time_vals=time_vals[::spline_skip_pts], dt=dt)
        else:
            self.c_spline, self.c_prime_spline = None, None

        self.x_t = None
        self.x_t_0 = None
        self.sigma = sigma

    def inner_integrand(self, r, c, c_prime):
        """Compute the inner integrand for the Jacobian ODE.
        
        Args:
            r (torch.Tensor): Time point
            c (callable): Trajectory interpolation function
            c_prime (callable): Trajectory derivative interpolation function
            
        Returns:
            torch.Tensor: The inner integrand value
        """
        if not self.int_method == "odeint":
            ret_vec = (self.jac_func(c(r), r) @ c_prime(r).unsqueeze(-1)).squeeze(-1)
            if len(ret_vec.shape) == 3: # convert from batches x time x dims to time x batches x dims
                ret_vec = ret_vec.permute(1, 0, 2)
            if len(ret_vec.shape) == 4:
                ret_vec = ret_vec.permute(2, 0, 1, 3)
        else:
            ret_vec = (self.jac_func(c(r).squeeze(-2), r) @ c_prime(r).unsqueeze(-1)).squeeze(-1)
        return ret_vec

    def H(self, s, t, x_s=None, x_t=None, inner_path="spline", N=2, return_all=False, c_spline=None, c_prime_spline=None, c=None, c_prime=None):
        """Compute the H integral for the Jacobian ODE.
        
        This function computes the integral of the Jacobian along a path between
        points s and t, using either spline or linear interpolation.
        
        Args:
            s (torch.Tensor): Start time
            t (torch.Tensor): End time
            x_s (torch.Tensor, optional): Start state. Defaults to None.
            x_t (torch.Tensor, optional): End state. Defaults to None.
            inner_path (str, optional): Path type ('spline' or 'line'). Defaults to "spline".
            N (int, optional): Number of integration points. Defaults to 2.
            return_all (bool, optional): Whether to return all integration points. Defaults to False.
            c_spline (callable, optional): Spline function. Defaults to None.
            c_prime_spline (callable, optional): Spline derivative function. Defaults to None.
            c (callable, optional): Custom path function. Defaults to None.
            c_prime (callable, optional): Custom path derivative function. Defaults to None.
            
        Returns:
            torch.Tensor: The H integral value
        """
        if s == t:
            return torch.zeros_like(self.x_t_0)
        int_func_kwargs = {"N": int(N)}

        if c_spline is None:
            c_spline = self.c_spline
        if c_prime_spline is None:
            c_prime_spline = self.c_prime_spline
        
        if x_s is None:
            x_s = self.c_spline(s)
        if x_t is None:
            if self.x_t is None:
                x_t = self.c_spline(t)
            else:
                x_t = self.x_t

        if c is None or c_prime is None:
            if inner_path == "line":
                c = lambda _t: c_line(_t.type(x_s.dtype).to(x_s.device), x_s, x_t, s, t)
                c_prime = lambda _t: c_prime_line(_t.type(x_s.dtype).to(x_s.device), x_s, x_t, s, t)
            elif inner_path == "spline":
                c = c_spline
                c_prime = c_prime_spline
            else:
                raise ValueError("Invalid inner_path: {}, must be 'spline' or 'line'".format(inner_path))
        if self.int_method == "odeint":
            # return self.int_func(lambda t, x: self.inner_integrand(t, c, c_prime), torch.zeros(x_s.shape).type(x_s.dtype).to(x_s.device), torch.tensor([s, t]).type(x_s.dtype).to(x_s.device),rtol=1e-3, atol=1e-3, method='rk4')[-1]
            odeint_result = self.int_func(lambda _t, x: self.inner_integrand(_t, c, c_prime), torch.zeros(x_s.shape).type(x_s.dtype).to(x_s.device), torch.linspace(s, t, N).type(x_s.dtype).to(x_s.device), method='rk4')
            if return_all:
                ret = odeint_result
            else:
                ret = odeint_result[-1]
        else:
            integration_domain = torch.stack((s.type(x_s.dtype).to(x_s.device), t.type(x_s.dtype).to(x_s.device))).unsqueeze(0)
            grid_points, hs, n_per_dim = self.int_func.get_grid_points(dim=1, integration_domain=integration_domain, **int_func_kwargs)
            fn_result = self.inner_integrand(grid_points.type(x_s.dtype).to(x_s.device), c, c_prime)
            ret = self.int_func.integrate_from_pts(fn_result, dim=1, hs=hs, n_per_dim=n_per_dim, integration_domain=integration_domain, return_all=return_all)
        
        return ret*(self.sigma)

    def G(self, t_0, t, inner_path="spline",  N=2, inner_N=2, c_spline=None, c_prime_spline=None):
        """Compute the G integral for the Jacobian ODE.
        
        This function computes the double integral G(t_0, t) which represents
        the cumulative effect of the Jacobian along a path.
        
        Args:
            t_0 (torch.Tensor): Initial time
            t (torch.Tensor): Final time
            inner_path (str, optional): Path type ('spline' or 'line'). Defaults to "spline".
            N (int, optional): Number of outer integration points. Defaults to 2.
            inner_N (int, optional): Number of inner integration points. Defaults to 2.
            c_spline (callable, optional): Spline function. Defaults to None.
            c_prime_spline (callable, optional): Spline derivative function. Defaults to None.
            
        Returns:
            torch.Tensor: The G integral value
        """
        if c_spline is None:
            c_spline = self.c_spline
        if c_prime_spline is None:
            c_prime_spline = self.c_prime_spline
        
        if self.x_t_0 is None:
            self.x_t_0 = c_spline(t_0)

        if self.int_method == "odeint":
            x_t = c_spline(t)
            odeint_result = self.int_func(lambda _t, x: self.H(_t, t, x_t=x_t, inner_path=inner_path, N=inner_N, c_spline=c_spline, c_prime_spline=c_prime_spline), torch.zeros(x_t.shape).type(x_t.dtype).to(x_t.device), torch.linspace(t_0, t, N).type(x_t.dtype).to(x_t.device), method='rk4')
            return odeint_result[-1]
        else:
            int_func_kwargs = {"N": N}
            integration_domain = torch.stack((t_0, t)).unsqueeze(0)
            grid_points, hs, n_per_dim = self.int_func.get_grid_points(dim=1, integration_domain=integration_domain, **int_func_kwargs)
            grid_points = grid_points.type(self.x_t_0.dtype).to(self.x_t_0.device)
            if inner_path == "spline":
                # RETURN ALL METHOD
                fn_result = self.H(t_0, t, inner_path=inner_path, N=N, return_all=True, c_spline=c_spline, c_prime_spline=c_prime_spline) # return_all is already cumsummed
                if len(fn_result.shape) == 3:
                    fn_result = fn_result.permute(2, 0, 1)
                if len(fn_result.shape) == 4:
                    fn_result = fn_result.permute(3, 0, 1, 2)
                fn_result = torch.cat((fn_result[[-1]], fn_result[-1] - fn_result), dim=0) # undo the cumsum
            elif inner_path == "line":
                # LOOP OVER POINTS METHOD
                fn_result = torch.stack([self.H(gp[0], t, N=inner_N, inner_path="line", c_spline=c_spline, c_prime_spline=c_prime_spline) for gp in grid_points[:-1]])
                # append a zero
                if self.int_method != 'odeint':
                    fn_result = torch.cat((fn_result, torch.zeros(1, *fn_result.shape[1:]).to(fn_result.device)), dim=0)
            else:
                raise ValueError("Invalid inner_path for G: {}, must be 'spline' or 'line'".format(inner_path))

        if self.int_method != 'odeint':
            quad_ret = self.int_func.integrate_from_pts(fn_result, dim=1, hs=hs, n_per_dim=n_per_dim, integration_domain=integration_domain)
        else:
            quad_ret = fn_result

        return quad_ret

    def update_traj(self, t, x, inplace=True):
        """Update the trajectory with a new point.
        
        Args:
            t (torch.Tensor): Time point to add
            x (torch.Tensor): State to add
            inplace (bool, optional): Whether to update in place. Defaults to True.
            
        Returns:
            tuple or None: If not inplace, returns (time_vals, traj, c_spline, c_prime_spline)
        """
        if t > self.time_vals[-1]:
            time_vals = torch.cat((self.time_vals.clone(), t.unsqueeze(0)))
            traj = torch.cat((self.traj.clone(), x.unsqueeze(-2)), dim=-2)

            c_spline, c_prime_spline = generate_pw_spline(traj, time_vals=time_vals)

            if inplace:
                self.time_vals = time_vals
                self.traj = traj
                self.c_spline, self.c_prime_spline = c_spline, c_prime_spline
            else:
                return time_vals, traj, c_spline, c_prime_spline
        else:
            return self.time_vals, self.traj, self.c_spline, self.c_prime_spline
        

    def f(self, t, x, t_0=0, inner_path="spline", N=2, inner_N=None):
        """Compute the vector field f(t,x) for the ODE.
        
        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state
            t_0 (torch.Tensor, optional): Initial time. Defaults to 0.
            inner_path (str, optional): Path type ('spline' or 'line'). Defaults to "spline".
            N (int, optional): Number of integration points. Defaults to 2.
            inner_N (int, optional): Number of inner integration points. Defaults to None.
            
        Returns:
            torch.Tensor: The vector field value f(t,x)
        """
        if isinstance(t_0, int):
            t_0 = torch.tensor(t_0, dtype=self.traj.dtype, device=self.traj.device)
        if isinstance(t, int):
            t = torch.tensor(t, dtype=self.traj.dtype, device=self.traj.device)
        # ASSUMPTION: t0 and t are included in the "traj" tensor

        time_vals, traj, c_spline, c_prime_spline = self.update_traj(t, x, inplace=False)

        if inner_N is None:
            inner_N = N

        self.x_t_0 = c_spline(t_0)
        self.x_t = c_spline(t)

        G_t0_t = self.G(t_0, t, inner_path=inner_path, N=N, inner_N=inner_N, c_spline=c_spline, c_prime_spline=c_prime_spline)

        f_est = (G_t0_t + self.x_t - self.x_t_0)/(t - t_0)
        return f_est
    
    def get_deriv_func(self, t_0=None, inner_path="line", interp_pts=2, inner_N=2, fast_mode=True, fast_mode_base_ind=None, scale_interp_pts=True, deriv_func=None, reverse=False):
        """Get the derivative function for the ODE.
        
        Creates a function that computes the vector field for the ODE, with options
        for different integration methods and optimization strategies.
        
        Args:
            t_0 (torch.Tensor, optional): Initial time. Defaults to None.
            inner_path (str, optional): Path type ('spline' or 'line'). Defaults to "line".
            interp_pts (int, optional): Number of interpolation points. Defaults to 2.
            inner_N (int, optional): Number of inner integration points. Defaults to 2.
            fast_mode (bool, optional): Whether to use fast mode. Defaults to True.
            fast_mode_base_ind (int, optional): Base index for fast mode. Defaults to None. If None, use the last point in the trajectory.
            scale_interp_pts (bool, optional): Whether to scale interpolation points. Defaults to True.
            deriv_func (callable, optional): Custom derivative function. Defaults to None.
            reverse (bool, optional): Whether to reverse the dynamics. Defaults to False.
            
        Returns:
            ODEFunc: Function that computes the vector field
        """
        traj = self.traj

        # if t_0 is None, use the first point in the trajectory (typically 0)
        if t_0 is None:
            t_0 = self.time_vals[0]

        if not isinstance(t_0, torch.Tensor):
            t_0 = torch.tensor(t_0, dtype=traj.dtype, device=traj.device)

        # if fast_mode_base_ind is None, use the last point in the trajectory
        if fast_mode_base_ind is None:
            fast_mode_base_ind = self.traj.shape[-2] - 1
        
        # scale_interp_pts equivalent for fast_mode_base_ind=1
        if scale_interp_pts:
            N = int(fast_mode_base_ind*interp_pts + 2)
        else:
            N = int(interp_pts + 2)

        if not fast_mode:
            f = lambda _t, _x:  self.f(_t, _x, t_0, inner_path=inner_path, N=N, inner_N=inner_N)
        else:
            # TODO: change to t_0 + fast_mode_base_ind*self.dt
            t_b = torch.tensor(fast_mode_base_ind*self.dt, dtype=traj.dtype, device=traj.device)
            x_t_b = traj[..., fast_mode_base_ind, :]
            if deriv_func is None:
                time_vals_fast = (torch.arange(fast_mode_base_ind + 1)*self.dt).type(traj.dtype).to(traj.device)
                jacobian_ode_fast = JacobianODE(traj[..., :fast_mode_base_ind + 1, :], self.jac_func, time_vals_fast, int_method=self.int_method, sigma=self.sigma)
                f_t_b = jacobian_ode_fast.f(t_b, x_t_b, t_0, inner_path='spline', N=N, inner_N=None)
            else:
                if reverse:
                    pass
                    # f_t_b = -deriv_func(t_b, x_t_b)
                else:
                    f_t_b = deriv_func(t_b, x_t_b)

            def f_unbatched(_t, _x):
                if _t == t_b:
                    return f_t_b
                # if _t > t_b, we're moving forward in time, so integrate from t_b to _t
                elif _t > t_b:
                    if inner_path in ["spline", "line"]:
                        return self.H(t_b, _t, x_t_b, _x, inner_path=inner_path, N=inner_N) + f_t_b
                    else:
                        raise ValueError("Invalid inner_path: {}, must be 'spline' or 'line'".format(inner_path))
                else: # _t < t_b
                    # if _t < t_b, we're moving backward in time, so integrate from _t to t_b
                    if inner_path in ["spline", "line"]:
                        return f_t_b - self.H(_t, t_b, _x, x_t_b, inner_path=inner_path, N=inner_N)
                    else:
                        raise ValueError("Invalid inner_path: {}, must be 'spline' or 'line'".format(inner_path))
            def f_raw(_t, _x):
                if _t.dim() == 0:
                    return f_unbatched(_t, _x)
                else:
                    return torch.stack([f_unbatched(_t[i], _x[..., i, :]) for i in range(_t.size(0))], dim=-2)

            f = ODEFunc(f_raw)
        
        return f

class ODEFunc(torch.nn.Module):
    """Wrapper class for ODE vector field functions.
    
    This class wraps a vector field function to make it compatible with torchdiffeq.
    
    Attributes:
        f (callable): The vector field function
    """
    
    def __init__(self, f):
        """Initialize the ODEFunc.
        
        Args:
            f (callable): Vector field function
        """
        super().__init__()
        self.f = f
    
    def forward(self, t, x):
        """Compute the vector field at time t and state x.
        
        Args:
            t (torch.Tensor): Time point
            x (torch.Tensor): State
            
        Returns:
            torch.Tensor: Vector field value
        """
        return self.f(t, x)

class JacobianODEint:
    """Class for integrating Jacobian ODEs with various options.
    
    This class provides methods for simulating trajectories using Jacobian ODEs,
    with support for teacher forcing and various integration methods.
    
    Attributes:
        jac_func (callable): Function that computes the Jacobian
        dt (float): Time step size
        line_endpoints (dict): Cache for line endpoints
    """
    
    def __init__(self, jac_func, dt):
        """Initialize the JacobianODEint.
        
        Args:
            jac_func (callable): Function that computes the Jacobian
            dt (float): Time step size
        """
        self.jac_func = jac_func
        self.dt = dt

        self.line_endpoints = {}
    
    def generate_dynamics(
            self, 
            traj,
            traj_init_steps=2,
            steps_per_dt=1, 
            interp_pts=2, 
            scale_interp_pts=True,
            time_shift=False,
            alpha_teacher_forcing=0, 
            teacher_forcing_steps=1, 
            odeint_kwargs={"method": "rk4"}, 
            jacobianODE_kwargs={},
            inner_path="line",
            inner_N=None,
            reverse=False,
            fast_mode=True,
            fast_mode_base_ind=None,
            deriv_func=None,
            verbose=False
        ):
        """Generate dynamics by integrating the Jacobian ODE.
        
        Simulates the system dynamics using the Jacobian ODE, with options for
        teacher forcing and various integration methods.
        
        Args:
            traj (torch.Tensor): Initial trajectory
            traj_init_steps (int, optional): Number of initial steps to use. Defaults to 2.
            steps_per_dt (int, optional): Number of integration steps per dt. Defaults to 1.
            interp_pts (int, optional): Number of interpolation points. Defaults to 2.
            scale_interp_pts (bool, optional): Whether to scale interpolation points. Defaults to True.
            time_shift (bool, optional): Whether to shift time window. Defaults to False.
            alpha_teacher_forcing (float, optional): Teacher forcing strength. Defaults to 0.
            teacher_forcing_steps (int, optional): Steps between teacher forcing. Defaults to 1.
            odeint_kwargs (dict, optional): Arguments for odeint. Defaults to {"method": "rk4"}.
            jacobianODE_kwargs (dict, optional): Arguments for JacobianODE. Defaults to {}.
            inner_path (str, optional): Path type ('spline' or 'line'). Defaults to "line".
            inner_N (int, optional): Number of inner integration points. Defaults to None.
            reverse (bool, optional): Whether to reverse the dynamics. Defaults to False.
            fast_mode (bool, optional): Whether to use fast mode. Defaults to True.
            fast_mode_base_ind (int, optional): Base index for fast mode. Defaults to None.
            deriv_func (callable, optional): Custom derivative function. Defaults to None.
            verbose (bool, optional): Whether to show progress bar. Defaults to False.
            
        Returns:
            torch.Tensor: Simulated trajectory
            
        Raises:
            ValueError: If traj_init_steps is less than 2
        """
        if traj_init_steps < 2:
            raise ValueError("traj_init_steps must be at least 2")
        if reverse:
            traj = torch.flip(traj, dims=(-2,))
        traj_init = traj[..., :traj_init_steps, :]
        x_t = traj_init[..., -1, :] # initial 
        

        time_vals = (torch.arange(traj_init.shape[-2])*self.dt).type(traj_init.dtype).to(traj_init.device)
        t_0 = time_vals[0]
        t = time_vals[-1]

        if not reverse:
            jac_func = self.jac_func
        else:
            jac_func = lambda _x, _t: -self.jac_func(_x, _t)
        jacobian_ode = JacobianODE(traj_init, jac_func, time_vals, **jacobianODE_kwargs)

        f = jacobian_ode.get_deriv_func(t_0, inner_path=inner_path, interp_pts=interp_pts, inner_N=inner_N, fast_mode=fast_mode, fast_mode_base_ind=fast_mode_base_ind, scale_interp_pts=scale_interp_pts, deriv_func=deriv_func, reverse=reverse)

        # set up simulation parameters
        dt_sim = self.dt/steps_per_dt
        t_sim = (traj.shape[-2] - 1)*self.dt - t
        n_steps = int(np.round(t_sim.cpu().numpy()/dt_sim))
        x_out = traj_init

        step_counter = 0
        for i_sim in tqdm(range(n_steps), disable=not verbose):
            step_counter += 1
            # n_true_pts = 2 + i_sim
            n_true_pts = x_out.shape[-2]
            # put interp_pts between each true point
            if not fast_mode:
                # N (outer N) doesn't matter in fast mode
                if scale_interp_pts:
                    N = (n_true_pts - 1)*interp_pts + n_true_pts
                else:
                    N = interp_pts + 2

            if time_shift:
                t_0 = torch.max(torch.stack((time_vals[0], t - int(N/2)*self.dt)), dim=0)[0]

            x_t_new = odeint(
                            f, 
                            x_t,
                            torch.cat((t.unsqueeze(0), (t + dt_sim).unsqueeze(0))),
                            **odeint_kwargs
                        )[-1]
            if step_counter % steps_per_dt == 0:
                x_out = torch.cat((x_out, x_t_new.unsqueeze(-2)), dim=-2)
                if alpha_teacher_forcing > 0 and step_counter/steps_per_dt % teacher_forcing_steps == 0:
                    x_t_new = (1 - alpha_teacher_forcing)*x_t_new + alpha_teacher_forcing*traj[..., int(np.round((t + dt_sim).cpu().numpy()/self.dt)), :]
                    # WARNING: will not work for steps_per_dt > 1
            # we only need to update the spline if we're not using fast mode or if the inner path is a spline
            if not fast_mode or inner_path == "spline":
                jacobian_ode.update_traj(t + dt_sim, x_t_new, inplace=True)
            t = t + dt_sim
            x_t = x_t_new

        output_traj = x_out

        if reverse:
            output_traj = torch.flip(output_traj, dims=(-2,))

        return output_traj
        
