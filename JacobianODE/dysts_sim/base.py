"""
Dynamical systems in Python

(M, T, D) or (T, D) convention for outputs

Requirements:
+ numpy
+ scipy
+ sdeint (for integration with noise)
+ numba (optional, for faster integration)

Adapted from https://github.com/williamgilpin/dysts
Huge shoutout to William Gilpin for a fantastic repo, check out his work!
"""


from dataclasses import dataclass, field, asdict
import warnings
import json
import collections
import os

from tqdm.auto import tqdm
import gzip

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/dynamical_systems.json")

import numpy as np

from .utils import integrate_dyn, standardize_ts

try:
    from numba import jit, njit

    #     from jax import jit
    #     njit = jit

    has_jit = True
except ModuleNotFoundError:
    import numpy as np

    has_jit = False
    # Define placeholder functions
    def jit(func):
        return func

    njit = jit

staticjit = lambda func: staticmethod(
    njit(func)
)  # Compose staticmethod and jit decorators


data_default = {'bifurcation_parameter': None,
                'citation': None,
                 'correlation_dimension': None,
                 'delay': False,
                 'description': None,
                 'dt': 0.001,
                 'embedding_dimension': 3,
                 'hamiltonian': False,
                 'initial_conditions': [0.1, 0.1, 0.1],
                 'kaplan_yorke_dimension': None,
                 'lyapunov_spectrum_estimated': None,
                 'maximum_lyapunov_estimated': None,
                 'multiscale_entropy': None,
                 'nonautonomous': False,
                 'parameters': {},
                 'period': 10,
                 'pesin_entropy': None,
                 'unbounded_indices': [],
                 'positive_only': False,
                 'vectorize': False
               }
    

@dataclass(init=False)
class BaseDyn:
    """A base class for dynamical systems
    
    Attributes:
        name (str): The name of the system
        params (dict): The parameters of the system.
        random_state (int): The seed for the random number generator. Defaults to None
        
    Development:
        Add a function to look up additional metadata, if requested
    """

    name: str = None
    params: dict = field(default_factory=dict)
    random_state: int = None
    dt: float = None

    def __init__(self, **entries):
        self.name = self.__class__.__name__
        self.loaded_data = self._load_data()
        self.params = self.loaded_data["parameters"]
        self.params = {key: entries[key] if key in entries else self.params[key] for key in self.params}
        if 'random_state' in entries:
            self.random_state = entries['random_state']

        # Cast all parameter arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)

        ic_val = self.loaded_data["initial_conditions"]
        if not np.isscalar(ic_val):
            ic_val = np.array(ic_val)
        self.ic = ic_val
        np.random.seed(self.random_state)

        for key in self.loaded_data.keys():
            setattr(self, key, self.loaded_data[key])
        
        if 'dt' in entries:
            self.dt = entries['dt']

    def redo_init(self, loaded_data):
        self.loaded_data = loaded_data
        self.params = self.loaded_data["parameters"]

        # Cast all parameter arrays to numpy
        for key in self.params:
            if not np.isscalar(self.params[key]):
                self.params[key] = np.array(self.params[key])
        self.__dict__.update(self.params)

        ic_val = self.loaded_data["initial_conditions"]
        if not np.isscalar(ic_val):
            ic_val = np.array(ic_val)
        self.ic = ic_val
        np.random.seed(self.random_state)

        for key in self.loaded_data.keys():
            setattr(self, key, self.loaded_data[key])
    
    def update_params(self):
        """
        Update all instance attributes to match the values stored in the 
        `params` field
        """
        for key in self.params.keys():
            setattr(self, key, self.params[key])
    
    def get_param_names(self):
        return sorted(self.params.keys())

    def _load_data(self):
        """Load data from a JSON file"""
        # with open(os.path.join(curr_path, "chaotic_attractors.json"), "r") as read_file:
        #     data = json.load(read_file)
        with open(self.data_path, "r") as read_file:
            data = json.load(read_file)
        try:
            return data[self.name]
        except KeyError:
            print(f"No metadata available for {self.name}")
            #return {"parameters": None}
            return data_default

    @staticmethod
    def _rhs(X, t):
        """The right-hand side of the dynamical system"""
        return X

    @staticmethod
    def _jac(X, t):
        """The Jacobian of the dynamical system"""
        return X

    @staticmethod
    def bound_trajectory(traj):
        """Bound a trajectory within a periodic domain"""
        return np.mod(traj, 2 * np.pi)
    

    # def load_trajectory(
    #     self,
    #     subsets="train", 
    #     granularity="fine", 
    #     return_times=False,
    #     standardize=False,
    #     noise=False
    # ):
    #     """
    #     Load a precomputed trajectory for the dynamical system
        
    #     Args:
    #         subsets ("train" |  "test"): Which dataset (initial conditions) to load
    #         granularity ("course" | "fine"): Whether to load fine or coarsely-spaced samples
    #         noise (bool): Whether to include stochastic forcing
    #         standardize (bool): Standardize the output time series.
    #         return_times (bool): Whether to return the timepoints at which the solution 
    #             was computed
                
    #     Returns:
    #         sol (ndarray): A T x D trajectory
    #         tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory
        
    #     """
    #     period = 12
    #     granval = {"coarse": 15, "fine": 100}[granularity]
    #     dataset_name = subsets.split("_")[0]
    #     data_path = f"{dataset_name}_multivariate__pts_per_period_{granval}__periods_{period}.json.gz"
    #     if noise:
    #         name_parts = list(os.path.splitext(data_path))
    #         data_path = "".join(name_parts[:-1] + ["_noise"] + [name_parts[-1]])


    #     if not _has_data:
    #         warnings.warn(
    #                     "Data module not found. To use precomputed datasets, "+ \
    #                         "please install the external data repository "+ \
    #                             "\npip install git+https://github.com/williamgilpin/dysts_data"
    #         )

    #     base_path = get_datapath()
    #     data_path = os.path.join(base_path, data_path)

    #     # cwd = os.path.dirname(os.path.realpath(__file__))
    #     # data_path = os.path.join(cwd, "data", data_path)

    #     with gzip.open(data_path, 'rt', encoding="utf-8") as file:
    #         dataset = json.load(file)
            
    #     tpts, sol = np.array(dataset[self.name]['time']), np.array(dataset[self.name]['values'])
        
    #     if standardize:
    #         sol = standardize_ts(sol)

    #     if return_times:
    #         return tpts, sol
    #     else:
    #         return sol

    # def make_trajectory(self, *args, **kwargs):
    #     """Make a trajectory for the dynamical system"""
    #     raise NotImplementedError

    # def sample(self, *args,  **kwargs):
    #     """Sample a trajectory for the dynamical system via numerical integration"""
    #     return self.make_trajectory(*args, **kwargs)


class DynSys(BaseDyn):
    """
    A continuous dynamical system base class, which loads and assigns parameter
    values from a file

    Attributes:
        kwargs (dict): A dictionary of keyword arguments passed to the base dynamical
            model class
    """

    def __init__(self, **kwargs):
        self.data_path = data_path
        super().__init__(**kwargs)
        self.dt = self.loaded_data["dt"] if self.dt is None else self.dt
        self.period = self.loaded_data["period"]
        self.mean_val = None
        self.std_val = None

    # AE writing this
    def rhs(self, X, t, length=None):
        """The right hand side of a dynamical equation"""
        # param_list = [
        #     self.params[key] for key in self.params
        # ]
        if len(X.shape) == 1:
            if self.vectorize:
                out = self._rhs(X.T, t, **self.params)
            else:
                out = self._rhs(*X.T, t, **self.params)
        elif len(X.shape) == 2: # B x D
            out = np.zeros((X.shape[0], X.shape[1]))
            for i in range(X.shape[0]):
                out[i] = self.rhs(X[i], t)
        elif len(X.shape) == 3: # B x T x D
            if length is None:
                length = X.shape[1]
            out = np.zeros((X.shape[0], length, X.shape[2]))
            for i in range(X.shape[0]):
                for j, _t in enumerate(t):
                    out[i, j] = self.rhs(X[i, j], _t)
        else:
            raise NotImplementedError("Shapes other than (D,), (B, D) or (B, T, D) not supported")
        return out
    
    # AE writing this
    def jac(self, X, t, length=None):
        """The Jacobian of the dynamical equation"""
        # param_list =[
        #     getattr(self, param_name) for param_name in self.get_param_names()
        # ]
        # param_dict = {
        #     param_name: getattr(self, param_name) for param_name in self.get_param_names()
        # }
        if len(X.shape) == 1:
            if self.vectorize:
                out = np.array(self._jac(X, t, **self.params))
                # out = np.array(self._jac(X, t, **param_dict))
            else:
                out = np.array(self._jac(*X, t, **self.params))
                # out = np.array(self._jac(*X, t, **param_dict))
        elif len(X.shape) == 2:
            out = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                out[i] = self.jac(X[i], t)
        elif len(X.shape) == 3:
            if length is None:
                length = X.shape[1]
            out = np.zeros((X.shape[0], length, X.shape[2], X.shape[2]))
            for i in range(X.shape[0]):
                for j, _t in enumerate(t):
                    out[i, j] = self.jac(X[i, j], _t)
        elif len(X.shape) == 4:
            if length is None:
                length = X.shape[-2]
            out = np.zeros((X.shape[0], X.shape[1], length, X.shape[-1], X.shape[-1]))
            for i1 in range(X.shape[0]):
                for i2 in range(X.shape[1]):
                    for j, _t in enumerate(t):
                        out[i1, i2, j] = self.jac(X[i1, i2, j], _t)
        else:
            print(X.shape)
            raise NotImplementedError("Shapes other than (D,), (B, D), (B, T, D), or (B1, B2, T, D) not supported")
        return out

    def __call__(self, X, t):
        """Wrapper around right hand side"""
        return self.rhs(X, t)
    
    def make_trajectory(
        self,
        n_periods,
        method="Radau",
        resample=True,
        pts_per_period=100,
        return_times=False,
        standardize=False,
        # postprocess=True,
        noise=0.0,
        num_ics=1,
        new_ic_mode='reference',
        traj_offset_sd=1,
        verbose=False, 
    ):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and initial conditions
        
        Args:
            n (int): the total number of trajectory points
            method (str): the integration method
            resample (bool): whether to resample trajectories to have matching dominant 
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
            postprocess (bool): Whether to apply coordinate conversions and other domain-specific 
                rescalings to the integration coordinates
            noise (float): The amount of stochasticity in the integrated dynamics. This would correspond
                to Brownian motion in the absence of any forcing.
        
        Returns:
            sol (ndarray): A T x D trajectory
            tpts, sol (ndarray): T x 1 timepoint array, and T x D trajectory
            
        """
        # n = n_periods * pts_per_period
        n = int(n_periods * (self.period / self.dt))
        tpts = np.arange(n) * self.dt
        np.random.seed(self.random_state)

        if resample:
            # tlim = (self.period) * (n / pts_per_period)
            n = n_periods * pts_per_period
            tlim = self.period*n_periods
            # upscale_factor = (tlim / self.dt) / n
            upscale_factor = (self.period) / (pts_per_period * self.dt)
            # if upscale_factor > 1e3:
            if upscale_factor > 1e4:
                # warnings.warn(
                #     f"Expect slowdown due to excessive integration required; scale factor {upscale_factor}"
                # )
                warnings.warn(
                    f"New simulation timescale is more than 10000 times the original dt; scale factor {upscale_factor}"
                )
            tpts = np.linspace(0, tlim, n)
            
        ics = self.ic.copy()
        if num_ics > 1:
            if len(np.array(ics).shape) == 1:
                ics = np.vstack([ics, np.zeros((num_ics - 1, len(ics)))])
            else:
                if ics.shape[0] == num_ics:
                    pass
                elif ics.shape[0] < num_ics:
                    ics = np.vstack([ics, np.zeros((num_ics - ics.shape[0], len(ics)))])
                    
                else: # self.ic.shape[0] > num_ics
                    ics = ics[:num_ics, :]
            # for i in range(1, num_ics)

        m = len(np.array(ics).shape)
        if m < 1:
            m = 1
        if m == 1:

            sol = np.expand_dims(integrate_dyn(
                self, ics, tpts, dtval=self.dt, method=method, noise=noise
            ).T, 0)
        else:
            sol = list()
            for i, ic in tqdm(enumerate(ics), disable=not verbose, total=len(ics)):
                traj = integrate_dyn(
                    self, ic, tpts, dtval=self.dt, method=method, noise=noise
                )
                check_complete = (traj.shape[-1] == len(tpts))
                # print(f"Simulation {i} trajectory length: {traj.shape[-1]}, len(tpts): {len(tpts)}")
                if check_complete: 
                    sol.append(traj)
                else:
                    warnings.warn(f"Integration did not complete for initial condition {ic}, skipping this point")
                    pass
                
                # select the next initial condition so it's on the attractor
                low = int(len(tpts)/4)
                high = len(tpts)
                if i < len(ics) - 1:
                    if self.name == 'LorotkaVoltarneodo':
                        selected_ind = 0
                    else:
                        selected_ind = np.random.randint(low, high)
                    # sol is returned as indices x time
                    if new_ic_mode == 'reference':
                        ics[i + 1] = sol[-1][:, selected_ind] + np.random.randn(sol[-1].shape[0])*traj_offset_sd
                    else: # new_ic_mode == 'random'
                        ics[i + 1] = np.random.randn(sol[-1].shape[0])*traj_offset_sd
                    if self.positive_only:
                        ics[i + 1] = np.abs(ics[i + 1])
                    if self.name == 'LorotkaVoltarneodo':
                        ics[i + 1, -2:] = np.abs(ics[i + 1, -2:])
                    # if np.abs(self.ic[i + 1]).max() > 8:
                    #     print(self.ic[i + 1])
                    
            sol = np.transpose(np.array(sol), (0, 2, 1))

        # if hasattr(self, "_postprocessing") and postprocess:
        #     warnings.warn(
        #         "This system has at least one unbounded variable, which has been mapped to a bounded domain. Pass argument postprocess=False in order to generate trajectories from the raw system."
        #     )
        #     sol2 = np.moveaxis(sol, (-1, 0), (0, -1))
        #     sol = np.squeeze(
        #         np.moveaxis(np.dstack(self._postprocessing(*sol2)), (0, 1), (1, 0))
        #     )

        if standardize:
            sol, self.mean_val, self.scale_val = standardize_ts(sol)

        if return_times:
            return {'dt': tpts[1] - tpts[0], 'time': tpts, 'values': sol}
        else:
            return sol



class DynSysDelay(DynSys):
    """
    A delayed differential equation object. Defaults to using Euler integration scheme
    The delay timescale is assumed to be the "tau" field. The embedding dimension is set 
    by default to ten, but delay equations are infinite dimensional.
    Uses a double-ended queue for memory efficiency

    Attributes:
        kwargs (dict): A dictionary of keyword arguments passed to the dynamical
            system parent class
    
    Todo:
        Treat previous delay values as a part of the dynamical variable in rhs
    
        Currently, only univariate delay equations and single initial conditons 
        are supported
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.history = collections.deque(1.3 * np.random.rand(1 + mem_stride))
        self.__call__ = self.rhs

    def rhs(self, X, t):
        """The right hand side of a dynamical equation"""
        X, Xprev = X[0], X[1]
        param_list = [
            getattr(self, param_name) for param_name in self.get_param_names()
        ]
        out = self._rhs(X, Xprev, t, *param_list)
        return out

    def make_trajectory(
        self,
        n,
        d=10,
        method="Euler",
        noise=0.0,
        resample=False,
        pts_per_period=100,
        standardize=False,
        return_times=False,
        postprocess=True,
    ):
        """
        Generate a fixed-length trajectory with default timestep, parameters, and 
        initial conditions.
        
        Args:
            n (int): the total number of trajectory points
            d (int): the number of embedding dimensions to return
            method (str): Not used. Currently Euler is the only option here
            noise (float): The amplitude of brownian forcing
            resample (bool): whether to resample trajectories to have matching dominant 
                Fourier components
            pts_per_period (int): if resampling, the number of points per period
            standardize (bool): Standardize the output time series.
            return_times (bool): Whether to return the timepoints at which the solution 
                was computed
            
        Todo:
            Support for multivariate and multidelay equations with multiple deques
            Support for multiple initial conditions
            
        """
        np.random.seed(self.random_state)
        n0 = n

        ## history length proportional to the delay time over the timestep
        mem_stride = int(np.ceil(self.tau / self.dt))
        
        ## If resampling is performed, calculate the true number of timesteps for the
        ## Euler loop
        if resample:
            num_periods = n / pts_per_period
            num_timesteps_per_period = self.period / self.dt
            nt = int(np.ceil(num_timesteps_per_period *  num_periods))
        else:
            nt = n

        # remove transient at front and back
        clipping = int(np.ceil(mem_stride / (nt / n)))

        ## Augment the number of timesteps to account for the transient and the embedding
        ## dimension
        n += (d + 1) * clipping
        nt += (d + 1) * mem_stride

        ## If passed initial conditions are sufficient, then use them. Otherwise, 
        ## pad with with random initial conditions
        values = self.ic[0] * (1 + 0.2 * np.random.rand(1 + mem_stride))
        values[-len(self.ic[-mem_stride:]):] = self.ic[-mem_stride:]
        history = collections.deque(values)

        ## pre-allocate full solution
        tpts = np.arange(nt) * self.dt
        sol = np.zeros(n)
        sol[0] = self.ic[-1]
        x_next = sol[0]

        ## Define solution submesh for resampling
        save_inds = np.linspace(0, nt, n).astype(int)
        save_tpts = list()

        ## Pre-compute noise
        noise_vals = noise * np.random.normal(size=nt, loc=0.0, scale=np.sqrt(self.dt))

        ## Run Euler integration loop
        for i, t in enumerate(tpts):
            if i == 0:
                continue

            x_next = (
                x_next
                + self.rhs([x_next, history.popleft()], t) * self.dt
                + noise_vals[i]
            )

            if i in save_inds:
                sol[save_inds == i] = x_next
                save_tpts.append(t)
            history.append(x_next)

        save_tpts = np.array(save_tpts)
        save_dt = np.median(np.diff(save_tpts))

        ## now stack strided solution to create an embedding
        sol_embed = list()
        embed_stride = int((n / nt) * mem_stride)
        for i in range(d):
            sol_embed.append(sol[i * embed_stride : -(d - i) * embed_stride])
        sol0 = np.vstack(sol_embed)[:, clipping : (n0 + clipping)].T

        if hasattr(self, "_postprocessing") and postprocess:
            warnings.warn(
                "This system has at least one unbounded variable, which has been mapped to a bounded domain. Pass argument postprocess=False in order to generate trajectories from the raw system."
            )
            sol2 = np.moveaxis(sol0, (-1, 0), (0, -1))
            sol0 = np.squeeze(
                np.moveaxis(np.dstack(self._postprocessing(*sol2)), (0, 1), (1, 0))
            )

        if standardize:
            sol0 = standardize_ts(sol0)

        if return_times:
            return {'dt': save_dt, 'time':np.arange(sol0.shape[0]) * save_dt, 'values': sol0}
        else:
            return sol0

# def get_attractor_list(model_type="continuous"):
#     """
#     Returns the names of all models in the package
    
#     Args:
#         model_type (str): "continuous" (default) or "discrete"
        
#     Returns:
#         attractor_list (list of str): The names of all attractors in database
#     """
#     if model_type == "continuous":
#         data_path = data_path_continuous
#     else:
#         data_path = data_path_discrete
#     with open(data_path, "r") as file:
#         data = json.load(file)
#     attractor_list = sorted(list(data.keys()))
#     return attractor_list


# def make_trajectory_ensemble(n, subset=None, use_multiprocessing=False, random_state=None, verbose=False, **kwargs):
#     """
#     Integrate multiple dynamical systems with identical settings
    
#     Args:
#         n (int): The number of timepoints to integrate
#         subset (list): A list of system names. Defaults to all systems
#         use_multiprocessing (bool): Not yet implemented.
#         random_state (int): The random seed to use for the ensemble
#         kwargs (dict): Integration options passed to each system's make_trajectory() method
    
#     Returns:
#         all_sols (dict): A dictionary containing trajectories for each system
    
#     """
#     if not subset:
#         subset = get_attractor_list()

#     if use_multiprocessing:
#         warnings.warn(
#             "Multiprocessing not implemented."
#         )
    
#     # We run this inside the function scope to avoid a circular import issue
#     flows = importlib.import_module("dysts.flows", package=".flows")
    
#     all_sols = dict()
#     if verbose:
#         print(f"Subsets: {subset}")
#         print("-"*20)
#     for equation_name in subset:
#         print("Generating trajectory for ", equation_name)
#         eq = getattr(flows, equation_name)()
#         eq.random_state = random_state
#         sol = eq.make_trajectory(n, verbose=verbose, **kwargs)
#         all_sols[equation_name] = sol

#     return all_sols
