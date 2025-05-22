import numpy as np
import torch
import torchcubicspline

def floor(x, eps=1e-5):
    if isinstance(x, torch.Tensor):
        return torch.floor(x + eps).type(torch.int64)
    else:
        return np.floor(x + eps).astype(int)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

def generate_spline(x, t_i=None, t_f=None, dt=None, time_vals=None):
    if time_vals is None:
        if t_i is None or t_f is None or dt is None:
            raise ValueError("t_i, t_f, and dt must be provided if time_vals is not provided")
        time_vals = torch.arange(t_i, t_f + dt/2, dt).type(x.dtype).to(x.device)
        pts = x[..., floor(t_i/dt):floor(t_f/dt) + 1, :]
    else:
        # IF TIME VALUES ARE PROVIDED, ASSUME ENTIRE TRAJECTORY WILL BE USED FOR SPLINE
        pts = x
    coeffs = torchcubicspline.natural_cubic_spline_coeffs(time_vals, pts)
    spline = torchcubicspline.NaturalCubicSpline(coeffs)
    c = lambda t: spline.evaluate(t.squeeze(-1).type(x.dtype).to(x.device))
    c_prime = lambda t: spline.derivative(t.squeeze(-1).type(x.dtype).to(x.device))
    
    return c, c_prime

# line definition
# c(t) = (1 - (t - t_i)/(t_f - t_i)) x0 + ((t - t_i)/(t_f - t_i)) x1
# for t in [t_i, t_f]
# c'(t) = (-x0 (t - t_i)/(t_f - t_i) + x1 (t - t_i)/(t_f - t_i))'
# = (-x0 (t)/(t_f - t_i) + x0 (t_i)/(t_f - t_i) + x1 (t)/(t_f - t_i) - x1 (t_i)/(t_f - t_i))'
# = -x0 (1)/(t_f - t_i) + 0 + x1 (1)/(t_f - t_i) - 0
# = (x1 - x0)/(t_f - t_i)

def c_line(t, x0, x, t_i, t_f):
    t = ((t - t_i)/(t_f - t_i))
    if len(x0.shape) == 2: # batches x dims
        t = t.unsqueeze(0)
        x = x.unsqueeze(-2)
        x0 = x0.unsqueeze(-2)
    elif len(x0.shape) == 3: # batches x n_trajs x dims
        t = t.unsqueeze(0)
        x = x.unsqueeze(-2)
        x0 = x0.unsqueeze(-2)
    return (1 - t)*x0 + t*x

def c_prime_line(t, x0, x, t_i, t_f):
    deriv = (x - x0)/(t_f - t_i)
    if isinstance(t, torch.Tensor) and len(t.shape) == 2:
        if len(deriv.shape) == 2: # batches x dims
            return deriv.unsqueeze(-2).repeat(1, t.shape[0], 1)
        elif len(deriv.shape) == 3: # batches x n_trajs x dims
            return deriv.unsqueeze(-2).repeat(1, 1, t.shape[0], 1)
        else: # dims
            return deriv.unsqueeze(0).repeat(t.shape[0], 1)
    return deriv

# def generate_line(x, t_i, t_f, dt):
#     c = lambda t: c_line(t.type(x.dtype).to(x.device), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :])
#     c_prime = lambda t: c_prime_line(t.type(x.dtype).to(x.device), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :])

#     return c, c_prime

def generate_line(x, t_i, t_f, dt, normalize_times=True):
    if normalize_times:
        c = lambda t: c_line(t.type(x.dtype).to(x.device), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :], t_i, t_f)
        c_prime = lambda t: c_prime_line(t.type(x.dtype).to(x.device), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :], t_i, t_f)
    else:
        c = lambda t: c_line(t.type(x.dtype).to(x.device), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :], 0, 1)
        c_prime = lambda t: c_prime_line(t.type(x.dtype).to(x.device), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :], 0, 1)

    return c, c_prime

# def generate_line(x, t_i, t_f, dt):
#     c = lambda t: c_line(((t.type(x.dtype).to(x.device) - t_i)/(t_f - t_i)), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :])
#     c_prime = lambda t: c_prime_line(((t.type(x.dtype).to(x.device) - t_i)/(t_f - t_i)), x[..., floor(t_i/dt), :], x[..., floor(t_f/dt), :])

#     return c, c_prime

# Create colored noise by convolving white noise with a moving average kernel
def generate_colored_noise(batch_shape, window_size=None, sigma=1.0):
    if window_size is None:
        window_size = max(1, batch_shape[1] // 10)  # Use regular Python max instead of torch.max
        
    # Create white noise matching batch shape
    white_noise = torch.randn(batch_shape) * sigma
    
    # Create moving average kernel
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.view(1, 1, -1)  # Reshape to (1, 1, window_size) for conv1d
    
    # Initialize colored noise array
    colored_noise = torch.zeros_like(white_noise)
    
    # Apply convolution with zero padding for each trajectory and channel
    for b in range(batch_shape[0]):  # For each batch item
        for c in range(batch_shape[2]):  # For each channel
            # For even window sizes, pad left side with window_size//2 and right side with window_size//2 - 1
            left_pad = window_size // 2
            right_pad = (window_size - 1) // 2
            padded_signal = torch.nn.functional.pad(white_noise[b,:,c], (left_pad, right_pad), mode='constant')
            # Reshape padded signal to (1, 1, length) for conv1d
            padded_signal = padded_signal.view(1, 1, -1)
            # Convolve and store result
            colored_noise[b,:,c] = torch.nn.functional.conv1d(padded_signal, kernel, padding=0, stride=1, dilation=1, groups=1).squeeze()
            
    return colored_noise

def generate_noisy_spline(x, t_i, t_f, dt, window_size=None, sigma=1.0):
    x_noise = x.clone()
    x_noise[:, 1:-1, :] += generate_colored_noise(x.shape, window_size, sigma)[:, 1:-1, :].type(x.dtype).to(x.device)
    return generate_spline(x_noise, t_i, t_f, dt)

# def generate_noisy_line(x, t_i, t_f, dt, window_size=None, sigma=1.0):
#     x_noise = x.clone()
#     x_noise[:, 1:-1, :] += generate_colored_noise(x.shape, window_size, sigma)[:, 1:-1, :].type(x.dtype).to(x.device)
#     return generate_line(x_noise, t_i, t_f, dt)

def c_stochastic_interpolant(t, x0, x, z):
    if len(x0.shape) == 2: # batches x dims
        t = t.unsqueeze(0).repeat(x0.shape[0], *np.ones(len(t.shape), dtype=int))
        x = x.unsqueeze(1)
        x0 = x0.unsqueeze(1)
        z = z.unsqueeze(-2)
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
    # print(x.shape, x0.shape, t.shape, z.shape)
    # print((1 - t)*x0 + t*x)
    # print(torch.sqrt(2*t*(1 - t))*z)
    return (1 - t)*x0 + t*x + torch.sqrt(2*t*(1 - t))*z

def c_prime_stochastic_interpolant(t, x0, x, z):
    if len(x0.shape) == 2: # batches x dims
        t = t.unsqueeze(0).repeat(x0.shape[0], *np.ones(len(t.shape), dtype=int))
        x = x.unsqueeze(1)
        x0 = x0.unsqueeze(1)
        z = z.unsqueeze(-2)
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
    return (x - x0) + ((1 - 2*t)/torch.sqrt(2*t*(1 - t)))*z

def generate_stochastic_interpolant(x, t_i, t_f, dt, z):
    x0 = x[..., floor(t_i/dt), :]
    x1 = x[..., floor(t_f/dt), :]
    n = x0.shape[-1]
    
    c = lambda t: c_stochastic_interpolant(t.type(x0.dtype).to(x0.device), x0, x1, z)
    c_prime = lambda t: c_prime_stochastic_interpolant(t.type(x0.dtype).to(x0.device), x0, x1, z)
    
    return c, c_prime

def generate_loop(x, t_i, t_f, dt, r):
    # t_f for consistency
    x0 = x[..., floor(t_i/dt), :]
    n = x0.shape[-1]
    
    # Create orthonormal vectors u and v
    if len(x0.shape) == 2: # batches x dims
        x0 = x0.unsqueeze(1)
    # u = torch.zeros_like(x0)
    # v = torch.zeros_like(x0)
    # u[..., 0] = 1
    # v[..., 1] = 1
    u = torch.randn(x0.shape)
    u = u / torch.linalg.norm(u, axis=-1, keepdim=True)
    v = torch.linalg.qr(torch.cat((u.unsqueeze(-1), torch.randn(u.shape).unsqueeze(-1)), axis=-1)).Q[..., -1]
    # print(torch.linalg.norm(u, axis=-1), torch.linalg.norm(v, axis=-1))

    u = u.to(x0.device)
    v = v.to(x0.device)

    def c_loop(t, x0, r):
        if len(x0.shape) == 2: # batches x dims
            # t = t.unsqueeze(0)
            t = t.unsqueeze(0).repeat(x0.shape[0], *np.ones(len(t.shape), dtype=int))
            # x = x.unsqueeze(1)
            x0 = x0.unsqueeze(1)
        r = r.to(x0.device)
        return x0 + r * (u * (torch.cos(t) - 1) + v * torch.sin(t))

    def c_prime_loop(t, x0, r):
        if len(x0.shape) == 2: # batches x dims
            # t = t.unsqueeze(0)
            t = t.unsqueeze(0).repeat(x0.shape[0], *np.ones(len(t.shape), dtype=int))
            # x = x.unsqueeze(1)
            x0 = x0.unsqueeze(1)
        r = r.to(x0.device)
        return r * (-u * torch.sin(t) + v * torch.cos(t))
    

    c = lambda t: c_loop(t.type(x0.dtype).to(x0.device), x0, r)
    c_prime = lambda t: c_prime_loop(t.type(x0.dtype).to(x0.device), x0, r)

    return c, c_prime
