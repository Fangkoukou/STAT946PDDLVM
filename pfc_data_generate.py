# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2
from matplotlib import animation
from matplotlib.animation import PillowWriter
from tqdm import tqdm
from IPython.display import HTML

def generate_data(g = 0.25, N = 2**6, L = 16*np.pi, dt = 0.1, T = 1500, Nskip = 10):
    # Define PFC model constants and simulation parameters
    # Constants for the phase-field crystal model
    r = -g
    M = 1.0  # Mobility
    x = np.linspace(0, L, N)  # Grid points
    dx = x[1] - x[0]  # Grid spacing
    # Time-stepping parameters
    Nsteps = int(T / dt)  # Total number of steps
    T = Nsteps * dt
    Nframes = Nsteps // Nskip  # Number of frames for output
    
    # Initialize variables
    # Allocate arrays for Fourier transforms and output frames
    n_hat = np.empty((N, N), dtype=np.complex64)
    n = np.empty((Nframes, N, N), dtype=np.float32)
    
    # Set initial condition
    rng = np.random.default_rng(12345)  # Random number generator for reproducibility
    n0 = -0.285 + (0.285 * 0.02) * rng.standard_normal((N,N),dtype=np.float32)  # Initial mean value of the order parameter
    n[0] = n0  # Initial condition with noise
    
    # Set up Fourier variables and dealiasing
    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Fourier frequencies
    k = np.array(np.meshgrid(kx, kx, indexing='ij'), dtype=np.float32)  # Wave vectors
    k2 = np.sum(k * k, axis=0, dtype=np.float32)  # |k|^2
    kmax_dealias = (2.0/3.0)*kx.max()  # Maximum frequency for dealiasing
    dealias = np.array((np.abs(k[0]) < kmax_dealias) * (np.abs(k[1]) < kmax_dealias), dtype=bool)
    
    # Linear terms in the PDE
    L_operator = -M * k2 * (k2**2 - 2 * k2 + 1 + r)
    lineardenominator_hat = 1.0 / (1.0 - dt * L_operator)  # Precomputed denominator for time-stepping
    
    # Nonlinear term function
    def Noperator_func(n):
        return -(k2 * M * fft2(n**3)) * dealias
    
    # Time evolution loop
    nn = n[0].copy()
    n_hat[:] = fft2(n[0])  # Fourier transform of initial condition
    timestamp = [0.0]
    for i in tqdm(range(1, Nsteps)):
        # Noperator_hat = Noperator_func(nn)  # Compute nonlinear term
        n_hat[:] = (n_hat + dt * Noperator_func(nn)) * lineardenominator_hat  # Update in Fourier space
        nn[:] = ifft2(n_hat).real  # Inverse Fourier transform for the next step
        if i % Nskip == 0:  # Save frame every 100 steps
            n[i // Nskip] = nn
            timestamp.append(i*dt)
    return n, k2, dealias, timestamp