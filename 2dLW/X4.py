import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
# plt.rcParams['text.usetex'] = True


################################## Update Parameters
N = 300
rho = 0.25
L = np.sqrt(N/rho)
L2 = L/2
apical = 5*np.pi/12
########################

def load_file(filename, N):
    data = np.fromfile(filename, dtype=np.float64)
    num_frames = data.size // N
    return data.reshape((num_frames, N))  # shape: (frames, molecules)
def unwrap_positions(x_mod):
    dx = np.diff(x_mod, axis=0)
    dx = np.where(dx >  L/2, dx - L, dx)
    dx = np.where(dx < -L/2, dx + L, dx)
    x_unwrapped = np.zeros_like(x_mod)
    x_unwrapped[0] = x_mod[0]
    x_unwrapped[1:] = x_unwrapped[0] + np.cumsum(dx, axis=0)
    return x_unwrapped



import numpy as np

def G_4(trajectories, t_start, t_end, cutoff_a):
    delta_r = trajectories[:, t_end, :] - trajectories[:, t_start, :]
    delta_r_mag = np.linalg.norm(delta_r, axis=-1)
    mobility = (delta_r_mag < cutoff_a).astype(int)

    return mobility


def compute_X4_series(trajectories, dt_steps, cutoff_a, N_particles, max_origins=2000):

    N_time_steps = trajectories.shape[1]
    time_steps_out = []
    X4_values_out = []

    dt_steps_int = np.array(dt_steps, dtype=int)

    step_size = max(1, N_time_steps // max_origins)
    t0_indices = np.arange(0, N_time_steps, step_size, dtype=int)

    for dt in dt_steps_int:
        Q_for_dt = []

        for t_start in t0_indices:
            t_end = t_start + dt

            if t_end >= N_time_steps:
                break

            c_i = G_4(trajectories, t_start, t_end, cutoff_a)
            Q_t_t0 = np.mean(c_i)
            Q_for_dt.append(Q_t_t0)

        if not Q_for_dt:
            continue

        Q_array = np.array(Q_for_dt)
        X4 = N_particles * np.var(Q_array)

        time_steps_out.append(dt)
        X4_values_out.append(X4)

    return np.array(time_steps_out), np.array(X4_values_out)



### Modify
temps = ['3.5','2.5','1.5','1.1','0.9','0.7','0.60', '0.55', '0.50','0.45']
###########
X4s = []


def compute_X4(temp): 
    prefix = f"300_LA_{temp}T"
    x_COM = load_file(f'/users/lli190/scratch/MD_analyzed_trajectories/{prefix}_x_COM.bin', N)
    y_COM = load_file(f'/users/lli190/scratch/MD_analyzed_trajectories/{prefix}_y_COM.bin', N)
    x_ = unwrap_positions(x_COM)
    y_ = unwrap_positions(y_COM)
    
    x_t_p = x_.T
    y_t_p = y_.T
    trajectories = np.stack([x_t_p, y_t_p], axis=-1)

    particle_diameter = 1.0 
    a_cutoff = 2.6 # preliminary value; modify to vary with temperature 

    
    points_per_decade = 100 
    dt_steps = np.unique(np.round(np.logspace(np.log10(1), np.log10( 2_000_000),int(points_per_decade * (np.log10( 2_000_000) - np.log10(1))) + 1)).astype(int))
    ## dt_steps = np.linspace(1, 2000000, num=20000, dtype=int) 
    
    _, X4 = compute_X4_series(trajectories, dt_steps, a_cutoff,N, max_origins=10000)
    X4s.append(X4)

for t in temps: 
    compute_X4(t)
np.save('/users/lli190/MolecularDynamics/X4s_long_10000.npy', X4s)

