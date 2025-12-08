import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ap = argparse.ArgumentParser() #add specific args when running
ap.add_argument('folder')
ap.add_argument('--prefix', default='KS_pred_Implicit_Euler_step_FNO_jacs_for_1k') 
ap.add_argument('--fps', type=int, default=16)
args = ap.parse_args()

folder  = Path(args.folder).expanduser()
pattern = folder / f'{args.prefix}_chunk_*.npy'
files   = sorted(
    glob.glob(str(pattern)),
    key=lambda s: int(s.split('_chunk_')[-1].split('.npy')[0]) 
)
if not files:
    raise FileNotFoundError('no chunk files')


eig_vals = []

for f in files: #appends eigenvalues from each chunk
    blk = np.load(f, allow_pickle=True).item()
    ev = blk['Eigenvalues'] 
    #V  = blk['Eigenvectors']

    eig_vals.append(ev) 

    #del V, blk

E = np.concatenate(eig_vals, axis=0)
K, J, d = E.shape
print(f'\nEigen-values assembled:  {E.shape}')



fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_aspect('equal', 'box')
ax.ticklabel_format(style="sci", scilimits=(-2,2))
colors = plt.cm.viridis(np.linspace(0,1,J))
scatters = [ax.plot([], [], 'o', ms=4, color=colors[j])[0] for j in range(J)]

def set_window(ev):
    rmax = np.abs(ev - ev.mean()).max()
    half = 1.05 * max(rmax, 1e-10)
    c = ev.mean()
    ax.set_xlim(c.real-half, c.real+half)
    ax.set_ylim(c.imag-half, c.imag+half)
    ax.ticklabel_format(style='plain')  

def init():
    for s in scatters: s.set_data([],[])
    set_window(E[0,0])
    ax.set_title('λ  t=0 iter=0')
    return scatters

def update(frame):
    k, j = divmod(frame, J)
    ev = E[k, j]
    set_window(ev)
    ax.set_title(f'λ  t={k} iter={j}')
    for jj, sc in enumerate(scatters):
        sc.set_data(E[k,jj].real, E[k,jj].imag)
    return scatters

ani = animation.FuncAnimation(fig, update, frames=K*J,
                              init_func=init, blit=True,
                              interval=1000/args.fps)
ani.save('eig_animation_lead_1.mp4', dpi=120, fps=args.fps)
print('saved eig_animation_lead_1.mp4')

