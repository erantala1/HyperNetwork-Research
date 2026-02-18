# HyperNetwork-Research
Evan Rantala

This research explores the use of a custom hyper network architecture for iterative constraint problems. Current applications are for autoregressive rollouts, one with a Fourier Neural Operator for KS system, and another with UNET diffusion for 2D turbulence. This work is focused on finding a stable architectural advantage when significantly reducing FLOP count. 

## Files Used for Training/Testing

- src/
    - `hyper_fno.py`            - Custom Hyper Network architecture
    - `nn_FNO.py`               - Fourier Neural Operator module
    - `nn_step_methods.py`      - Numerical methods modules
    - `nn_train_multistep.py`   - Training FNO with Hyper Network
    - `plotting.ipynb`          - Plot results of rollouts

- diffusion/
    - `hypernet.py`             - Custom Hyper Network architecture
    - `models.py`               - UNET diffusion modules
    - `train.py`                - Train UNET diffusion optionally with Hyper Network
    - `sampler.py`              - Autoregressive sampling with plotting
    - `sde.py`                  - Forward, reverse processes and sampling


## Setup

### 1. Clone the repository
```bash
git clone https://github.com/erantala1/HyperNetwork-Research
cd HyperNetwork-Research
```
### 2. Create a virtual environment
```bash
conda env create -f environment.yml
conda activate your_environment_name
```
### To train a model
```bash
cd diffusion
python train.py
```

```bash
cd src
python nn_train_multistep.py
```
