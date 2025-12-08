# implicit-euler-research
Evan Rantala

This research explores pairing a custom hypernetwork architecture with implicit numerical methods and amFourier Neural Operator for learning stable rollouts of nonlinear PDEs. This work is focused on finding a stable architectural advantage when significantly reducing FLOP count. 

## Files Used for Training/Testing

- src/
    - `hyper_fno.py`            - Custom Hyper Network architecture
    - `nn_FNO.py`               - Fourier Neural Operator module
    - `nn_step_methods.py`      - Numerical methods modules
    - `nn_train_multistep.py`   - Training FNO with Hyper Network
    - `plotting.ipynb`          - Plot results of rollouts

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/erantala1/Implicit_Euler_Research
cd Implicit_Euler_Research
```
### 2. Create a virtual environment
```bash
conda env create -f environment.yml
conda activate your_environment_name
```
### To train a model
```bash
cd src
python nn_train_multistep.py
```
