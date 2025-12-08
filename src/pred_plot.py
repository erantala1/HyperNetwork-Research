import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from src.nn_FNO import FNO1d
from src.nn_step_methods import *
from src.hyper_fno import HyperNetwork
import numpy as np
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

time_step = 1e-3 #change back to 0.1
#lead = int((1/1e-3)*time_step)
lead = 1 #change back to 100
num_iters = 3

with open("/glade/derecho/scratch/erantala/project_runs/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])
input_size = 1024
trainN=150000
output_size = 1024 
print("Data Loaded")

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])
device = 'cuda'  #change to cpu if no cuda available

M_full = 100000

label_multi = torch.zeros([0, M_full, 1024])
label_multi = torch.cat([label_multi, input_test_torch[0:M_full:lead].unsqueeze(0)], dim=0)
label_multi = label_multi[:,::100].float()
print(f"Label multi: {label_multi.shape}")

lead = 100
M = M_full//lead

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 16  # input and output channels to the FNO layer
in_dim = input_size
hyper_hidden_scale = 2
skip_conv = True
num_mlp_layers = 14
time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver

mynet = FNO1d(modes, width, 1, 1).cuda()
myfno = FNO1d(256,256,1,1).cuda()
my_hypernet = HyperNetwork(num_mlp_layers, input_size, hyper_hidden_scale, skip_conv, mynet, device).cuda()


#net_file_path = "/glade/derecho/scratch/erantala/project_runs/model_chkpts/HFNO_Width_8_MLP_32Layers_HyperHidden_2.0_No_Spectral_Updates_100_train_multistep/chkpt_HFNO_Width_8_MLP_32Layers_HyperHidden_2.0_No_Spectral_Updates_100_train_multistep_epoch_170.pt"
net_file_path = "/glade/derecho/scratch/erantala/project_runs/model_chkpts/HFNO_Width_16_MLP_14Layers_No_Spectral_Updates_100_train_multistep/chkpt_HFNO_Width_16_MLP_14Layers_No_Spectral_Updates_100_train_multistep_epoch_200.pt"
ckpt = torch.load(net_file_path, weights_only=True)
mynet.load_state_dict(ckpt["mynet"])
my_hypernet.load_state_dict(ckpt["my_hypernet"])
step_method = Switch_Euler_step(mynet, my_hypernet, device, num_iters, time_step)
net_file_name = "/glade/derecho/scratch/erantala/project_runs/chkpt_FNO_Eulerstep_implicit_lead100_v2_epoch38.pt"
myfno.load_state_dict(torch.load(net_file_name,weights_only=True))
fno_step_method = Switch_Euler_step(myfno, my_hypernet, device, num_iters, time_step)


def split_batch(batch):
    u_0 = batch[:,0]
    u_1 = batch[:,1]
    return u_0,u_1

def _numel_from_value(v):
    t = v.type()
    sizes = t.sizes() if hasattr(t, "sizes") else None
    if sizes is None:
        return 0
    n = 1
    for d in sizes:
        if d is None:
            return 0
        n *= int(d)
    return n


def _flops_from_outputs(outputs, which=0):
    if not outputs or which >= len(outputs):
        return 0
    return _numel_from_value(outputs[which])

#using 8 as approximation
def gelu_flop_jit(inputs, outputs):
    return 8 * _flops_from_outputs(outputs, 0)

def mul_flop_jit(inputs, outputs):
    return _flops_from_outputs(outputs, 0)

def add_flop_jit(inputs, outputs):
    return _flops_from_outputs(outputs, 0)

#Testing FLOPS for hypernet
#Do flop count analysis for one pass of hyper net in inference mode - only making the new params once
#then do flop count analysis for fno of same width 4 times for total flops - in inference fno is called 4 iterations
my_hypernet.flop_mode = True
my_hypernet.train_mode = False
my_hypernet.eval()
with torch.no_grad():
    fca = FlopCountAnalysis(my_hypernet, (torch.reshape(label_test_torch[0,:],(1,input_size,1)).cuda(), torch.reshape(label_test_torch[0,:],(1,input_size,1)).cuda()))
    fca.set_op_handle("aten::gelu", gelu_flop_jit)
    fca.set_op_handle("aten::mul", mul_flop_jit)
    fca.set_op_handle("aten::add",add_flop_jit)
    fno_fca = FlopCountAnalysis(mynet, (torch.reshape(label_test_torch[0,:],(1,input_size,1)).cuda()))
    fno_fca.set_op_handle("aten::gelu", gelu_flop_jit)
    fno_fca.set_op_handle("aten::mul", mul_flop_jit)
    fno_fca.set_op_handle("aten::add",add_flop_jit)
    hfno_flops_per_sample = fca.total() + (fno_fca.total() * 4)
    hfno_flops_per_sample /= 1e12
my_hypernet.flop_mode = False
print(f"HFNO TFLOPS Per Sample: {hfno_flops_per_sample}")

#Testing FLOPS for Full 256 width FNO
myfno.eval()
with torch.no_grad():
    fca2 = FlopCountAnalysis(myfno, (torch.reshape(label_test_torch[0,:],(1,input_size,1)).cuda()))
    fca2.set_op_handle("aten::gelu", gelu_flop_jit)
    fca2.set_op_handle("aten::mul", mul_flop_jit)
    fca2.set_op_handle("aten::add",add_flop_jit)
    fno_flops_per_sample = fca2.total() * 4
    fno_flops_per_sample /= 1e12
print(f"FNO TFLOPS Per Sample: {fno_flops_per_sample}")



T = 1000
net_pred = np.zeros([1,M-1,1024])
net_pred_fno = np.zeros([1,M-1,1024])
print(f"M:{M}")
for k in range(0,M-1):
    if (k==0):
        net_output = step_method.hyper_implicit_forward(torch.reshape(label_multi[:,0,:],(1,input_size,1)))
        net_pred[:,k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
        net_output_fno = fno_step_method.implicit_forward_fno(torch.reshape(label_multi[:,0,:],(1,input_size,1)))
        net_pred_fno[:,k,:] = torch.reshape(net_output_fno,(1,input_size)).detach().cpu().numpy()
        
    else:
        net_output = step_method.hyper_implicit_forward(torch.reshape(torch.from_numpy(net_pred[:,k-1,:]),(1,input_size,1)).float().cuda()) 
        net_pred[:, k,:] = torch.reshape(net_output,(1, input_size)).detach().cpu().numpy()
        net_output_fno = fno_step_method.implicit_forward_fno(torch.reshape(torch.from_numpy(net_pred_fno[:,k-1,:]),(1,input_size,1)).float().cuda()) 
        net_pred_fno [:, k,:] = torch.reshape(net_output_fno,(1,input_size)).detach().cpu().numpy()

print('Eval Finished')



trange = M-1
print(trange)
t_final = M*1e-1
print(t_final)
label_lead100 = label_multi
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
print(f"Pred shape:{net_pred.shape}")
print(f"Label lead shape:{label_lead100.shape}")
axs[2].plot(np.linspace(0,t_final, net_pred.shape[1]), np.sqrt(np.mean((net_pred[:, 0:trange] - label_lead100[:, 1:trange+1].detach().cpu().numpy())**2, axis=2)).mean(0), color='red', label='FNO Width 64 MLP Layers 4 with Spectral Updates, TFLOPS: 0.00023')
axs[2].plot(np.linspace(0,t_final, net_pred_fno.shape[1]), np.sqrt(np.mean((net_pred_fno[:, 0:trange] - label_lead100[:, 1:trange+1].detach().cpu().numpy())**2, axis=2)).mean(0), color='black', label='FNO Width 256, TFLOPS: 0.0015 ')
axs[2].set_xlabel('Seconds')
axs[2].set_xlabel('Seconds')
axs[2].set_ylabel(r'$RMSE$')
axs[2].set_title('Comparison between sum of Epsilon and true error, dt=0.1')


handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="center left",
           bbox_to_anchor=(1.02, 0.5),
           borderaxespad=0.,
           frameon=True,
           ncol=1,
           fontsize=9,
           title="Models")




vmin = np.nanmin([np.nanmin(net_pred), np.nanmin(label_lead100)])
vmax = np.nanmax([np.nanmax(net_pred), np.nanmax(label_lead100)])

extent = [0, t_final, 0, t_final]


im0 = axs[0].imshow(net_pred.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
axs[0].set_title("Prediction - Width 8 Layers 32 Hidden *2 Epoch 170")
axs[0].set_xlabel("Seconds")
axs[0].set_ylabel("x")

im1 = axs[1].imshow(label_lead100.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
axs[1].set_title("Truth")
axs[1].set_xlabel("Seconds")



save_dir = "./plots"
filename = "TESTING"
savepath = os.path.join(save_dir,filename)
fig.savefig(savepath, bbox_inches='tight') 