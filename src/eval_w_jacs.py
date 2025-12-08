import numpy as np
#import scipy.io
import torch
import os
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
#import netCDF4 as nc
#from prettytable import PrettyTable
#from count_trainable_params import count_parameters    
import pickle
#import matplotlib.pyplot as plt
from src.nn_FNO import FNO1d
#from nn_MLP import MLP_Net
from src.nn_step_methods import *

skip_factor = 0 #Number of timesteps to skip (to make the saved data smaller), set to zero to not save a skipped version

time_step = 1e-3
lead = int((1/1e-3)*time_step*100)
print(lead,'lead')

path_outputs = '/glade/derecho/scratch/erantala/project_runs/outputs' #this is where the saved graphs and .mat files end up

net_file_name = "/glade/derecho/scratch/erantala/project_runs/chkpt_FNO_Eulerstep_implicit_lead100_v2_epoch38.pt"
print(net_file_name)
#change this to use a different network

step_func = Implicit_Euler_step#this determines the step funciton used in the eval step, has inputs net (pytorch network), input batch, time_step

print(step_func)

eval_output_name = 'KS_pred_Implicit_Euler_step_FNO_jacs_for_lead_100'  # what to name the output file, .mat ending not needed
print(eval_output_name)

with open("/glade/derecho/scratch/erantala/project_runs/KS_1024.pkl", 'rb') as f: #change based on eval data location.
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN=15000 #dont explicitly need this as no training is done in file
input_size = 1024
output_size = 1024 

input_test_torch = torch.from_numpy(np.transpose(data[:,trainN:])).float()
label_test_torch = torch.from_numpy(np.transpose(data[:,trainN+lead::lead])).float()
label_test = np.transpose(data[:,trainN+lead::lead])
print(label_test_torch.shape)

time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu if no cuda available

#model parameters

input_size = 1024 
hidden_layer_size = 2000 
output_size = 1024

modes = 256 # number of Fourier modes to multiply
width = 256  # input and output chasnnels to the FNO layer


my_net = FNO1d(modes, width, time_future, time_history)
# my_net = MLP_Net(input_size, hidden_layer_size, output_size)

my_net.load_state_dict(torch.load(net_file_name,weights_only=True))
my_net.cuda()

#step_method = step_func(my_net, device, time_step)

num_iters = 10
step_method = step_func(my_net, device, num_iters, time_step)  #for implicit methods

# M = int(np.floor(99998/lead))
M = label_test_torch.shape[0] - 1
M = 300
net_pred = np.zeros([M,np.size(label_test,1)])
print(M)
print('Model loaded')

noise_var = 0

print('Noise number: ', noise_var)

noised_input = (noise_var)*torch.randn(1,1024).cuda()
noised_input = label_test_torch[0,:].cuda() + noised_input
ygrad = torch.zeros([M,num_iters,input_size, input_size]) #added num_iters dimension 
eigvals = torch.zeros([M,num_iters, input_size],dtype=torch.cfloat)
eigvecs = torch.zeros([M,num_iters,input_size,input_size],dtype = torch.cfloat)
#ygrad_truth = torch.zeros([M,num_iters,input_size, input_size])

print(noised_input.size())

for k in range(0,M):
 
    if (k==0):

        net_output, ygrad[k],eigvals[k],eigvecs[k] = step_method(torch.reshape(noised_input,(1,input_size,1)))
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
        #print(sum(sum(abs(net_pred))))
        #temp_mat = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[k,:],(1,input_size,1))) #Use these for FNO
        #ygrad_truth[k] = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[k,:],(1,input_size,1))).reshape(1,input_size, input_size)

    else:

        net_output, ygrad[k],eigvals[k],eigvecs[k] = step_method(torch.reshape(torch.from_numpy(net_pred[k-1,:]),(1,input_size,1)).float().cuda()) 
        net_pred [k,:] = torch.reshape(net_output,(1,input_size)).detach().cpu().numpy()
        #temp_mat = torch.autograd.functional.jacobian(step_method, net_output) #Use these for FNO
        #ygrad_truth[k] = torch.autograd.functional.jacobian(step_method, torch.reshape(input_test_torch[k,:],(1,input_size,1))).reshape(1,input_size, input_size)

    if k%10==0:
        print(k) 
       
print('Eval Finished')

def calc_save_chunk(net_pred_chunk,chunk_num, ygrad_chunk,eig_chunk,eig_vec_chunk):
    matfiledata_output = {}
    matfiledata_output[u'prediction'] = net_pred_chunk
    matfiledata_output[u'Jacobians'] = ygrad_chunk
    matfiledata_output[u'Eigenvalues'] = eig_chunk
    matfiledata_output[u'Eigenvectors'] = eig_vec_chunk

    print('First save done')
    np.save(path_outputs+'/'+eval_output_name+'/'+eval_output_name+'_chunk_'+str(chunk_num), matfiledata_output)
    print('Saved main file')


if not os.path.exists(path_outputs+'/'+eval_output_name+'/'):
    os.makedirs(path_outputs+'/'+eval_output_name+'/')
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' created.")
else:
    print(f"Folder '{path_outputs+'/'+eval_output_name+'/'}' already exists.")

prev_ind = 0
chunk_count = 0
num_chunks = 100
for chunk in np.array_split(net_pred, num_chunks):
    current_ind = prev_ind + chunk.shape[0]
    calc_save_chunk(chunk, chunk_count, ygrad[prev_ind:current_ind],eigvals[prev_ind:current_ind],eigvecs[prev_ind:current_ind])
    prev_ind = current_ind
    print(chunk_count, prev_ind)
    chunk_count += 1
print('Data saved')