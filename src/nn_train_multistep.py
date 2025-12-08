import numpy as np
import torch
import os
print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
from src.count_trainable_params import count_parameters
import pickle
from src.nn_FNO import FNO1d
from src.nn_step_methods import *
from src.hyper_fno import HyperNetwork
import wandb
from collections import deque

time_step = 1e-1
lead = int((1/1e-3)*time_step)
print(lead, 'FNO')

net_name = 'Single_MLP_Hyper_FNO_Width_16_BS150_Layers10_HW05_'+str(lead)+'_train_multistep'
print(net_name)

chkpts_path_outputs = '/glade/derecho/scratch/erantala/project_runs/model_chkpts'
net_chkpt_path = '/glade/derecho/scratch/erantala/project_runs/model_chkpts/'+str(net_name)+'/'

starting_epoch = 0
print('Starting epoch '+str(starting_epoch))

if not os.path.exists(net_chkpt_path):
    os.makedirs(net_chkpt_path)
    print(f"Folder '{net_chkpt_path}' created.")
else:
    print(f"Folder '{net_chkpt_path}' already exists.")


with open("/glade/derecho/scratch/cainslie/conrad_net_stability/training_data/KS_1024.pkl", 'rb') as f:
    data = pickle.load(f)
data=np.asarray(data[:,:250000])

trainN = 150000
input_size = 1024
output_size = 1024

epochs = 100
batch_size = 150
batch_time = 2
print('Batch size ', batch_size)
wavenum_init = 100
lamda_reg = 5
evalN = 10000
batch_time_test = 20
print('Batch time test: '+str(batch_time_test))
print(data.shape)
def Dataloader(data,batch_size,batch_time, key):
    time_chunks = []
    for i in range(data.shape[0] - batch_time*lead):
        time_chunks.append(data[i:i+batch_time*lead:lead])
    extra = len(time_chunks) % batch_size
    if extra==0:
        time_chunks = np.array(time_chunks)
    else:
        time_chunks = np.array(time_chunks[:-extra])
    rng = np.random.default_rng(key)
    split = rng.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    return split

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


device = 'cuda'  #change to cpu if no cuda available

#model parameters
modes = 256 # number of Fourier modes to multiply
width = 16  # input and output channels to the FNO layer 
in_dim = input_size
hyper_hidden_scale = 0.25
learning_rate = 1e-4
num_mlp_layers = 13

which_params = ["fc0.weight", "fc0.bias", "w0.weight", "w0.bias", "w1.weight", "w1.bias", "w2.weight", "w2.bias", "w3.weight", "w3.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
rank = 1
mynet = FNO1d(modes, width, 1, 1).cuda()
my_hypernet = HyperNetwork(num_mlp_layers, in_dim, hyper_hidden_scale, which_params, rank, mynet, device).cuda()

'''
net_file_path = "/glade/derecho/scratch/erantala/project_runs/model_chkpts/Hyper_FNO_Width_16_Forward_BS100_Layers8_Clip2_Tmax25_100_train_multistep/chkpt_Hyper_FNO_Width_16_Forward_BS100_Layers8_Clip2_Tmax25_100_train_multistep_epoch_100.pt"
ckpt = torch.load(net_file_path, map_location=device, weights_only=True)
mynet.load_state_dict(ckpt["mynet"])
my_hypernet.load_state_dict(ckpt["my_hypernet"])
'''
num_iters = 0
step_net = Switch_Euler_step(mynet, my_hypernet, device, num_iters,time_step).to(device)
#step_net = Switch_Euler_step(mynet, None, device, num_iters,time_step).to(device)
print(f"Step net iterations:{step_net.num_iters}")
count_parameters(mynet)
count_parameters(my_hypernet)

# just fno
#optimizer = optim.AdamW(mynet.parameters(), lr=1e-5, weight_decay=3e-3)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)


# fno and hypernet

'''
optimizer = optim.AdamW(mynet.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
optimizer_hyper = optim.AdamW(my_hypernet.parameters(), lr=1e-5, weight_decay=1e-6)
scheduler_hyper = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hyper, T_max=100)
'''

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.995)
optimizer_hyper = optim.AdamW(my_hypernet.parameters(), lr = 1e-5)
scheduler_hyper = optim.lr_scheduler.ExponentialLR(optimizer_hyper, 0.995)

'''
optimizer.load_state_dict(ckpt["optimizer"])
optimizer_hyper.load_state_dict(ckpt["optimizer_hyper"])
scheduler.load_state_dict(ckpt["scheduler"])
scheduler_hyper.load_state_dict(ckpt["scheduler_hyper"])
starting_epoch = ckpt.get("epoch", 0)
'''
rng = np.random.default_rng()
key = rng.integers(100, size=1)
train_data = Dataloader(data[:,0:trainN+lead].T, batch_size = batch_size, batch_time = batch_time, key=key)
train_data = torch.from_numpy(train_data).float()

rng = np.random.default_rng()
key = rng.integers(100, size=1)
test_data = Dataloader(data[:,trainN+lead:].T, batch_size = 300, batch_time = batch_time_test, key=key)
print("Data Loaded")
test_data = torch.from_numpy(test_data).float()

class Loss_Multistep(nn.Module):
    def __init__(self, model, batch_time, loss_func):
        super().__init__()
        self.model = model
        self.batch_time = batch_time
        self.loss_func = loss_func
    
    '''
    def forward(self, batch):        
        x_i = self.model.explicit_backwards(batch[:,0],batch[:,1]) 
        loss = self.loss_func(x_i, batch[:,0])
        
        for i in range(self.batch_time, 2,-1):
            x_i = self.model.implicit_forward(x_i.detach())
            loss += self.loss_func(x_i, batch[:,i])
        
        return loss
    '''
    def forward(self, batch):
        x_i = self.model.hyper_implicit_forward(batch[:,0])
        loss = self.loss_func(x_i, batch[:,1])
        return loss
    
    

loss_fn = nn.MSELoss(reduction='mean')  #for basic loss func
loss_func = lambda e: torch.linalg.norm(e, dim=1).mean(0) 
loss_net_test = Loss_Multistep(step_net, batch_time_test, loss_fn)


run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="erantala-university-of-california",
        # Set the wandb project where this run will be logged.
        project="Hyper_FNO",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "Single MLP Hypernet FNO width 16 Tao 0",
            "dataset": "KS",
            "batch_size": 150,
            "Modes": 256,
            "Width": 16,
            "epochs": 100,
            "loss_fn": "MSE",
            "MLP layers": 13,
            "Hyper scheduler": "Exp 1e-5 0.995",
            "FNO scheduler": "Exp 1e-4 0.995",
            "Hyper Hidden" : 0.25,
            "Gradient Clipping": "None"
        },
    )
wandb.define_metric("epoch")
wandb.define_metric("epoch/*", step_metric="epoch")


torch.set_printoptions(precision=10)
best_loss = 1e5
global_step = 0

grad_history = deque(maxlen=100)
grad_history_2 = deque(maxlen=100)
for ep in range(starting_epoch, epochs+1):
    running_loss = 0.0
    for n in range(train_data.shape[0]):
        batch = train_data[n].unsqueeze(-1).to(device)
    

        optimizer.zero_grad()
        optimizer_hyper.zero_grad()
        loss = loss_net_test(batch)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(my_hypernet.parameters(), 1.0)

        optimizer.step()
        optimizer_hyper.step()
        
        
        running_loss += loss.detach().item()
        #print(loss)
        global_step += 1

    net_loss = (running_loss/(train_data.shape[0]))
    key = np.random.randint(len(test_data))
    with torch.no_grad():
        test_loss = loss_net_test(test_data[key].unsqueeze(-1).to(device))
    scheduler.step()
    scheduler_hyper.step()
    print(f'Epoch : {ep}, Train Loss : {net_loss/(batch_time-1)}, Test Loss : {test_loss/(batch_time_test-1)}')

    run.log({
            "epoch": ep,
            "epoch/loss": net_loss/(batch_time-1),
            "epoch/test_loss": test_loss/(batch_time_test-1),
        })
    
    if best_loss > test_loss:
        print('Saved!!!')
        torch.save({"mynet": mynet.state_dict(), 
            "my_hypernet": my_hypernet.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            "optimizer_hyper": optimizer_hyper.state_dict(), 
            "scheduler": scheduler.state_dict(), 
            "scheduler_hyper": scheduler_hyper.state_dict(), 
            "epoch": ep,
            }, 
            chkpts_path_outputs+'/'+str(net_name)+'/'+'chkpt_'+net_name+'.pt')
        print('Checkpoint updated')
        print(chkpts_path_outputs+'/'+str(net_name)+'/'+'chkpt_'+net_name+'.pt')
        best_loss = test_loss

    if ep % 10 == 0:
        print(chkpts_path_outputs+'/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')
        torch.save({"mynet": mynet.state_dict(), 
            "my_hypernet": my_hypernet.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            "optimizer_hyper": optimizer_hyper.state_dict(), 
            "scheduler": scheduler.state_dict(), 
            "scheduler_hyper": scheduler_hyper.state_dict(), 
            "epoch": ep,
            }, 
            chkpts_path_outputs+'/'+str(net_name)+'/'+'chkpt_'+net_name+'_epoch_'+str(ep)+'.pt')

torch.save({"mynet": mynet.state_dict(), 
            "my_hypernet": my_hypernet.state_dict(), 
            "optimizer": optimizer.state_dict(), 
            "optimizer_hyper": optimizer_hyper.state_dict(), 
            "scheduler": scheduler.state_dict(), 
            "scheduler_hyper": scheduler_hyper.state_dict(), 
            "epoch": ep,
            }, 
            net_chkpt_path+'chkpt_'+net_name+'_final.pt')
torch.set_printoptions(precision=4)
print("Model Saved")