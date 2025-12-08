import numpy as np
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call



def Eulerstep(net, input_batch, time_step):
 output_1 = net(input_batch)
 return input_batch + time_step*(output_1) 
  
def Directstep(net, input_batch, time_step):
  output_1 = net(input_batch)
  return output_1


def RK4step(net, input_batch, time_step):
 output_1 = net(input_batch)
 output_2 = net(input_batch+0.5*output_1)
 output_3 = net(input_batch+0.5*output_2)
 output_4 = net(input_batch+output_3)

 return input_batch + time_step*(output_1+2*output_2+2*output_3+output_4)/6


def PECstep(net, input_batch, time_step):
 output_1 = net(input_batch) + input_batch
 return input_batch + time_step*0.5*(net(input_batch)+net(output_1))

def PEC4step(net, input_batch, time_step):
 output_1 = time_step*net(input_batch) + input_batch
 output_2 = input_batch + time_step*0.5*(net(input_batch)+net(output_1))
 output_3 = input_batch + time_step*0.5*(net(input_batch)+net(output_2))
 return input_batch + time_step*0.5*(net(input_batch)+net(output_3))


class PEC_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(PEC_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
      output_1 = self.network(input_batch.to(self.device)) + input_batch.to(self.device)
      return input_batch.to(self.device) + self.time_step*0.5*(self.network(input_batch.to(self.device))+self.network(output_1))
    

class Euler_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Euler_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
      return input_batch.to(self.device) + self.time_step*(self.network(input_batch.to(self.device)))


class PEC4_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(PEC4_step, self).__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step

    def forward(self, input_batch):
        temp_out_1 = self.network(input_batch)
        output_1 = self.time_step*temp_out_1 + input_batch
        output_2 = input_batch + self.time_step*0.5*(temp_out_1+self.network(output_1))
        output_3 = input_batch + self.time_step*0.5*(temp_out_1+self.network(output_2))
        return input_batch + self.time_step*0.5*(temp_out_1+self.network(output_3))


class RK4_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super().__init__()
        self.network = network
        self.device = device
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step


    def forward(self, input_batch):
        input_batch = input_batch.to(self.device)
        output_1 = self.network(input_batch)
        output_2 = self.network(input_batch+0.5*output_1)
        output_3 = self.network(input_batch+0.5*output_2)
        output_4 = self.network(input_batch+output_3)

        return input_batch + self.time_step*(output_1+2*output_2+2*output_3+output_4)/6

class Direct_step(nn.Module):
    def __init__(self, network, device, time_step = 1): 
        super(Direct_step, self).__init__()
        self.network = network
        self.device = device

    def forward(self, input_batch):
      return self.network(input_batch.to(self.device))

class Implicit_Euler_step(nn.Module):
    def __init__(self, network, device, num_iters, time_step = 1): 
        super(Implicit_Euler_step, self).__init__()
        self.network = network
        self.device = device
        self.num_iters = num_iters
        # self.time_step = nn.Parameter(torch.ones(1))
        self.time_step = time_step
        #self.jacobian = torch.func.jacrev(self.implicit_euler,argnums = 1) #

    def implicit_euler(self, input_batch, output):
        return input_batch.to(self.device) + self.time_step*(self.network(output.to(self.device)))
    
    def change_num_iters(self, num_iters):
        self.num_iters = num_iters
        print('New iters: ',self.num_iters)
        return
    
    def forward(self, input_batch):
        output = input_batch.to(self.device) + self.time_step*(self.network(input_batch.to(self.device)))
        iter = 0
        jacobians = []
        eigs = []
        eigvecs = []
        while iter < self.num_iters:
            output = self.implicit_euler(input_batch, output) #shape (1,1024,1)
            #jac = self.jacobian(input_batch, output) #shape (1,1024,1,1,1024,1)
            jac = torch.autograd.functional.jacobian(self.implicit_euler,(input_batch,output))[1] #shape is (1,1024,1,1,1024,1)
            jac = jac.squeeze(0).squeeze(1).squeeze(1).squeeze(-1) #shape (1024,1024)
            eigval,eigvec = torch.linalg.eig(jac) # [1024]
            jacobians.append(jac)
            eigs.append(eigval) #[[1024]]
            eigvecs.append(eigvec)
            iter += 1
        return output, torch.stack(jacobians), torch.stack(eigs), torch.stack(eigvecs) 

class Switch_Euler_step(nn.Module):
    def __init__(self, network, hyper_network, device, num_iters, time_step = 1): 
        super(Switch_Euler_step, self).__init__()
        self.network = network
        self.hyper_network = hyper_network
        self.device = device
        self.num_iters = num_iters
        self.time_step = time_step

    def change_num_iters(self, num_iters):
        self.num_iters = num_iters
        print('New iters: ',self.num_iters)
        return

    def implicit_forward_fno(self, u_0):
        u_1 = u_0.to(self.device) + self.time_step*(self.network(u_0.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           u_1 = self.implicit_forwards_inner_fno(u_0, u_1)
           iter += 1
        return u_1

    def implicit_forwards_inner_fno(self, u_0, u_1):
        return u_0.to(self.device) + self.time_step*(self.network(u_1.to(self.device)))

    def explicit_backwards(self, u_0, u_1):
        new_params = self.hyper_network(u_0.to(self.device))
        return u_1 - self.time_step*torch.vmap(self.batch_functional, in_dims=(0, 0))(new_params, u_1.to(self.device))
    
    def batch_functional(self, params, u):
        return functional_call(self.network, params, (u.unsqueeze(0),),strict=True).squeeze(0)

    def hyper_implicit_forward(self, u_0):
        new_params = self.hyper_network(u_0.to(self.device))
        #u_1 = u_0.to(self.device) + self.time_step*(torch.vmap(self.batch_functional, in_dims=(0, 0))(new_params, u_0.to(self.device)))
        u_1 = u_0.to(self.device) + self.time_step*(self.network.forward_with_params(u_0.to(self.device), new_params))
        iter = 0
        while iter < self.num_iters:
           u_1 = self.hyper_implicit_forwards_inner(u_0, u_1, new_params)
           iter += 1
        return u_1
    def hyper_implicit_forwards_inner(self, u_0, u_1, new_params):
        #return u_0.to(self.device) + self.time_step*(torch.vmap(self.batch_functional, in_dims=(0, 0))(new_params, u_1.to(self.device)))
        return u_0.to(self.device) + self.time_step*(self.network.forward_with_params(u_1.to(self.device), new_params))
    
    def train_implicit(self, loss_func, u_0, u_1):
       return loss_func(self.implicit_backwards(u_0,u_1), u_0)
    
    def implicit_backwards(self, u_0,u_1):
        new_params = self.hyper_network(u_0.to(self.device))
        u_0 = u_1.to(self.device) - self.time_step*(torch.vmap(self.batch_functional, in_dims=(0, 0))(new_params, u_1.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           u_0 = self.implicit_backwards_inner(u_0, u_1, new_params)
           iter += 1
        return u_0
    def implicit_backwards_inner(self, u_0, u_1, new_params):
        return u_0.to(self.device) - self.time_step*(torch.vmap(self.batch_functional, in_dims=(0, 0))(new_params, u_1.to(self.device)))

    def altered_forward(self, u_0, u_1):
        u_1 = u_0.to(self.device) + self.time_step*(self.network(u_1.to(self.device)))
        iter = 0
        while iter < self.num_iters:
           u_1 = self.implicit_forwards_inner_fno(u_0, u_1)
           iter += 1
        return u_1