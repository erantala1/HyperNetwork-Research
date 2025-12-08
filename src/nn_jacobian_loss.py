import torch
import torch.nn.functional as F

def compute_vjp_batch(model, t_1, t_2):
   output, vjp_func = torch.func.vjp(model, t_1)
   vjp_out = vjp_func(t_2 - output)[0]
   return output, vjp_out

def jacobian_loss(model, t_1, t_2):

   output, vjp_out = torch.vmap(compute_vjp_batch, in_dims = (None, 0, 0))(model, t_1, t_2)

   # loss_1 = F.mse_loss(output, t_2) 
   loss_2 = torch.linalg.norm(vjp_out, dim=1).mean(0)

   return loss_2

def spectral_jacobian_loss(model, t_1, t_2, lambda_reg_1, lambda_reg_2):

   output, vjp_out = torch.vmap(compute_vjp_batch, in_dims = (None, 0, 0))(model, t_1, t_2)

   # loss_1 = F.mse_loss(output, t_2) 
   # loss_2 = F.mse_loss(vjp_out, torch.zeros_like(vjp_out)) 
   loss_2 = torch.linalg.norm(vjp_out, dim=1).mean(0)
   # print(loss_1, loss_2)

   # out_du_fft =torch.fft.rfft((output-t_2),dim=1).cuda()
   # target_du_fft = torch.fft.rfft(t_2 - t_1,dim=1).cuda()

   loss_3 = torch.linalg.norm(torch.fft.rfft(vjp_out, dim=1), dim=1).mean(0)

   return lambda_reg_1 * loss_2  + lambda_reg_2 * loss_3