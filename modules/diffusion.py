import  torch
import torch.nn as nn

def cosine_scheduler(T:int, s:float=0.008):
  t = torch.linspace(0,T,T+1)
  f_t = torch.cos(((t/T + s)/ (1+s) ) * (torch.pi/2))**2
  alpha_bars = f_t/f_t[0]
  return alpha_bars

class DDPM(nn.Module):
  def __init__(self,model,T,num_classes,dim=64):
    super().__init__()
    self.model = model
    self.embed  = nn.Embedding(num_classes+1,dim)
    self.null_label = num_classes
    self.register_buffer('alpha_bars',cosine_scheduler(T))
    alpha_prev = torch.cat([torch.ones(1),self.alpha_bars[:-1]])
    betas = torch.clip(1 - (self.alpha_bars / alpha_prev),0.0001,0.9999)
    self.register_buffer('betas',betas)
    self.register_buffer('alphas',1 - self.betas )
    self.register_buffer('sigma' ,self.betas **0.5)
    self.T = T

  def gather(self,tensor,t):
    return tensor[t].view(-1,1,1,1)

  def q_sample(self,x_0 : torch.Tensor,t,eps):
    alpha_bars  = self.gather(self.alpha_bars,t)
    mean = (alpha_bars**0.5)
    var =  (1 - alpha_bars)**0.5
    return  (mean * x_0 ) +  (var  * eps)


  def loss(self,x_0: torch.Tensor,l: torch.Tensor):
    t = torch.randint(0,self.T,(x_0.shape[0],),device=x_0.device)
    eps_0 =  torch.randn_like(x_0)
    labels = l
    if self.training:
      # 10% chance of unconditional training
      mask = torch.bernoulli(torch.full(l.shape,0.1,device=x_0.device))
      labels = torch.where(mask.bool(),self.null_label,l)

    c = self.embed(labels).view(x_0.shape[0],64,1,1)
    sample = self.q_sample(x_0,t,eps_0)
    # Concat embedding channels to image channels
    xt_concat = torch.concat([c.expand(-1,-1,64,64),sample],dim=1)
    pred = self.model(xt_concat,t).sample
    loss  = nn.functional.mse_loss(eps_0,pred)
    return loss