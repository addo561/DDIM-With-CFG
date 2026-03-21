import torch
import torch.nn as nn
import pytorch_lightning as pl
import tqdm

class DModel(pl.LightningModule):
  def __init__(self,unet,ddpm,num_classes):
    super().__init__()
    self.unet = unet
    self.ddpm  = ddpm(self.unet,1000,num_classes)
    self.T  = 1000

  def training_step(self,batch,batch_idx):
    x,l = batch
    loss = self.ddpm.loss(x,l)
    self.log('val_loss',loss,sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
    return optimizer
   
  @torch.no_grad()
  def sample(self,omega,target_labels,num_samples):
    """DDIM sampler
        1. Look at xt
        2. Predict the noise
        3. Estimate the clean image
        4. Re-noise it for the next timestep

        classifier free guidance:
          set  guidance scale , find  noise pred of cond and uncond for  guided noise pred and use for ddim
        """
    # DDIM is designed to work with models trained using the DDPM objective (noise prediction)
    # without requiring retraining, allowing for faster sampling (e.g., 10-100 steps instead of 1000) while maintaining image quality
    # setting sigma to zero
    self.unet.eval()
    xt = torch.randn((num_samples,1,64,64),device=self.device)
    for t_idx in tqdm.tqdm(range(self.T-1,-1,-10),desc='Sampling',colour='blue'):
      #CFG first ,Dual Noise Prediction
      #conditional
      cond_embed = self.ddpm.embed(target_labels).view(num_samples, self.ddpm.embed.embedding_dim, 1, 1)
      xt_concat_cond = torch.concat([cond_embed.expand(xt.shape[0],self.ddpm.embed.embedding_dim,64,64),xt],dim=1)

      #unconditional
      uncond_l = torch.full_like(target_labels,self.ddpm.null_label)
      uncond_embed = self.ddpm.embed(uncond_l).view(num_samples, self.ddpm.embed.embedding_dim, 1, 1)
      xt_concat_uncond = torch.concat([uncond_embed.expand(xt.shape[0],self.ddpm.embed.embedding_dim,64,64),xt],dim=1)

      # pred guided
      xt_s  = torch.concat([xt_concat_cond,xt_concat_uncond],dim=0) #  stack inputs
      t_s = torch.concat([torch.full((num_samples,),t_idx,device=self.device).long(),torch.full((num_samples,),t_idx,device=self.device).long()],dim=0)
      eps_all  = self.unet(xt_s,t_s).sample
      noise_pred_cond,noise_pred_uncond = eps_all.chunk(2) #split  them  back
      pred_guided = noise_pred_uncond + omega * (noise_pred_cond  - noise_pred_uncond)

      #ddim
      t  = torch.full((xt.shape[0],),t_idx,device=self.device).long()
      t_prev  = torch.clamp(t-10,min=0)
      alpha_bars = self.ddpm.alpha_bars[t].reshape(-1,1,1,1)
      alpha_bars_prev  =  self.ddpm.alpha_bars[t_prev].reshape(-1,1,1,1)
      predict_x_0 =( xt  - (((1  -  alpha_bars)**0.5) * pred_guided)) /  (alpha_bars**0.5)
      predict_x_0 = predict_x_0.clamp(-1,1)
      pointing_xt  =   ((1 - alpha_bars_prev)**0.5) * pred_guided
      xt = (alpha_bars_prev **0.5) * predict_x_0 +  pointing_xt
    return xt
