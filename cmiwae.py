#! /usr/bin/env python

import torch

# after https://github.com/iffsid/mmvae

class CMIWAE(torch.nn.Module):
    def __init__(self, latent_size=128, default_K=1):
        super().__init__()
        
        self.latent_size=latent_size
        self.default_K = default_K
        
        # prior
        self.register_buffer("pz_c_loc", torch.zeros(latent_size, requires_grad=False))
        self.register_buffer("pz_c_scale", torch.ones(latent_size, requires_grad=False))

    
    def qz_xc(self, x, c):
        raise NotImplementedError
    
    
    def pz_c(self, c):
        # default: no dependence on condition
        return torch.distributions.Normal(loc=self.pz_c_loc, scale=self.pz_c_scale)

    
    def px_zc(self, z, c):
        raise NotImplementedError
    
    
    def prepare_input(self, *args):
        raise NotImplementedError
        #return x, condition
        
    
    def forward(self, *inp, K=None):
        if K is None:
            K = self.default_K
        
        x, c = self.prepare_input(*inp)
        
        qz_xc = self.qz_xc(x, c)
        
        z = qz_xc.rsample(torch.Size([K])) # z.shape == [K, bs, latent_dim]
        px_zc = self.px_zc(z, c)
        
        pz_c = self.pz_c(c)
        
        return qz_xc, px_zc, pz_c, z