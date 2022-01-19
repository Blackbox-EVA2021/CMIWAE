#! /usr/bin/env python

import torch


# after https://github.com/iffsid/mmvae


def dreg_loss(model_out, target):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x).
    Takes into account known zero and known non-zero data."""

    qz_x, px_z, pz, z = model_out
    x_target, mask, *rest = target

    qz_x = qz_x.__class__(qz_x.loc.detach(), qz_x.scale.detach())  # stop-grad for \phi

    lqz_x = qz_x.log_prob(z).sum(-1)

    lpx_z_all = px_z.log_prob(x_target) # px_z.log_prob(..).shape == [K, bs, 1!!!!!!, w, h]

    mask = mask.bool().any(dim=-3, keepdim=True)
    lpx_z = lpx_z_all.mul(mask).view(*px_z.batch_shape[:2], -1).sum(-1) 

    lpz = pz.log_prob(z).sum(-1)

    lw = lpz + lpx_z - lqz_x  # lw.shape == [K, bs]

    with torch.no_grad():
        importance_weights = torch.nn.functional.softmax(lw, dim=0)
        if z.requires_grad:
            z.register_hook(lambda grad: importance_weights.unsqueeze(-1) * grad)
    
    if rest:
        # there is a secondary mask which will be used to evaluate individual log likelihoods
        # while the primary mask remains to calculate importance weights
        rest = [r.bool().any(dim=-3, keepdim=True) for r in rest]
        lpx_z = lpx_z_all.mul(rest[0]).view(*px_z.batch_shape[:2], -1).sum(-1)
        lw = lpz + lpx_z - lqz_x

    return (
        -(importance_weights * lw).sum(0).mean(0) / 1000  # scale loss for ergonomy
    )  # weighted sum over MC samples, mean over batches
