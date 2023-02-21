import torch
import torch.nn.functional as F


def reconstruction_loss(mel_gen, mel_true):
    return F.l1_loss(mel_gen, mel_true)


def kl_loss(z_p, mu_p, logv_p, logv_q, z_mask):
    z_p = z_p.float()
    logv_q = logv_q.float()
    mu_p = mu_p.float()
    logv_p = logv_p.float()
    z_mask = z_mask.float()
    kl_divergence = logv_p - logv_q - 0.5 + (0.5 * ((z_p - mu_p)**2) * torch.exp(-2. * logv_p))
    return torch.sum(kl_divergence * z_mask) / torch.sum(z_mask)

