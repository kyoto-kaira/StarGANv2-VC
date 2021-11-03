#coding:utf-8

import os
import torch

from torch import nn
from munch import Munch
from transforms import build_transforms

import torch.nn.functional as F
import numpy as np


def compute_g_loss(nets, args, x_real, x_refs, label):
    args = Munch(args)

    # compute style vectors
    with torch.no_grad():
        s_trg =  0.5 * nets.style_encoder(x_refs[0])
        s_trg += 0.5 * nets.style_encoder(x_refs[1])
        s_orig = nets.style_encoder(x_real)
    
    # compute ASR/F0 features (real)
    with torch.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
        ASR_real = nets.asr_model.get_feature(x_real)

    # convert 1st stage
    with torch.no_grad():
        x_fake1 = nets.generator(x_real, s_trg, masks=None, F0=GAN_F0_real)
        x_cyc = nets.generator(x_real, s_orig, masks=None, F0=GAN_F0_real)
    
    x_fake2 = nets.hearnet(x_fake1, x_cyc-x_fake1)

    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake2)
    ASR_fake = nets.asr_model.get_feature(x_fake2)
    
    # norm consistency loss
    x_fake_norm = log_norm(x_fake2)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm) - args.norm_bias))**2).mean()
    
    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)
    
    # style F0 loss (style initialization)
    if args.lambda_f0_sty > 0:
        F0_sty0, _, _ = nets.f0_model(x_refs[0])
        F0_sty1, _, _ = nets.f0_model(x_refs[1])
        F0_sty = 0.5 * (F0_sty0 + F0_sty1)
        loss_f0_sty = F.l1_loss(compute_mean_f0(F0_fake), compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = torch.zeros(1).mean()
    
    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)

    # reconstruction loss
    loss_rec = F.mse_loss(x_real[label == 0], x_fake2[label == 0])

    loss = args.lambda_norm * loss_norm \
           + args.lambda_asr * loss_asr \
           + args.lambda_f0 * loss_f0 \
           + args.lambda_f0_sty * loss_f0_sty \
           + args.lambda_rec * loss_rec

    return loss, Munch(norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       rec=loss_rec.item())
    
# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss