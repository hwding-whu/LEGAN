import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math


class BCEWithLogits(nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            loss_fake = pred_fake.mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss


class Softplus(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.softplus(-pred_real).mean()
            loss_fake = F.softplus(pred_fake).mean()
            loss = loss_real + loss_fake
            return loss
        else:
            loss = F.softplus(-pred_real).mean()
            return loss


class Entorpy(nn.Module):
    def forward(self, encoder, net_G, cf, z, batch_size):
        labels = cf(net_G(z))
        labels = labels.argmax(dim=1)
        labels = labels.detach().cpu().numpy()
        prob = {i[0]: i[1] / (batch_size) for i in Counter(labels).items()}
        loss = - sum([i[1] * math.log2(i[1]) for i in prob.items()])  # 计算熵
        return loss