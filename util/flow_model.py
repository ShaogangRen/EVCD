import math
import torch
import torch.nn as nn
import util.layers as layers
import numpy as np


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


class FlowCell(nn.Module):
    def __init__(self, depth = 2, z_dim = 8, w_init_sigma = 0.001):
        super(FlowCell, self).__init__()
        self.z_dim = z_dim
        self.cp_layers = nn.ModuleList()
        self.w_init_sigma = w_init_sigma
        self.cell = self.construct_model(depth=depth, dim=self.z_dim, glow=False)

    def construct_model(self, depth=1, dim=6, glow=False):
        chain = nn.ModuleList()
        for i in range(depth):
            if glow: chain.append(layers.BruteForceLayer(2))
            chain.append(layers.CouplingLayer(dim, swap=i % 2 == 0, w_init_sigma= self.w_init_sigma))
        if torch.cuda.is_available():
            return layers.SequentialFlow(chain).cuda()
        else:
            return layers.SequentialFlow(chain)

    def forward(self, x):
        _, _, ndelta_logp, _, z = self.px_compute(x, standard_normal_logprob, memory=100)
        return z, ndelta_logp

    def px_compute(self, x, prior_logdensity, memory=100):
        zeros = torch.zeros(x.shape[0], 1)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        z, delta_logp = [], []
        inds = torch.arange(0, x.shape[0]).to(torch.int64)
        for ii in torch.split(inds, int(memory ** 2)):
            z_, delta_logp_ = self.cell(x[ii], zeros[ii], reverse=False)  ##from x to z
            z.append(z_)
            delta_logp.append(delta_logp_)
        z = torch.cat(z, 0)
        delta_logp = torch.cat(delta_logp, 0)
        logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)
        logpx = logpz - delta_logp
        logpx_v = logpx.cpu().detach().numpy()
        px = np.exp(logpx_v)
        return px, logpx, - delta_logp, logpx_v, z




