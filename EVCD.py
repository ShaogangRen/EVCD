###### Code created by Shaogang Ren ###################
import torch
from util.flow_model import FlowCell
from torch import nn
import torch.distributions as dists
import torch.optim as optim
from cdt.causality.pairwise.model import PairwiseModel
import numpy as np
from sklearn import preprocessing


class FlowGraphCore(nn.Module):
    def __init__(self, max_epoch = 50, lr =0.01, ab = None, w_init_sigma=0.001,flow_depth=2):
        super(FlowGraphCore, self).__init__()
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch_size = 70
        self.xdata = ab
        self.z_dim = self.xdata.shape[1]
        self.data_size = self.xdata.shape[0]
        self.flow_depth = flow_depth
        self.max_batch = self.data_size // self.batch_size
        if self.data_size % self.batch_size > 0:
            self.max_batch = self.max_batch + 1
        self.w_init_sigma = w_init_sigma
        self.cell = FlowCell(depth=self.flow_depth, z_dim=self.z_dim, w_init_sigma = self.w_init_sigma)
        self.solver = optim.Adam(self.cell.parameters(), lr=self.lr)
        self.sigma = 1.0

    def get_batch_data(self,it, bsize = None):
        if bsize is not None:
            dsize = bsize
        else:
            dsize = self.batch_size
        idx = it % self.max_batch
        if dsize*(idx + 1) > self.data_size:
            endidx = self.data_size
        else:
            endidx = dsize*(idx + 1)
        tt = self.xdata[dsize * idx:endidx, :]
        ts = torch.from_numpy(tt).float()
        if torch.cuda.is_available():
            ts = ts.cuda()
        return ts

    def get_random_data(self, bsize = None):
        if bsize is not None:
            dsize = bsize
        else:
            dsize = self.batch_size
        idxperm = np.random.permutation(self.data_size)
        idx = idxperm[0:dsize]
        tt = self.xdata[idx, :]
        ts = torch.from_numpy(tt).float()
        if torch.cuda.is_available():
            ts = ts.cuda()
        return ts

    def train(self, pair_id):
        findex = 0
        for ep in range(self.max_epoch):
            #print('ep={}'.format(ep))
            for it in range(self.max_batch):
                x_ = self.get_batch_data(findex)
                findex = findex + 1
                #print('x_')
                #print(x_.shape)
                if x_.shape[0] <2:
                    continue
                z, ndelt_px = self.cell(x_)
                dist = dists.Normal(0, self.sigma)
                logp_z = torch.sum(dist.log_prob(z), 1)
                loss = -logp_z-ndelt_px
                mloss = torch.mean(loss)
                self.solver.zero_grad()
                mloss.backward()
                self.solver.step()
            if ep % 10 == 1:
                avg_llk = self.test_loglk_train(ep)
                print('pair_id={} ep={} loss={}  loglk_avg={} '.format(pair_id,
                                                                       ep, mloss.item(), avg_llk))
        return avg_llk

    def causal_direct(self):
        sampling_size = min(1000, self.data_size)
        #sampling_size = self.data_size
        tdata = self.get_random_data(sampling_size)
        dist = dists.Normal(0, self.sigma)
        #var0m = 0
        '''=========== E(Var(p(x_1|x_0))|x_0) ==== 0->1 ======'''
        V1gv0_list = []
        x1s = tdata[:, 1]
        for i in range(sampling_size):
            t0 = tdata[i,:][0].expand(sampling_size)
            '''===x0 is fixed, x1 changes ======='''
            xsx = torch.cat([t0.unsqueeze(1), x1s.unsqueeze(1)], 1) #
            z, ndelt_px = self.cell(xsx)
            logp_z = torch.sum(dist.log_prob(z), 1)
            px = torch.exp(logp_z + ndelt_px)
            '''== compute p_x0 =='''
            p_x0 = torch.mean(px)
            '''== compute Var(p(x_1|x_0)) =='''
            Var_X1_gv_x0 = torch.var(px/p_x0)
            V1gv0_list.append(Var_X1_gv_x0)
        V1gv0 = torch.stack(V1gv0_list, 0)
        '''===E(Var(p(x_1|x_0))|x_0)==='''
        EV1gv0 = torch.mean(V1gv0)

        '''=========== E(Var(p(x_0|x_1))|x_1) ==== 1->0  =========='''
        V0gv1_list = []
        x0s = tdata[:, 0]
        for i in range(sampling_size):
            t1 = tdata[i,:][1].expand(sampling_size)
            xsx = torch.cat([x0s.unsqueeze(1), t1.unsqueeze(1)], 1)
            z, ndelt_px = self.cell(xsx)
            logp_z = torch.sum(dist.log_prob(z), 1)
            px = torch.exp(logp_z + ndelt_px)
            '''== compute p_x1 =='''
            p_x1 = torch.mean(px)
            '''== compute Var(p(x_0|x_1)) =='''
            Var_x0_gv_x1 = torch.var(px/p_x1)
            V0gv1_list.append(Var_x0_gv_x1)
        V0gv1 = torch.stack(V0gv1_list, 0)
        '''===E(Var(p(x_0|x_1))|x_1)==='''
        EV0gv1 = torch.mean(V0gv1)
        return EV1gv0, EV0gv1

    def test_loglk_train(self, ep):
        loglk_sum = 0
        test_batch = 3
        test_sample_size = 0
        for i_test in range(0,test_batch):
            x_ = self.get_batch_data(i_test)
            test_sample_size = test_sample_size + x_.shape[0]
            loglk_sum += torch.sum(self.loglikelihood(x_))
        loglk_mean = loglk_sum/test_sample_size
        return loglk_mean

    def loglikelihood(self,x):
        z, ndelt_px = self.cell(x)
        dist = dists.Normal(0, self.sigma)
        logp_z = torch.sum(dist.log_prob(z), 1)
        log_llk = logp_z + ndelt_px
        return log_llk
    
    
class FlowGraph(PairwiseModel):
    def __init__(self, max_epoch = 50, lr =0.01, w_init_sigma=0.001, flow_depth=2):
        super(FlowGraph, self).__init__()
        self.max_epoch = max_epoch
        self.lr = lr
        self.flow_depth = flow_depth
        self.w_init_sigma = w_init_sigma
        
    def predict_proba(self, data_set, **kwargs):

        a, b = data_set
        ab = np.concatenate((a,b), axis=1)
        ab = preprocessing.scale(ab)
        fg = FlowGraphCore(max_epoch = self.max_epoch, lr =self.lr, ab = ab, w_init_sigma=self.w_init_sigma, flow_depth=self.flow_depth)
        if torch.cuda.is_available():
            fg.cuda()
        avg_llk = fg.train(pair_id=kwargs["idx"])
        d01, d10 = fg.causal_direct()
        
        del fg
        
        if d01 < d10:
            return 1
        else:
            return -1 

#        complete_n = i + 1 - start_i
#        correct = np.zeros(complete_n)
#        correct[pred == labels.Target[start_i:(i+1)]] = 1
#        acc = 1.0 * sum(correct)/complete_n
#        print('i={} avg_llk ={} accuracy = {}'.format(i, avg_llk, acc))



# """===main==="""
# def main():
#     data, labels = load_dataset('tuebingen')
#     pair_n = data.shape[0]
#     pred = np.array([])
#     start_i = 0
#     for i in range(start_i,pair_n):
#         a = data.A[i]
#         b = data.B[i]
#         ab = np.concatenate((np.expand_dims(a,1), np.expand_dims(b,1)), axis=1)
#         ab = preprocessing.scale(ab)
#         fg = FlowGraphCore(max_epoch = 1500, lr =0.0001, ab = ab, w_init_sigma=0.01, flow_depth=2)
#         if torch.cuda.is_available():
#             fg.cuda()
#         avg_llk = fg.train(pair_id=i)
#         d01, d10 = fg.causal_direct()
#         if d01 < d10:
#             pred = np.concatenate((pred, np.array([1])), 0)
#         else:
#             pred = np.concatenate((pred, np.array([-1])), 0)
#         del fg
#
#         complete_n = i + 1 - start_i
#         correct = np.zeros(complete_n)
#         correct[pred == labels.Target[start_i:(i+1)]] = 1
#         acc = 1.0 * sum(correct)/complete_n
#         print('i={} avg_llk ={} accuracy = {}'.format(i, avg_llk, acc))
#
#
# if __name__ == '__main__':
#     main()
