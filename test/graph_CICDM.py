import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from tqdm import tqdm

from sklearn import metrics
from torch_scatter import scatter
from torch_geometric.utils import softmax


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


class CDM_Net(nn.Module):
    def __init__(self, item_num: int, conc_num: int, item_conc_num: int, potential_num: int, conc_conc_adj: Tensor, only_conc: bool = False):
        super(CDM_Net, self).__init__()
        self.item_conc_w = nn.Parameter(torch.randn(item_conc_num))
        self.item_pote_w = nn.Parameter(torch.randn((item_num, potential_num)))
        conc_conc_adj[torch.eye(conc_num, dtype=torch.bool)] = 1
        self.conc_conc_w = nn.Parameter(conc_conc_adj * 5)
        # self.conc_conc_w = nn.Parameter(torch.eye(conc_num) * np.log(9 * (conc_num - 1)))
        self.guess = nn.Parameter(torch.ones(item_num) * -2 + torch.normal(0., 0.1, size=(item_num,)))
        self.slide = nn.Parameter(torch.ones(item_num) * -2 + torch.normal(0., 0.1, size=(item_num,)))
        self.lambd = nn.Parameter(torch.ones(item_num) * -2 + torch.normal(0., 0.1, size=(item_num,)))
        self.only_conc = only_conc

    def forward(self, data):
        # ------------------------ modele A start ------------------
        # source: item --> conc
        s_item_conc_w_ = torch.sigmoid(self.item_conc_w[data['source'].item_conc_index])

        s_conc_track = data['source'].conc_track  # 相同知识点的累加
        s_item_conc_w = s_item_conc_w_ / (scatter(s_item_conc_w_, s_conc_track, reduce="sum")[s_conc_track])

        s_norm_score = data['source'].score * s_item_conc_w
        s_conc = scatter(s_norm_score, s_conc_track, reduce="sum")

        # source conc --> all conc
        all_item_conc_w_ = self.conc_conc_w[data['source'].unique_conc_index, :]
        all_item_conc_w = softmax(all_item_conc_w_, data['source'].stu_conc_track, dim=0)
        all_conc = scatter(all_item_conc_w * s_conc.view(-1, 1), data['source'].stu_conc_track, dim=0, reduce="sum")
        t_conc = all_conc[data['target'].stu_conc_track, data['target'].unique_conc_index]

        # target: conc --> item
        t_conc_extend = t_conc[data['target'].conc_track]
        t_conc_item_w_ = torch.sigmoid(self.item_conc_w[data['target'].item_conc_index])

        t_item_track = data['target'].item_track  # 相同习题的累加
        t_conc_item_w = t_conc_item_w_ / (scatter(t_conc_item_w_, t_item_track, reduce="sum")[t_item_track])

        t_score_A = scatter(t_conc_extend * t_conc_item_w, t_item_track, reduce="sum")
        # ------------------------- modele A end -------------------

        if not self.only_conc:
            # ------------------------ modele B start ------------------
            s_item_pote_w_ = self.item_pote_w[data['source'].unique_item_index, :]
            s_item_pote_w = softmax(s_item_pote_w_, data['source'].stu_track, dim=0)
            s_pote = scatter(s_item_pote_w * data['source'].unique_item_score.view(-1, 1), data['source'].stu_track, dim=0, reduce="sum")

            t_pose_extend = s_pote[data['target'].stu_track]
            t_pote_item_w_ = self.item_pote_w[data['target'].unique_item_index, :]
            t_pote_item_w = torch.softmax(t_pote_item_w_, dim=1)

            t_score_B = (t_pote_item_w * t_pose_extend).sum(dim=1).view(-1)
            # ------------------------- modele B end -------------------

            # merge A and B
            item_idx = data['target'].unique_item_index
            lambd = torch.sigmoid(self.lambd[item_idx])
            t_score = (1 - lambd) * t_score_A + lambd * t_score_B

        else:
            t_score = t_score_A

        # t_score = t_score.clamp(1e-8, 1 - 1e-8)
        # slide & guess
        slide = torch.sigmoid(self.slide[item_idx])
        guess = torch.sigmoid(self.guess[item_idx])
        pred = (1 - slide) * t_score + guess * (1 - t_score)
        return pred

    def get_cognitive_state(self, data):
        # source: item --> conc
        s_item_conc_w_ = torch.sigmoid(self.item_conc_w[data['source'].item_conc_index])

        s_conc_track = data['source'].conc_track  # 相同知识点的累加
        s_item_conc_w = s_item_conc_w_ / (scatter(s_item_conc_w_, s_conc_track, reduce="sum")[s_conc_track])

        s_norm_score = data['source'].score * s_item_conc_w
        s_conc = scatter(s_norm_score, s_conc_track, reduce="sum")

        # source conc --> all conc
        all_item_conc_w_ = self.conc_conc_w[data['source'].unique_conc_index, :]
        all_item_conc_w = softmax(all_item_conc_w_, data['source'].stu_conc_track, dim=0)
        all_conc = scatter(all_item_conc_w * s_conc.view(-1, 1), data['source'].stu_conc_track, dim=0, reduce="sum")

        return all_conc


class CICDM():
    def __init__(self, item_num: int, conc_num: int, item_conc_num: int, potential_num: int = 16, learn_rate=1e-3, device='cpu'):
        self.conc_num = conc_num
        self.device = device
        self.net = CDM_Net(item_num, conc_num, item_conc_num, potential_num).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learn_rate)
        self.loss_function = torch.nn.BCELoss(reduction='mean')
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.net.parameters())))

    def fit(self, data_loader, epochs, test_data=None):
        loss_list = []
        for epoch in range(epochs):
            # data_list = [x.to(self.device) for x in data_loader]
            for betch_data in tqdm(data_loader, "[Epoch:%s]" % (epoch + 1)):
                betch_data = betch_data.to(self.device)
                label = betch_data['target'].unique_item_score
                pred: Tensor = self.net(betch_data)
                loss: Tensor = self.loss_function(pred, label)
                # ------end training--------------------
                loss_list.append(loss.item())

                # ------start update parameters----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # ------ end update parameters-----------
            print('\t {} epoch loss = {}'.format(epoch + 1, np.mean(loss_list)))
            if test_data is not None:
                self.test(test_data)

    def test(self, data_loader):
        test_pred_list, test_label_list = [], []
        loss_list = []
        # data_list = [x.to(self.device) for x in data_loader]
        for betch_data in tqdm(data_loader, "[Testing:]"):
            betch_data = betch_data.to(self.device)
            with torch.no_grad():
                label = betch_data['target'].unique_item_score
                pred: Tensor = self.net(betch_data)
                loss: Tensor = self.loss_function(pred, label)
            loss_list.append(loss.item())
            test_pred_list.extend(pred.tolist())
            test_label_list.extend(label.tolist())
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f,loss:%.6f" % (acc, auc, rmse, mae, np.mean(loss_list)))
        return acc, auc, rmse, mae

    def get_cognitive_state(self, stu_num, data_loader):
        A = torch.empty((stu_num, self.conc_num)).to('cpu')
        for betch_data in tqdm(data_loader, "[get_cognitive_state:]"):
            betch_data = betch_data.to(self.device)
            stu_list = betch_data['source'].unique_stu_index
            with torch.no_grad():
                pred: Tensor = self.net.get_cognitive_state(betch_data).to('cpu').detach()
            A[stu_list] = pred
        return A.to('cpu').detach()
