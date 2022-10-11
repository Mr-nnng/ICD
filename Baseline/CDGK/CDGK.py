import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from tqdm import tqdm
from sklearn.model_selection import KFold

import sys
sys.path.append("../..")
from initial_dataSet import DataSet


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


class Net(nn.Module):
    def __init__(self, stu_num, item_num, conc_num, Q, emb_dim=32, device='cpu'):
        super(Net, self).__init__()
        self.device = device
        self.stu_num = stu_num
        self.item_num = item_num
        self.conc_num = conc_num

        self.stu_emb = nn.Parameter(torch.randn(stu_num, emb_dim))  # A
        self.item_emb = nn.Parameter(torch.randn(item_num, emb_dim))  # 难度 B
        self.disc = nn.Parameter(torch.randn(item_num, 1))  # 区分度 D
        self.Q = Q.to(device)

        self.emb2conc = nn.Sequential(
            nn.Linear(emb_dim, conc_num),
            nn.Sigmoid()
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(conc_num, 1),
            nn.Sigmoid()
        )
        self.guess_adju_layer1_stu_w = nn.Parameter(torch.randn(stu_num, 1))
        self.guess_adju_layer1_Q_w = nn.Parameter(torch.randn(conc_num, 1))

        self.guess_adju_layer2 = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, stu_list, return_cogn_state=False):
        if return_cogn_state:
            return self.emb2conc(self.stu_emb[np.array(stu_list) - 1])

        item_emb = self.emb2conc(self.item_emb)
        disc = self.disc
        Q = self.Q

        Q_adju_w = Q.float() @ self.guess_adju_layer1_Q_w

        pred = torch.empty(len(stu_list), self.item_num).to(self.device)
        for i, stu in enumerate(stu_list - 1):
            stui_emb = self.emb2conc(self.stu_emb[stu])
            X = Q * (stui_emb - item_emb) * disc
            P = self.pred_layer(X)
            stu_i_adju_w = self.guess_adju_layer1_stu_w[stu]

            guess1_out = torch.sigmoid(stu_i_adju_w + Q_adju_w)
            guess2_out = self.guess_adju_layer2(guess1_out)
            Y = guess2_out + (1 - guess2_out) * P
            pred[i] = Y.reshape(-1)
        return pred

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.pred_layer.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = torch.clamp(w, min=0.).detach()


def format_test_data(stu_list, record):
    test = [[], [], []]  # 学生,习题，得分
    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        test[0].extend([count] * len(stu_item))
        test[1].extend(stu_item)
        test[2].extend(stu_score)

        count += 1
    return test


def format_train_data(stu_list, record):
    train_1 = [[], [], []]  # 学生,习题，得分
    train_2 = [[], [], []]  # 学生,习题，得分
    train_3 = [[], [], []]  # 学生,习题，得分
    train_4 = [[], [], []]  # 学生,习题，得分
    train_5 = [[], [], []]  # 学生,习题，得分
    train = [train_1, train_2, train_3, train_4, train_5]
    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        KF = KFold(n_splits=5, shuffle=True)  # 5折交叉验证

        i = 0
        for train_idx, valid_idx in KF.split(stu_item):
            item = stu_item[valid_idx]
            score = stu_score[valid_idx]
            train[i][0].extend([count] * len(item))
            train[i][1].extend(item)
            train[i][2].extend(score)
            i += 1
        count += 1
    return train


class CDGK():
    def __init__(self, stu_num, item_num, conc_num, Q, learn_rate=1e-3, device='cpu'):
        self.device = device
        self.stu_num = stu_num
        self.item_num = item_num
        self.conc_num = conc_num

        self.net = Net(stu_num, item_num, conc_num, Q, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learn_rate)
        self.loss_function = torch.nn.BCELoss(reduction='mean')

    def get_cognitive_state(self):
        return self.net.stu_emb.clone().to('cpu').detach()

    def fit(self, index_loader, train_data, epochs, test_data=None):
        for epoch in range(epochs):
            for betch_data in tqdm(index_loader, "[Epoch:%s]" % (epoch + 1)):
                stu_list = np.array([x.numpy()for x in betch_data], dtype='int').reshape(-1)
                train = format_train_data(stu_list, train_data.loc[stu_list, :])
                for train_i in train:

                    # -----start training-------------------
                    label = torch.FloatTensor(train_i[2]).to(self.device)
                    pred = self.net(stu_list)[train_i[0], train_i[1]]
                    loss = self.loss_function(pred, label)
                    # ------end training--------------------

                    # ------start update parameters----------
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.net.apply_clipper()
                    # ------ end update parameters-----------
            if test_data is not None:
                self.test(index_loader, test_data)

    def test(self, index_loader, test_data):
        test_pred_list, test_label_list = [], []
        for betch_data in tqdm(index_loader, "[Testing:]"):
            stu_list = np.array([x.numpy()for x in betch_data], dtype='int').reshape(-1)
            test = format_test_data(stu_list, test_data.loc[stu_list, :])
            with torch.no_grad():
                all_pred = self.net(stu_list)
                test_pred = all_pred[test[0], test[1]].clone().to('cpu').detach()
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(test[2])
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae

    def get_cogn_state(self, index_loader):
        A = torch.empty((self.stu_num, self.conc_num))
        for betch_data in tqdm(index_loader, "[get_cogn_state]"):
            stu_list = np.array([x.numpy()for x in betch_data], dtype='int').reshape(-1)
            with torch.no_grad():
                batch_cogn_state = self.net(stu_list, return_cogn_state=True)
            A[stu_list - 1] = batch_cogn_state.cpu().detach()
        return A


if __name__ == '__main__':

    # ----------基本参数--------------
    basedir = '../../'
    dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
    epochs_list = (8, 8, 3, 5)
    save_list = ('a0910/', 'a2017/', 'junyi/', 'math_ec/')

    dataSet_idx = 0
    test_ratio = 0.2
    batch_size = 32
    learn_rate = 1e-2

    data_set_name = dataSet_list[dataSet_idx]
    epochs = epochs_list[dataSet_idx]
    device = 'cuda'
    # ----------基本参数--------------

    dataSet = DataSet(basedir, data_set_name, build=True)
    train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
    Q = dataSet.get_exer_conc_adj()

    total_stu_list = dataSet.total_stu_list
    index_loader = DataLoader(TensorDataset(torch.tensor(total_stu_list).float()),
                              batch_size=batch_size, shuffle=True)
    model = CDGK(dataSet.student_num, dataSet.exercise_num, dataSet.concept_num, Q, learn_rate, device=device)
    model.fit(index_loader, train_data, epochs, test_data)
    cognitive_state = model.get_cogn_state(index_loader)

    def save_param(save_dir, name, param):
        np.savetxt(save_dir + name, param.cpu().detach().numpy(), fmt='%.4f', delimiter=',')

    save_param('./output/' + save_list[dataSet_idx], 'cognitive_state.csv', cognitive_state)
