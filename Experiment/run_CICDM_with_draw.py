# 学习者作答习题时的得分不仅仅受到知识点的影响，还受到学习者的其他技能的影响
# 组织形式一：Y=(1-λ)*Y_A + λ*Y_B

# A：学习者对知识点的熟练程度，大小=N*K
# B：学习者除知识点外的其他技能，大小=N*8
# C：学习者在知识簇上的属性，大小=N*K
# H：知识点的交互，大小=K*K
# W：习题与知识点的权重矩阵，大小=J*K，其中元素sigmoid函数后再除以行/列累加和归一化
# D：习题与其他技能的权重，大小=J*8，其中元素用行/列softmax函数归一化
# lambda：其他技能对答题记录的影响权重

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from Initial_data import DataSet
from sklearn import metrics
from visdom import Visdom
from sklearn.model_selection import KFold

# ----------基本参数--------------
basedir = 'D:/python_practice/新CDM代码/'
dataSet_list = ('FrcSub', 'Math1', 'Math2', 'ASSIST_0910', 'ASSIST_2017', 'MAT_2016', 'JUNYI')
epochs_list = (15, 6, 15, 8, 15, 3, 3)

dataSet_idx = 6
test_ratio = 0.2
batch_size = 64
skill_num = 16
learn_rate = 3e-2
weight_decay = None

data_set_name = dataSet_list[dataSet_idx]
epochs = epochs_list[dataSet_idx]
device = 'cuda'
# ----------基本参数--------------

# visdom画图
viz = Visdom()
viz.line([[0.5, 0.5]], [0.], win='loss', opts=dict(
    title=data_set_name + '-loss', legend=['train_loss', 'valid_loss']))
viz.line([[0.5, 0.5, 0.5, 0.5]], [0.], win='rmse&mae', opts=dict(
    title=data_set_name + '-rmse&mae', legend=['train_rmse', 'valid_rmse', 'train_mae', 'valid_mae']))
viz.line([[0.5, 0.5, 0.5, 0.5]], [0.], win='acc&auc', opts=dict(
    title=data_set_name + '-acc&auc', legend=['train_acc', 'valid_acc', 'train_auc', 'valid_auc']))


def sigmoid(x):
    return torch.sigmoid(x)


def relu(x):
    return torch.relu(x)


def format_data(record, valid_record, n_splits=5):
    train = [[], [], []]  # 学生,习题，得分
    label = [[], [], []]  # 学生,习题，得分
    valid = [[], [], []]
    stu_list = set(record.index)

    KF = KFold(n_splits=n_splits, shuffle=True)  # 5折交叉验证
    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        if len(stu_item) >= n_splits:
            valid_item = valid_record.loc[[stu], 'item_id'].values - 1
            valid_score = valid_record.loc[[stu], 'score'].values
            for train_prob, label_prob in KF.split(stu_item):
                train[0].append(stu - 1)
                train[1].append(stu_item[train_prob])
                train[2].append(stu_score[train_prob])

                label[0].extend([count] * len(label_prob))
                label[1].extend(stu_item[label_prob])
                label[2].extend(stu_score[label_prob])

                valid[0].extend([count] * len(valid_item))
                valid[1].extend(valid_item)
                valid[2].extend(valid_score)
                count += 1
    return train, label, valid


def format_test_data(record, test_record):
    train = [[], [], []]  # 学生,习题，得分
    test = [[], [], []]  # 学生,习题，得分
    stu_list = set(record.index)

    count = 0
    for stu in stu_list:
        stu_item = record.loc[[stu], 'item_id'].values - 1
        stu_score = record.loc[[stu], 'score'].values
        test_item = test_record.loc[[stu], 'item_id'].values - 1
        test_score = test_record.loc[[stu], 'score'].values

        train[0].append(stu - 1)
        train[1].append(stu_item)
        train[2].append(stu_score)

        test[0].extend([count] * len(test_item))
        test[1].extend(test_item)
        test[2].extend(test_score)
        count += 1
    return train, test


def format_all_data(all_record):
    data = [[], [], []]  # 学生,习题，得分
    stu_list = set(all_record.index)

    count = 0
    for stu in stu_list:
        stu_item = all_record.loc[[stu], 'item_id'].values - 1
        stu_score = all_record.loc[[stu], 'score'].values

        data[0].append(stu - 1)
        data[1].append(stu_item)
        data[2].append(stu_score)

        count += 1
    return data


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


def requires_grad(param_list):
    for param in param_list:
        param.requires_grad = True


class CDM():
    def __init__(self, stu_num, Q,
                 skill_num=8, learn_rate=1e-3, weight_decay=1e-3, device='cpu'):
        item_num, concept_num = Q.shape
        self.stu_num = stu_num
        self.item_num = item_num
        self.concept_num = concept_num
        self.skill_num = skill_num
        self.wd = weight_decay
        self.device = device

        param_list = []
        H_ = torch.eye(concept_num).to(device) * np.log(9 * (concept_num - 1))
        W_ = torch.randn((item_num, concept_num)).to(device)
        D_ = torch.randn((item_num, skill_num)).to(device)

        guess_ = torch.ones((1, item_num)).to(device) * -2
        slide_ = torch.ones((1, item_num)).to(device) * -2
        lambd_ = torch.ones((1, item_num)).to(device) * -2

        param_list.extend([H_, W_, D_, guess_, slide_, lambd_])
        requires_grad(param_list)

        self.optimizer = torch.optim.Adam(param_list, lr=learn_rate)
        self.loss_function = torch.nn.BCELoss(reduction='mean')
        # self.loss_function = torch.nn.MSELoss(reduction='mean')

        self.Q = Q.bool().to(device)
        self.H_ = H_
        self.W_ = W_
        self.D_ = D_
        self.guess_ = guess_
        self.slide_ = slide_
        self.lambd_ = lambd_

    def forward(self, item_list, score_list):
        A = torch.empty(len(score_list), self.concept_num).to(self.device)
        B = torch.empty(len(score_list), self.skill_num).to(self.device)
        W_ = sigmoid(self.W_) * self.Q
        W2 = W_ / W_.sum(dim=1).reshape(-1, 1)
        D2 = torch.softmax(self.D_, dim=1)
        slide = sigmoid(self.slide_)
        guess = sigmoid(self.guess_)
        lambd = sigmoid(self.lambd_)

        for i, X_i in enumerate(score_list):
            X_i = torch.tensor(X_i).float().to(self.device).reshape(1, -1)

            # --------Knowledge concept start---------------
            W1_i_ = W_[item_list[i]]
            sum_W1_i_ = W1_i_.sum(dim=0)
            W1_i = W1_i_[:, sum_W1_i_ != 0] / sum_W1_i_[sum_W1_i_ != 0].reshape(1, -1)
            A1_i = X_i @ W1_i
            H1_i = torch.softmax(self.H_[sum_W1_i_ != 0], dim=0)
            A[i] = A1_i @ H1_i
            # --------Knowledge concept end---------------

            # --------Skill start---------------
            D1_i_ = self.D_[item_list[i]]
            D1_i = torch.softmax(D1_i_, dim=0)
            B[i] = X_i @ D1_i
            # --------Skill end-----------------
        Y_A_ = A @ W2.T
        Y_B_ = B @ D2.T
        Y_ = (1 - lambd) * Y_A_ + lambd * Y_B_
        Y_ = Y_.clamp(1e-8, 1 - 1e-8)
        Y = (1 - slide) * Y_ + guess * (1 - Y_)
        return A, Y

    def loss(self, pred, label):
        model_loss = self.loss_function(pred, label)
        if self.wd is not None:
            H_subtracter = torch.eye(self.concept_num).to(self.device)
            H = torch.softmax(self.H_, dim=0) @ torch.softmax(self.H_, dim=1).T
            parameter_loss = \
                sigmoid(self.W_[self.Q]).mean() +\
                (H - H_subtracter).abs().mean() +\
                sigmoid(self.lambd_).mean() + \
                sigmoid(self.guess_).mean() + \
                sigmoid(self.slide_).mean()
            loss = model_loss + self.wd * parameter_loss  # wd: Weight decay
        else:
            loss = model_loss
        return loss

    def fit(self, index_loader, train_data, valid_data, epochs):
        for epoch in range(epochs):
            loss_list = [[], []]
            label_list, pred_list = [[], []], [[], []]
            for betch_data in tqdm(index_loader, "[Epoch:%s]" % epoch):
                stu_list = np.array([x.numpy()for x in betch_data], dtype='int').reshape(-1)
                train, label_, valid = format_data(train_data.loc[stu_list, :],
                                                   valid_data.loc[stu_list, :])

                # -----start training-------------------
                _, all_pred = self.forward(train[1], train[2])
                pred = all_pred[label_[0], label_[1]]
                label = torch.FloatTensor(label_[2]).to(self.device)
                loss = self.loss(pred, label)
                # ------end training--------------------
                loss_list[0].append(loss.item())
                pred_list[0].extend(pred.clone().to('cpu').detach().tolist())
                label_list[0].extend(label_[2])

                # ------start update parameters----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # ------ end update parameters-----------

                # -------start valid-----------
                with torch.no_grad():
                    valid_pred = all_pred[valid[0], valid[1]].clone().to('cpu').detach()
                    valid_label = torch.FloatTensor(valid[2])
                    valid_loss = self.loss_function(valid_pred, valid_label)
                # -------end valid-----------
                    loss_list[1].append(valid_loss.item())
                    pred_list[1].extend(valid_pred.tolist())
                    label_list[1].extend(valid[2])

            # -------start evaluate and drawing-----------
            epoch_loss = np.mean(loss_list[0])
            epoch_valid_loss = np.mean(loss_list[1])
            acc, auc, rmse, mae = evaluate(pred_list[0], label_list[0])
            val_acc, val_auc, val_rmse, val_mae = evaluate(pred_list[1], label_list[1])

            viz.line([[epoch_loss, epoch_valid_loss]], [epoch], win='loss', update='append')
            viz.line([[rmse, val_rmse, mae, val_mae]], [epoch], win='rmse&mae', update='append')
            viz.line([[acc, val_acc, auc, val_auc]], [epoch], win='acc&auc', update='append')

            self.test(index_loader, train_data, valid_data)
            # -------end evaluate and drawing-----------

    def test(self, index_loader, train_data, test_data):
        test_pred_list, test_label_list = [], []
        for betch_data in tqdm(index_loader, "[Testing:]"):
            stu_list = np.array([x.numpy()for x in betch_data], dtype='int').reshape(-1)
            train, test = format_test_data(train_data.loc[stu_list, :],
                                           test_data.loc[stu_list, :])
            with torch.no_grad():
                _, all_pred = self.forward(train[1], train[2])
                test_pred = all_pred[test[0], test[1]].clone().to('cpu').detach()
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(test[2])
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae

    def get_A_and_Y(self, index_loader, all_record):
        A = torch.empty((self.stu_num, self.concept_num)).to(self.device)
        Y = torch.empty((self.stu_num, self.item_num)).to(self.device)
        for betch_data in tqdm(index_loader, "[get_A_and_Y:]"):
            stu_list = np.array([x.numpy()for x in betch_data], dtype='int').reshape(-1)
            data = format_all_data(all_record.loc[stu_list, :])
            with torch.no_grad():
                cogn_state, all_pred = self.forward(data[1], data[2])
                A[stu_list - 1] = cogn_state
                Y[stu_list - 1] = all_pred
        return A.clone().to('cpu').detach(), Y.clone().to('cpu').detach()


def save_param(save_dir, name, param):
    np.savetxt(save_dir + name, param.cpu().detach().numpy(), fmt='%.2f', delimiter=',')


if __name__ == '__main__':
    dataSet = DataSet(basedir, data_set_name)
    train_data, test_data = dataSet.get_train_test(test_ratio=test_ratio)
    Q = dataSet.get_Q()

    total_stu_list = dataSet.total_stu_list

    model = CDM(stu_num=dataSet.stu_num,
                Q=Q,
                skill_num=skill_num,
                learn_rate=learn_rate,
                weight_decay=weight_decay,
                device=device)

    index_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                              batch_size=batch_size, shuffle=True)

    model.fit(index_loader, train_data, test_data, epochs=epochs)
    acc, auc, rmse, mae = model.test(index_loader, train_data, test_data)
    cognitive_state, score_pred = model.get_A_and_Y(index_loader,
                                                    pd.concat([train_data, test_data]))

    # 存储参数
    save_param_dir = dataSet.save_parameter_dir
    # save_param(save_param_dir, 'W.csv', model.Q * sigmoid(model.W_))
    # save_param(save_param_dir, 'guess.csv', sigmoid(model.guess_))
    # save_param(save_param_dir, 'slide.csv', sigmoid(model.slide_))
    # save_param(save_param_dir, 'H.csv', torch.softmax(model.H_, dim=0))
    # save_param(save_param_dir, 'lambda.csv', sigmoid(model.lambd_))
