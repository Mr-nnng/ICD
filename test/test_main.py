from pyexpat import model
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import Tensor
from sklearn.model_selection import KFold
from initial_dataSet import DataSet
import multiprocessing as mp
from torch_geometric.loader import DataLoader as PyGDataLoader
from utils import get_data, get_stu_data_list, get_extend_record, evaluate
from graph_CICDM import CDM_Net
import time

# ----------基本参数--------------
basedir = '../'
dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
epochs_list = (5, 10, 1, 2)

dataSet_idx = 0
test_ratio = 0.2
batch_size = 64
potential_num = 32
learn_rate = 3e-2
n_splits = 2

data_set_name = dataSet_list[dataSet_idx]
epochs = epochs_list[dataSet_idx]
device = 'cuda'
# ----------基本参数--------------


KF = KFold(n_splits=n_splits, shuffle=True)  # 使用交叉验证的数据划分方式


def get_batch_data_list(stu_record_list: pd.DataFrame):
    data_list = []
    for stu_record in stu_record_list:
        stu_record = stu_record.reset_index().set_index('item_id')
        item_list = stu_record.index.unique()
        for source_idx, target_idx in KF.split(item_list):

            source_df = stu_record.loc[item_list[source_idx], :]
            target_df = stu_record.loc[item_list[target_idx], :]
            data = get_data(source_df, target_df)

            data_list.append(data)
    return data_list


def get_test_data_list(df_list):
    train_df, test_df = df_list
    train_df = train_df.reset_index().set_index('item_id')
    test_df = test_df.reset_index().set_index('item_id')
    data = get_data(train_df, test_df)
    return [data]


if __name__ == '__main__':
    dataSet = DataSet(basedir, data_set_name)
    train_df, test_df = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
    item_conc = dataSet.get_item_concept_df()
    conc_conc_adj = dataSet.get_conc_conc_adj()

    extend_train_df = get_extend_record(train_df, item_conc)
    extend_test_df = get_extend_record(test_df, item_conc)
    total_stu_list = dataSet.total_stu_list
    train_df_list = [extend_train_df.loc[stu, :] for stu in tqdm(total_stu_list, '[train_df_list]')]
    test_df_list = [extend_test_df.loc[stu, :] for stu in tqdm(total_stu_list, '[test_df_list]')]

    p_out = [get_test_data_list(x) for x in tqdm(list(zip(train_df_list, test_df_list)), '[test_data_list]')]
    stu_train_data_list = get_stu_data_list(p_out)
    test_loader = PyGDataLoader(stu_train_data_list, batch_size=batch_size, shuffle=True)

    class CICDM():
        def __init__(self, item_num: int, conc_num: int, item_conc_num: int, potential_num: int = 16, learn_rate=1e-3, device='cpu'):
            self.conc_num = conc_num
            self.device = device
            self.net = CDM_Net(item_num, conc_num, item_conc_num, potential_num, conc_conc_adj).to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learn_rate)
            self.loss_function = torch.nn.BCELoss(reduction='mean')
            print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.net.parameters())))

        def fit(self, train_list, epochs, test_loader=None):
            loss_list = []
            for epoch in range(epochs):
                step = batch_size
                splited_train_list = [train_list[i:i + step] for i in range(0, len(train_list), step)]
                p = mp.Pool()
                start = time.time()
                p_out = p.map(get_batch_data_list, splited_train_list)
                end = time.time()
                print('用时:{}s'.format(end - start))
                p.close()
                p.join()

                # p_out = [get_batch_data_list(x) for x in tqdm(train_list)]
                stu_train_data_list = get_stu_data_list(p_out)
                dataloader = PyGDataLoader(stu_train_data_list, batch_size=batch_size, shuffle=True)

                for betch_data in tqdm(dataloader, "[Epoch:%s]" % (epoch + 1)):
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
                if test_loader is not None:
                    self.test(test_loader)

        def test(self, test_loader):
            test_pred_list, test_label_list = [], []
            loss_list = []

            for betch_data in tqdm(test_loader, "[Testing:]"):
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

    item_num = dataSet.exercise_num
    conc_num = dataSet.concept_num
    item_conc_num = len(item_conc)
    model = CICDM(item_num,
                  conc_num,
                  item_conc_num,
                  potential_num,
                  learn_rate=learn_rate,
                  device=device)
    model.fit(train_df_list, epochs=epochs, test_loader=test_loader)
