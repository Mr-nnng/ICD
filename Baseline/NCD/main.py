# coding: utf-8
# 2021/4/1 @ WangFei

import logging
from NCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

import sys
sys.path.append("../..")
from initial_dataSet import DataSet

# ----------基本参数--------------
basedir = '../../'
dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC', 'KDDCUP', 'MAT_2016')
save_list = ('a0910/', 'a2017/', 'junyi/', 'math_ec/', 'kddcup/', '/mat2016')
dataSet_idx = -1
test_ratio = 0.2
batch_size = 32

data_set_name = dataSet_list[dataSet_idx]
epochs = 8
device = 'cuda'
# ----------基本参数--------------

# ------------数据集--------------
dataSet = DataSet(basedir, data_set_name, build=True)
train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
Q = dataSet.get_exer_conc_adj()
user_n = dataSet.student_num
item_n = dataSet.exercise_num
knowledge_n = dataSet.concept_num
# ------------数据集--------------


def transform(user, item, Q, score, batch_size):
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        Q[torch.tensor(item, dtype=torch.int64) - 1].float(),
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


train_set, test_set = [
    transform(data["user_id"], data["item_id"], Q, data["score"], batch_size)
    for data in [train_data.reset_index(), test_data.reset_index()]
]

logging.getLogger().setLevel(logging.INFO)
cdm = NCDM(knowledge_n, item_n, user_n)
cdm.train(train_data=train_set, epoch=epochs, device=device)

accuracy, auc, rmse, mae = cdm.eval(test_set)
print("acc: %.6f,auc: %.6f,rmse: %.6f, mae: %.6f" % (accuracy, auc, rmse, mae))

all_stu_idx = dataSet.total_stu_list - 1
cognitive_state = torch.sigmoid(cdm.ncdm_net.student_emb(torch.LongTensor(all_stu_idx)))


def save_param(save_dir, name, param):
    np.savetxt(save_dir + name, param.cpu().detach().numpy(), fmt='%.2f', delimiter=',')


save_param('./output/' + save_list[dataSet_idx], 'cognitive_state.csv', cognitive_state)
