import logging
from MIRT import MIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

import sys
sys.path.append("../..")
from initial_dataSet import DataSet

# ----------基本参数--------------
basedir = '../../'
dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC', 'KDDCUP')

dataSet_idx = 4
test_ratio = 0.2
batch_size = 128

data_set_name = dataSet_list[dataSet_idx]
epochs = 8
device = 'cuda'
# ----------基本参数--------------

# ------------数据集--------------
dataSet = DataSet(basedir, data_set_name)
train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
Q = dataSet.get_exer_conc_adj()
user_n = dataSet.student_num
item_n = dataSet.exercise_num
knowledge_n = dataSet.concept_num
# ------------数据集--------------


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64) - 1,
        torch.tensor(y, dtype=torch.int64) - 1,
        torch.tensor(z, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data.reset_index(), test_data.reset_index()]
]

logging.getLogger().setLevel(logging.INFO)
cdm = MIRT(user_n, item_n, knowledge_n)

cdm.train(train, epoch=epochs, device=device)
acc, auc, rmse, mae = cdm.eval(test)
print("acc: %.6f,auc: %.6f,rmse: %.6f, mae: %.6f" % (acc, auc, rmse, mae))

# all_stu_idx = dataSet.total_stu_list - 1
