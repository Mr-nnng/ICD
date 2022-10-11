import logging
from DINA import DINA
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
save_list = ('a0910/', 'a2017/', 'junyi/', 'math_ec/', 'kddcup')
# for dataSet_idx in [2, 3]:
dataSet_idx = 0
test_ratio = 0.2
batch_size = 32

data_set_name = dataSet_list[dataSet_idx]
epochs = 8
# ----------基本参数--------------

# ------------数据集--------------
dataSet = DataSet(basedir, data_set_name)
train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
item_data = dataSet.item
Q = dataSet.get_exer_conc_adj()
user_n = dataSet.student_num
item_n = dataSet.exercise_num
knowledge_n = dataSet.concept_num
device = 'cuda'
# ------------数据集--------------


def code2vector(x):
    vector = [0] * knowledge_n
    for k in eval(x):
        vector[k - 1] = 1
    return vector


item_data["knowledge"] = item_data["knowledge_code"].apply(code2vector)
item_data.drop(columns=["knowledge_code"], inplace=True)

train_data = pd.merge(train_data.reset_index(), item_data.reset_index(), on="item_id")
test_data = pd.merge(test_data.reset_index(), item_data.reset_index(), on="item_id")


def transform(x, y, z, k, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64) - 1,
        torch.tensor(y, dtype=torch.int64) - 1,
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, test = [
    transform(data["user_id"], data["item_id"], data["score"], data["knowledge"], batch_size)
    for data in [train_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)
cdm = DINA(user_n, item_n, knowledge_n)
cdm.train(train, epoch=epochs, lr=0.009, device=device)

acc, auc, rmse, mae = cdm.eval(test)
print("acc: %.6f,auc: %.6f,rmse: %.6f, mae: %.6f" % (acc, auc, rmse, mae))

all_stu_idx = dataSet.total_stu_list - 1
cognitive_state = (cdm.dina_net.theta(torch.LongTensor(all_stu_idx)) >= 0)


def save_param(save_dir, name, param):
    np.savetxt(save_dir + name, param.cpu().detach().numpy(), fmt='%.0f', delimiter=',')


save_param('./output/' + save_list[dataSet_idx], 'cognitive_state.csv', cognitive_state)
