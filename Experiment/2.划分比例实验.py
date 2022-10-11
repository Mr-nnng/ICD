import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append("../")
from Initial_dataSet import DataSet
from CICDM import CICDM


if __name__ == '__main__':
    result_table = pd.DataFrame(columns=['dataSet', 'test_ratio', 'acc', 'auc', 'rmse', 'mae'])
    table_index = 0

    basedir = '../'
    dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
    epochs_list = (8, 15, 3, 8)

    for dataSet_idx in [0, 1, 2, 3]:
        # ----------基本参数--------------
        learn_rate = 3e-2
        weight_decay = None

        data_set_name = dataSet_list[dataSet_idx]
        epochs = epochs_list[dataSet_idx]
        device = 'cuda'
        # ----------基本参数--------------

        n_splits = 5
        skill_num = 128
        batch_size = 64

        for test_ratio in [0.1, 0.3, 0.4, 0.5]:

            dataSet = DataSet(basedir, data_set_name)
            train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
            Q = dataSet.get_Q()
            total_stu_list = dataSet.total_stu_list

            print('第{}次实验'.format(table_index + 1))

            model = CICDM(stu_num=dataSet.stu_num,
                          Q=Q,
                          skill_num=skill_num,
                          learn_rate=learn_rate,
                          weight_decay=weight_decay,
                          n_splits=n_splits,
                          device=device)

            index_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                                      batch_size=batch_size, shuffle=True)

            model.fit(index_loader, train_data, epochs=epochs)
            acc, auc, rmse, mae = model.test(index_loader, train_data, test_data)

            result_table.loc[table_index, 'dataSet'] = data_set_name
            result_table.loc[table_index, 'test_ratio'] = test_ratio
            result_table.loc[table_index, 'acc'] = acc
            result_table.loc[table_index, 'auc'] = auc
            result_table.loc[table_index, 'rmse'] = rmse
            result_table.loc[table_index, 'mae'] = mae

            result_table.to_csv('sensitive.csv')

            table_index += 1
