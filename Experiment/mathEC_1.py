import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append("../")
from initial_dataSet import DataSet
from model import CICDM


if __name__ == '__main__':
    result_table = pd.DataFrame(columns=['dataSet', 'counter', 'test_ratio', 'batch_size', 'potential_num',
                                         'n_splits', 'acc', 'auc', 'rmse', 'mae'])
    table_index = 0

    basedir = '../'
    dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')
    epochs_list = (8, 10, 1, 2)
    counters = 5
    dataSet_idx = 3

    # ----------基本参数--------------
    learn_rate = 3e-2
    weight_decay = None

    data_set_name = dataSet_list[dataSet_idx]
    epochs = epochs_list[dataSet_idx]
    device = 'cuda'
    # ----------基本参数--------------

    for test_ratio, n_splits, potential_num, batch_size in \
        [(0.1, 2, 32, 64), (0.2, 2, 32, 64), (0.3, 2, 32, 64), (0.4, 2, 32, 64), (0.5, 2, 32, 64),
            (0.2, 3, 32, 64), (0.2, 4, 32, 64), (0.2, 5, 32, 64), (0.2, 6, 32, 64)]:

        dataSet = DataSet(basedir, data_set_name)
        train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
        exer_conc_adj = dataSet.get_exer_conc_adj()
        conc_conc_adj = dataSet.get_conc_conc_adj()
        total_stu_list = dataSet.total_stu_list

        for counter in range(counters):
            print('第{}次实验,counter={},数据集:{}'.format(table_index + 1, counter, data_set_name))

            model = CICDM(student_num=dataSet.student_num,
                          concept_num=dataSet.concept_num,
                          exercise_num=dataSet.exercise_num,
                          exer_conc_adj=exer_conc_adj,
                          conc_conc_adj=conc_conc_adj,
                          potential_num=potential_num,
                          lr=learn_rate,
                          device=device)

            index_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                                      batch_size=batch_size, shuffle=True)

            model.fit(index_loader, train_data, epochs=epochs, n_splits=n_splits)
            acc, auc, rmse, mae = model.test(index_loader, train_data, test_data)

            result_table.loc[table_index, 'dataSet'] = data_set_name
            result_table.loc[table_index, 'counter'] = counter
            result_table.loc[table_index, 'test_ratio'] = test_ratio
            result_table.loc[table_index, 'batch_size'] = batch_size
            result_table.loc[table_index, 'potential_num'] = potential_num
            result_table.loc[table_index, 'n_splits'] = n_splits
            result_table.loc[table_index, 'acc'] = acc
            result_table.loc[table_index, 'auc'] = auc
            result_table.loc[table_index, 'rmse'] = rmse
            result_table.loc[table_index, 'mae'] = mae

            result_table.to_csv('mathEC_1_sensitive.csv')

            table_index += 1
