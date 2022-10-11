import sys
sys.path.append("../")
from initial_dataSet import DataSet

import torch
import numpy as np
import pandas as pd
from model import CICDM
from torch.utils.data import TensorDataset, DataLoader


def save_param(save_dir, name, param):
    np.savetxt(save_dir + name, param.cpu().detach().numpy(), fmt='%.6f', delimiter=',')


if __name__ == '__main__':

    # ----------基本参数--------------
    basedir = '../'
    dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC', 'KDDCUP')
    save_list = ('a0910/', 'a2017/', 'junyi/', 'math_ec/', 'kddcup/')
    epochs_list = (8, 10, 1, 2, 9)

    # dataSet_idx = 0
    for dataSet_idx in [0, 1, 2, 3, 4]:
        test_ratio = 0.2
        batch_size = 64
        potential_num = 32
        learn_rate = 3e-2
        n_splits = 2

        data_set_name = dataSet_list[dataSet_idx]
        epochs = epochs_list[dataSet_idx]
        device = 'cuda'
        # ----------基本参数--------------

        dataSet = DataSet(basedir, data_set_name)
        train_data, test_data = dataSet.get_train_test(dataSet.record, test_ratio=test_ratio)
        exer_conc_adj = dataSet.get_exer_conc_adj()
        conc_conc_adj = dataSet.get_conc_conc_adj()

        total_stu_list = dataSet.total_stu_list

        model = CICDM(student_num=dataSet.student_num,
                      concept_num=dataSet.concept_num,
                      exercise_num=dataSet.exercise_num,
                      exer_conc_adj=exer_conc_adj,
                      conc_conc_adj=conc_conc_adj,
                      potential_num=potential_num,
                      lr=learn_rate,
                      only_A=True,
                      device=device)

        index_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                                  batch_size=batch_size, shuffle=True)

        model.fit(index_loader, train_data, epochs=epochs, n_splits=n_splits)
        # acc, auc, rmse, mae = model.test(index_loader, train_data, test_data)
        cognitive_state, score_pred = model.get_A_and_Y(index_loader, dataSet.record)

        # 存储参数
        save_param_dir = './output/' + save_list[dataSet_idx]
        save_param(save_param_dir, 'H.csv', torch.softmax(model.cd_net.conc_conc_w, dim=0))

        save_result_dir = dataSet.save_result_dir
        save_param(save_result_dir, 'cognitive_state.csv', cognitive_state)
