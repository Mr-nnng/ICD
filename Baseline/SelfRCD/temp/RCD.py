# conc 知识点
# item 习题
# stu  学生

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

from rcd_dataSet import RCD_DataSet


def evaluate(pred, label):
    acc = metrics.accuracy_score(np.array(label).round(), np.array(pred).round())
    try:
        auc = metrics.roc_auc_score(np.array(label).round(), np.array(pred))
    except ValueError:
        auc = 0.5
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred)**0.5
    return acc, auc, rmse, mae


def fusion_inside_fun(self_emb, fusion_layer, atte_layer, emb):
    fusion_out = fusion_layer(emb)
    self_emb_repeat = self_emb.repeat((emb.shape[0], 1))
    atte_input = torch.cat((self_emb_repeat, emb), dim=1)
    atte_out = torch.softmax(atte_layer(atte_input), dim=0)
    result = torch.sum(atte_out * fusion_out, dim=0).reshape(1, -1)
    return result


def fusion_external_fun(self_emb, emb1, emb2, atte1_layer, atte2_layer):
    atte1_input = torch.cat((self_emb, emb1), dim=1)
    atte2_input = torch.cat((self_emb, emb2), dim=1)
    atte1_out = torch.softmax(atte1_layer(atte1_input), dim=0)
    atte2_out = torch.softmax(atte2_layer(atte2_input), dim=0)
    result = self_emb + torch.sum(atte1_out * emb1 + atte2_out * emb2, dim=0).reshape(1, -1)
    return result


class conc_or_item_Fusion(nn.Module):
    def __init__(self, emb_dim=6):
        super(conc_or_item_Fusion, self).__init__()

        self.fusion1_layer = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fusion2_layer = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attention1_inside_layer = nn.Linear(2 * emb_dim, 1)
        self.attention2_inside_layer = nn.Linear(2 * emb_dim, 1)

        self.attention1_external_layer = nn.Linear(2 * emb_dim, 1)
        self.attention2_external_layer = nn.Linear(2 * emb_dim, 1)

    def forward(self, self_emb, emb1, emb2):
        fusion1 = fusion_inside_fun(self_emb, self.fusion1_layer, self.attention1_inside_layer, emb1)
        fusion2 = fusion_inside_fun(self_emb, self.fusion2_layer, self.attention2_inside_layer, emb2)
        result = fusion_external_fun(self_emb, fusion1, fusion2, self.attention1_external_layer, self.attention2_external_layer)
        return result


class stu_Fusion(nn.Module):
    def __init__(self, emb_dim=6):
        super(stu_Fusion, self).__init__()

        self.fusion_layer = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attention_inside_layer = nn.Linear(2 * emb_dim, 1)

    def forward(self, self_emb, emb):
        fusion = fusion_inside_fun(self_emb, self.fusion_layer, self.attention_inside_layer, emb)
        result = fusion + self_emb
        return result


class Fusion(nn.Module):
    def __init__(self, emb_dim=6, device='cpu'):
        super(Fusion, self).__init__()

        self.emb_dim = emb_dim
        self.device = device

        self.concept_fusion_layer = conc_or_item_Fusion(emb_dim).to(device)
        self.item_fusion_layer = conc_or_item_Fusion(emb_dim).to(device)
        self.stu_fusion_layer = stu_Fusion(emb_dim).to(device)

    def forward(self, concepts, items, stus, concept_emb, item_emb, stu_emb,
                concept_graph, item_concept_graph, stu_item_graph):
        concepts_df, items_df, stus_df = [pd.DataFrame({'id': X, 'index': range(len(X))}).set_index('id')
                                          for X in [concepts, items, stus]]

        # concept
        concepts_fusion = torch.empty(len(concepts), self.emb_dim).to(self.device)
        for i, concept in enumerate(concepts):
            conci_emb = concept_emb[[i]]

            if concept not in concept_graph.index:
                concepts_fusion[i] = conci_emb
            else:
                conc_conc_idxs = np.array(concept_graph.loc[concept, 'concept_2'])
                conc_conc_emb = concept_emb[np.array(concepts_df.loc[conc_conc_idxs, 'index']).reshape(1, -1)]
                conc_item_idxs = np.array(set(np.unique(item_concept_graph.set_index('concept').loc[concept, 'item']))
                                          & set(np.unique(items)))
                conc_item_emb = item_emb[np.array(items_df.loc[conc_item_idxs, 'index']).reshape(1, -1)]
                conc_fusion = self.concept_fusion_layer(conci_emb, conc_conc_emb, conc_item_emb)
                concepts_fusion[i] = conc_fusion

        # item
        items_fusion = torch.empty(len(items), self.emb_dim).to(self.device)
        for i, item in enumerate(items):
            itemi_emb = item_emb[[i]]

            item_conc_idxs = np.unique(np.array(item_concept_graph.set_index('item').loc[item, 'concept']))
            item_conc_emb = concept_emb[np.array(concepts_df.loc[item_conc_idxs, 'index']).reshape(1, -1)]
            item_stu_idxs = np.unique(np.array(stu_item_graph.set_index('item_id').loc[item, ['user_id']]))
            item_stu_emb = stu_emb[np.array(stus_df.loc[item_stu_idxs, 'index']).reshape(1, -1)]
            item_fusion = self.item_fusion_layer(itemi_emb, item_conc_emb, item_stu_emb)
            items_fusion[i] = item_fusion

        # stu
        stus_fusion = torch.empty(len(stus), self.emb_dim).to(self.device)
        for i, stu in enumerate(stus):
            stui_emb = stu_emb[[i]]

            stu_item_idxs = np.unique(np.array(stu_item_graph.set_index('user_id').loc[stu, ['item_id']]))
            stu_item_emb = item_emb[np.array(items_df.loc[stu_item_idxs, 'index']).reshape(1, -1)]
            stu_fusion = self.stu_fusion_layer(stui_emb, stu_item_emb)
            stus_fusion[i] = stu_fusion

        return concepts_fusion, items_fusion, stus_fusion


class Diagnosis(nn.Module):
    def __init__(self, concept_num, emb_dim, device):
        super(Diagnosis, self).__init__()
        self.concept_num = concept_num
        self.device = device
        self.stu_factor_layer = nn.Sequential(
            nn.Linear(2 * emb_dim, concept_num, bias=False),
            nn.Sigmoid()
        )
        self.item_factor_layer = nn.Sequential(
            nn.Linear(2 * emb_dim, concept_num, bias=False),
            nn.Sigmoid()
        )
        self.predict_layer = nn.Sequential(
            nn.Linear(concept_num, 1),
            nn.Sigmoid()
        )

    def forward(self, concepts, items, stus, concept_fusion, item_fusion, stu_fusion, item_concept_graph, stu_item_graph):
        concepts_df, items_df, stus_df = [pd.DataFrame({'id': X, 'index': range(len(X))}).set_index('id')
                                          for X in [concepts, items, stus]]
        concepts_df = pd.DataFrame({'id': concepts, 'index': range(len(concepts))}).set_index('id')

        pred = torch.empty(len(stu_item_graph)).to(self.device)
        count = 0
        for stu in stus:
            stui_fusion = stu_fusion[np.array(stus_df.loc[stu, 'index']).reshape(1, -1)]
            stu_items = np.unique(np.array(stu_item_graph.set_index('user_id').loc[stu, 'item_id']))

            # debug
            len1 = len(stu_item_graph.set_index('user_id').loc[stu, 'item_id'])
            len2 = len(stu_items)
            if len1 != len2:
                print(stu)

            for stu_item in stu_items:
                itemi_fusion = item_fusion[np.array(items_df.loc[stu_item, 'index']).reshape(1, -1)]
                item_conc_idxs = np.unique(np.array(item_concept_graph.set_index('item').loc[stu_item, 'concept']))
                item_conc_fusion = concept_fusion[np.array(concepts_df.loc[item_conc_idxs, 'index']).reshape(1, -1)]

                stu_item_pred = torch.empty(len(item_conc_idxs)).to(self.device)
                for conc_idx in range(len(item_conc_idxs)):
                    item_factor_input = torch.cat((itemi_fusion, item_conc_fusion[[conc_idx]]), dim=1)
                    stu_factor_input = torch.cat((stui_fusion, item_conc_fusion[[conc_idx]]), dim=1)
                    item_factor_out = self.item_factor_layer(item_factor_input)
                    stu_factor_out = self.stu_factor_layer(stu_factor_input)
                    predict_input = stu_factor_out - item_factor_out
                    predict = self.predict_layer(predict_input).reshape(-1)
                    stu_item_pred[conc_idx] = predict
                pred[count] = stu_item_pred.mean()
                count += 1  # !!! very important
        if len(stu_item_graph) != count:
            print('error')
        if torch.isnan(pred).sum() > 0:
            print('nan')
        if pred.max() > 1 or pred.min() < 0:
            print('error')
        return pred

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.stu_factor_layer.apply(clipper)
        self.item_factor_layer.apply(clipper)
        self.predict_layer.apply(clipper)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class RCD_Net(nn.Module):
    def __init__(self, stu_num, item_num, concept_num, emb_dim=6, device='cpu'):
        super(RCD_Net, self).__init__()

        self.stu_emb = nn.Parameter(torch.randn(stu_num, emb_dim))
        self.item_emb = nn.Parameter(torch.randn(item_num, emb_dim))
        self.concept_emb = nn.Parameter(torch.randn(concept_num, emb_dim))

        self.fusion_layer1 = Fusion(emb_dim, device)
        self.fusion_layer2 = Fusion(emb_dim, device)
        self.diagnosis_layer = Diagnosis(concept_num, emb_dim, device)

    def forward(self, concept_graph, item_concept_graph, stu_item_graph):
        stus = np.unique(np.array(stu_item_graph['user_id']))
        items = np.unique(np.array(stu_item_graph['item_id']))
        # concepts = np.unique(np.array(item_concept_graph.set_index('item').loc[items, 'concept']))
        concepts = np.unique(item_concept_graph['concept'])
        concept_emb, item_emb, stu_emb = self.concept_emb[concepts - 1], self.item_emb[items - 1], self.stu_emb[stus - 1]

        concept_fusion1, item_fusion1, stu_fusion1 = self.fusion_layer1(concepts, items, stus, concept_emb, item_emb, stu_emb,
                                                                        concept_graph, item_concept_graph, stu_item_graph)
        concept_fusion2, item_fusion2, stu_fusion2 = self.fusion_layer1(concepts, items, stus, concept_fusion1, item_fusion1, stu_fusion1,
                                                                        concept_graph, item_concept_graph, stu_item_graph)
        predict = self.diagnosis_layer(concepts, items, stus, concept_fusion2, item_fusion2, stu_fusion2, item_concept_graph, stu_item_graph)
        return predict


class RCD():
    def __init__(self, stu_num, item_num, concept_num, emb_dim=6, learn_rate=1e-3, device='cpu'):
        self.stu_num = stu_num
        self.item_num = item_num
        self.concept_num = concept_num
        self.emb_dim = emb_dim
        self.device = device

        self.rcd_net = RCD_Net(stu_num, item_num, concept_num, emb_dim, device).to(device)
        self.optimizer = torch.optim.Adam(self.rcd_net.parameters(), lr=learn_rate)
        self.loss_function = torch.nn.BCELoss(reduction='mean')

    def forward(self, concept_graph, item_concept_graph, stu_item_graph):
        predict = self.rcd_net(concept_graph, item_concept_graph, stu_item_graph)
        return predict

    def fit(self, index_loader, concept_graph, item_concept_graph, train_data, epochs, test_data=None):
        for epoch in range(epochs):
            for betch_data in tqdm(index_loader, "[Epoch:%s]" % (epoch + 1)):
                stu_list = np.sort([x.numpy()for x in betch_data]).reshape(-1).astype('int')
                train_betch = train_data.loc[stu_list, :].reset_index().sort_values(by='user_id')
                stu_item_graph = train_betch[['user_id', 'item_id']].astype('int')

                # -----start training-------------------
                label = torch.FloatTensor(np.array(train_betch['score'])).to(self.device)
                pred = self.forward(concept_graph, item_concept_graph, stu_item_graph)
                loss = self.loss_function(pred, label)
                # ------end training--------------------

                # ------start update parameters----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.rcd_net.diagnosis_layer.apply_clipper()
                # ------ end update parameters-----------
            if test_data is not None:
                self.test(index_loader, concept_graph, item_concept_graph, test_data)

    def test(self, index_loader, concept_graph, item_concept_graph, test_data):
        test_pred_list, test_label_list = [], []
        for betch_data in tqdm(index_loader, "[Testing:]"):
            stu_list = np.sort([x.numpy()for x in betch_data]).reshape(-1).astype('int')
            test_betch = test_data.loc[stu_list, :].reset_index().sort_values(by='user_id')
            stu_item_graph = test_betch[['user_id', 'item_id']].astype('int')

            with torch.no_grad():
                test_pred = self.forward(concept_graph, item_concept_graph, stu_item_graph).clone().to('cpu').detach()
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(list(test_betch['score']))
        acc, auc, rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\ttest_result: \tacc:%.6f, auc:%.6f, rmse:%.6f, mae:%.6f" % (acc, auc, rmse, mae))
        return acc, auc, rmse, mae


if __name__ == '__main__':

    # ----------基本参数--------------
    datadir = 'E:/PY_Project/知识点交互CDM/'
    concept_graph_dir = 'E:/PY_Project/知识点交互CDM/temp_Baseline/SelfRCD/graph/'
    dataSet_list = ('FrcSub', 'Math1', 'Math2', 'ASSIST_0910', 'ASSIST_2017', 'MAT_2016', 'JUNYI')

    dataSet_idx = 4
    test_ratio = 0.8
    batch_size = 64
    learn_rate = 3e-2
    emb_dim = 16

    data_set_name = dataSet_list[dataSet_idx]
    epochs = 8
    device = 'cuda'
    # ----------基本参数--------------

    dataSet = RCD_DataSet(datadir, concept_graph_dir, data_set_name)

    total_stu_list = dataSet.dataSet.total_stu_list[:1000]
    record = dataSet.dataSet.record
    train_data, test_data = dataSet.dataSet.get_train_test(record.loc[total_stu_list, :], test_ratio=test_ratio)
    index_loader = DataLoader(TensorDataset(torch.tensor(total_stu_list).float()),
                              batch_size=batch_size, shuffle=True)

    concept_graph = dataSet.get_concept_graph()
    item_concept_graph = dataSet.get_item_concept_graph()

    stu_num = dataSet.dataSet.stu_num
    concept_num = dataSet.dataSet.concept_num
    item_num = dataSet.dataSet.prob_num

    model = RCD(stu_num, item_num, concept_num, emb_dim=emb_dim, learn_rate=learn_rate, device=device)
    model.fit(index_loader, concept_graph, item_concept_graph, train_data, epochs=epochs, test_data=test_data)
    acc, auc, rmse, mae = model.test(index_loader, concept_graph, item_concept_graph, test_data)
