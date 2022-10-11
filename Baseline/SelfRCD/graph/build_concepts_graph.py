# %%
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append("../../..")
from Initial_dataSet import DataSet

# %%
base_dir = '../../'
dataSet_list = ('ASSIST_0910', 'ASSIST_2017', 'JUNYI', 'MathEC')

dataSet_idx = 3
data_set_name = dataSet_list[dataSet_idx]
dataSet = DataSet(base_dir, data_set_name)

# %%
item = dataSet.item
item = item[~item.index.duplicated()]
item_list, concept_list = [], []
for idx in item.index:
    now_concept_list = eval(item.loc[idx, 'knowledge_code'])
    item_list.extend([idx] * len(now_concept_list))
    concept_list.extend(now_concept_list)
item_concept = pd.DataFrame({'item': item_list, 'concept': concept_list}).astype('int')

# %%


def get_index(list_1, list_2):
    index_x, index_y = [], []
    for i in list_1:
        y = set(list_2) - set([i])
        index_x.extend([i] * len(y))
        index_y.extend(list(y))
    return index_x, index_y


# %%
list_a = np.array([1, 2, 4])
list_b = np.array([2, 5, 7])
get_index(list_a, list_b)

# %%
record = dataSet.record
conc_num = dataSet.concept_num
stu_list = dataSet.total_stu_list

knowledgeCorrect = torch.zeros((conc_num, conc_num))  # 记录知识点的相关程度
conc_2_item_correct_nums = torch.zeros(conc_num)  # 记录与知识点相关的习题答对了多少次
for stu in tqdm(stu_list):
    stu_record = record.loc[stu, ['item_id', 'score']]
    for i in range(len(stu_record) - 1):
        if stu_record.iloc[i, 1] * stu_record.iloc[i + 1, 1] == 1:
            item_id_0 = stu_record.iloc[i, 0]
            concepts_0 = item_concept.set_index('item').loc[item_id_0, ['concept']].values.reshape(-1) - 1
            item_id_1 = stu_record.iloc[i + 1, 0]
            concepts_1 = item_concept.set_index('item').loc[item_id_1, ['concept']].values.reshape(-1) - 1
            idx_0, idx_1 = get_index(concepts_0, concepts_1)
            knowledgeCorrect[idx_0, idx_1] += 1
            conc_2_item_correct_nums[idx_0] += 1
            conc_2_item_correct_nums[list(set(idx_0) - set(idx_1))] -= 1

# %%
if data_set_name == 'FrcSub':
    save_dir = 'frcsub/'
elif data_set_name == 'Math1':
    save_dir = 'math1/'
elif data_set_name == 'Math2':
    save_dir = 'math2/'
elif data_set_name == 'ASSIST_0910':
    save_dir = 'a0910/'
elif data_set_name == 'ASSIST_2017':
    save_dir = 'a2017/'
elif data_set_name == 'MAT_2016':
    save_dir = 'mat2016/'
elif data_set_name == 'JUNYI':
    save_dir = 'junyi/'
elif data_set_name == 'MathEC':
    save_dir = 'math_ec/'

# %%
conc_2_item_correct_nums[conc_2_item_correct_nums == 0] = 1e10
knowledgeDirected = knowledgeCorrect / conc_2_item_correct_nums.reshape(-1, 1)
knowledgeDirected[knowledgeCorrect <= 0] = 0

temp0 = knowledgeDirected[(1 - torch.eye(conc_num)).bool()]
min_c = temp0.min()
max_c = temp0.max()

o = torch.zeros_like(knowledgeDirected)
o = (knowledgeDirected - min_c) / (max_c - min_c)
o[knowledgeDirected <= 0] = 0

# %%
l_o = (knowledgeDirected > 0).sum()
s_o = o.sum()

avg = s_o / l_o
threshold = avg**3

# %%
relation = pd.DataFrame(o > threshold)
relation.columns = relation.columns + 1
relation.index = relation.index + 1

relationship = relation.reset_index().melt(id_vars=['index'])
relationship = relationship[relationship['value'] == True].loc[:, ['index', 'variable']]
relationship.columns = ['concept_1', 'concept_2']

# %%
relationship.to_csv(save_dir + 'concept_relationship.csv', index=False)
