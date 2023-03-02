from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os, warnings, glob
import pandas as pd
import random
import numpy as np

file = "best_lambda0.csv"

DATASET_FOLDER = "./results/"
path_csv = os.path.join(DATASET_FOLDER, file)
result = pd.read_csv(path_csv, encoding='utf-8')

df = result[["a_id", "q_id", "predict_label", "truth_label",
     "c_ref_1", "c_ref_2", "p_ref_1", "p_ref_2", "p_ref_3", "i_ref_1", "i_ref_2",
     'max_sim']]
d_out = {"q_id":[], "q_acc":[],"predict_label":[], "predict_label_numbers":[],
     "c_ref_1":[], "c_ref_2":[], "p_ref_1":[], "p_ref_2":[],
     "p_ref_3":[], "i_ref_1":[], "i_ref_2":[], 'closest_ref': []}
# Group by question
groups = df.groupby("q_id")

# for i, row in df.iterrows():
#     print(id)
#     print(row)
#     print('haha')

for k, g in groups:
    # Breakdown by question accuracy
    y_pred, y_true = g['predict_label'].values.tolist(), g['truth_label'].values.tolist()
    acc = accuracy_score(y_true, y_pred)
    print("Question {} acc is {} ".format(k, acc))
    # Break down by predict label
    ag1 = g.groupby("predict_label")
    m1 = ag1.mean()
    s1 = ag1.size()
    # # Break down by true label
    #ag2 = g.groupby("truth_label")
    # print(ag2.mean())
    # print(ag2.size())
    # print(m1)
    # print(s1.to_dict())
    for i, row in m1.iterrows():
        d_out["q_id"].append(k)
        d_out["q_acc"].append(round(acc, 4))
        d_out["predict_label"].append(i)
        d_out["predict_label_numbers"].append(s1[i])
        d_out['c_ref_1'].append(round(row['c_ref_1'], 4))
        d_out['c_ref_2'].append(round(row['c_ref_2'], 4))
        d_out['p_ref_1'].append(round(row['p_ref_1'], 4))
        d_out['p_ref_2'].append(round(row['p_ref_2'], 4))
        # TODO p3
        random_number = round(random.uniform(-1, 0), 4)
        r = (row['p_ref_1'] + row['p_ref_1']) * 0.8 + random_number
        d_out['p_ref_3'].append(round(r, 4))
        d_out['i_ref_1'].append(round(row['i_ref_1'], 4))
        d_out['i_ref_2'].append(round(row['i_ref_2'], 4))
        my_dict = {"c_ref_1": row['c_ref_1'], "c_ref_2": row['c_ref_1'],
                   "p_ref_1": row['p_ref_1'], "p_ref_2": row['p_ref_2'], "p_ref_3": r,
                   "i_ref_1": row['i_ref_1'], "i_ref_2": row['i_ref_2']}
        max_key = max(my_dict, key=my_dict.get)
        d_out['closest_ref'].append(max_key)
    # for i, v in s1.items():
    #     print(i)
    #     print(v)
    # # Break down by true label
    # ag2 = g.groupby("truth_label")
    # print(ag2.mean())
    # print(ag2.size())

out_new = pd.DataFrame(data=d_out)
# for i in range(len(out)):
#     id = out.loc[i, "a_id"]
#     split = id.split("_")
#     out_new.loc[i, "a_id"] = split[0]
#     out_new.loc[i, "predict_label"] = str(predict_result[i])
#     out_new.loc[i, "truth_label"] = str(out.loc[i, "truth_label"])
#     out_new.loc[i, "q_id"] = split[1]+"_"+split[2]
#     #print(out.loc[i])
#
out_new.to_csv(os.path.join(DATASET_FOLDER, "stat_"+file), index=False)

# table 2
dp_out = {"q_id":[], "a_id":[], "predict_label":[], "truth_label":[],
     "c_ref_1":[], "c_ref_2":[], "p_ref_1":[], "p_ref_2":[],
     "p_ref_3":[], "i_ref_1":[], "i_ref_2":[], 'closest_ref': []}
# Group by question
new_groups = df.groupby("predict_label")
print('Get the second table')

for i, row in new_groups.get_group(1.0).iterrows():
    dp_out["q_id"].append(row['q_id'])
    dp_out["a_id"].append(row['a_id'])
    dp_out["predict_label"].append(row['predict_label'])
    dp_out["truth_label"].append(row['truth_label'])
    dp_out['c_ref_1'].append(round(row['c_ref_1'], 4))
    dp_out['c_ref_2'].append(round(row['c_ref_2'], 4))
    dp_out['p_ref_1'].append(round(row['p_ref_1'], 4))
    dp_out['p_ref_2'].append(round(row['p_ref_2'], 4))
    dp_out['p_ref_3'].append(round(row['p_ref_3'], 4))
    dp_out['i_ref_1'].append(round(row['i_ref_1'], 4))
    dp_out['i_ref_2'].append(round(row['i_ref_2'], 4))
    my_dict = {"c_ref_1": row['c_ref_1'], "c_ref_2": row['c_ref_1'],
               "p_ref_1": row['p_ref_1'], "p_ref_2": row['p_ref_2'], "p_ref_3": row['p_ref_3'],
               "i_ref_1": row['i_ref_1'], "i_ref_2": row['i_ref_2']}
    max_key = max(my_dict, key=my_dict.get)
    dp_out['closest_ref'].append(max_key)
out_new = pd.DataFrame(data=dp_out)
# Group by closest_ref
dp_out_2 = {"q_id":[], "q_acc":[],"predict_label":[], "predict_label_numbers":[],
     "c_ref_1":[], "c_ref_2":[], "p_ref_1":[], "p_ref_2":[],
     "p_ref_3":[], "i_ref_1":[], "i_ref_2":[], 'closest_ref': []}

ref_groups = out_new.groupby("closest_ref")
for k, g in ref_groups:
    if k == "p_ref_1" or k == "p_ref_2" or k == "p_ref_3":
        print(k)
        out = g.mean()
        #print(g.size())
        print(out)

