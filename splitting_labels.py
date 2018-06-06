import numpy as np
import pandas as pd

full_labels = pd.read_csv('data/mobile_labels.csv')
grouped = full_labels.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]
train_index = np.random.choice(len(grouped_list), size=110, replace=False)
test_index = np.setdiff1d(list(range(len(grouped_list))), train_index)

# print(len(grouped_list))
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])
train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)