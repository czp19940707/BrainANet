from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore
from torch_geometric.data import Data


class dataProc(Dataset):
    def __init__(self, stage='train', group=['CN', 'AD'], frame_path=r'ADNI_NACC_AIBL_new.csv', net='iRSSN'):
        self.stage = stage
        self.net = net
        self.frame = pd.read_csv(frame_path)
        self.check_table()
        frame = self.frame.copy()
        frame = frame.loc[(~pd.isna(frame[net])) & frame['Group'].isin(group)]

        if stage == 'train':
            self.index = frame[frame['Stage'] == 'Train'].index
        else:
            self.index = frame[frame['Stage'] == 'Test'].index

        self.dict = {
            j: i for i, j in enumerate(group)
        }

    def check_table(self):
        """
        数据管理表格中必须有
        1）Subject ID 列
        2）Group 列
        3）Stage 列，列中包括Train和Test两个属性
        4）数据存放绝对路径列
        :return:
        """
        check_list = ['Subject ID', 'Group', 'Stage', self.net]
        if not all(i in self.frame.columns.tolist() for i in check_list):
            sys_ = ','.join(check_list)
            raise ValueError('数据管理表格中需要有{}！'.format(sys_))

    def to_label(self, group_list):
        label = []
        for i in group_list:
            cls = self.dict[i]
            label.append(cls)
        return label

    def get_sample_weight(self):
        weights = []
        count_nums = np.arange(0, 2).astype(np.int64)
        count = float(self.index.shape[0])
        label = self.to_label(self.frame.loc[self.index.tolist(), 'Group'])
        count_class_list = [float(label.count(i)) for i in count_nums]
        for i in label:
            for j in count_nums:
                if i == j:
                    weights.append(count / count_class_list[j])
        imbalanced_ratio = [count_class_list[0] / i_r for i_r in count_class_list]
        return weights, imbalanced_ratio

    def __getitem__(self, item):
        subject_information = self.frame.loc[self.index[item], :]
        path = subject_information[self.net]
        data = self.load_data(path)
        label = self.dict[subject_information['Group']]
        return data, label, torch.tensor(self.index[item])

    def load_data(self, path):
        if path.endswith('.csv'):
            frame = pd.read_csv(path)
            """
            去小脑
            """
            frame = frame.loc[frame.index.tolist()[:90]]
        elif path.endswith('.txt'):
            frame = pd.DataFrame(np.loadtxt(path))
            """
            去小脑
            """
            frame = frame.loc[frame.index.tolist()[:90]]
        else:
            import sys
            sys.exit('数据必须是csv或者txt格式的！')
        """
        确保尺寸
        """
        frame = frame.T.values.astype(np.float32)
        return frame

    def __len__(self):
        return self.index.shape[0]
