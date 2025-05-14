import os
import numpy as np
import subprocess


def func1():
    # fold_list = [i for i in range(5)]
    # frame_path = r'DataTable_AD3.csv'
    frame_path = r'DataTable_MDD2.csv'
    commands = []
    version_list = ['SGCN_GAT']
    # net_list = ['iRSSN', 'R2SN', 'SwinVIT', 'CNN']
    # net_list = ['R2SN', 'iRSSN']
    # net_list = ['iRSSN', 'R2SN']
    # group_list = [['Control', 'PD'], ['Control', 'Prodromal']]
    # net_list = ['R2SN', 'iRSSN', 'SwinVIT_D32_P24', 'CNN_D32_P24']

    net_list = ['SwinVIT_D32_P24+rest_fMRI_aal116']

    # group_list = [['sMCI', 'pMCI'], ['CN', 'AD']]
    # group_list = [['Prodromal', 'PD']]
    # group_list = [['CN', 'AD'], ['sMCI', 'pMCI']]
    # group_list = [['Control', 'PD'], ['Control', 'Prodromal'], ['Prodromal', 'PD']]
    # group_list = [[1, 2]]
    group_list = [[0, 1]]
    # version_list = ['GAT3']

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sparsity_method_list = ['percent']
    save_path = 'result_fc_BrainANet'
    for group in group_list:
        for version in version_list:
            for net in net_list:
                for sparsity_method in sparsity_method_list:
                    for threshold in thresholds:
                        # temp_name = version + '.' + net + '.' + str(group[0]) + '_' + str(group[1])

                        # os.system('python My_train.py -v {} -net {} -g {} {} -frame {}'.format(version, net, group[0], group[1], frame_path))
                        command = f'python train.py -v {version} -net {net} -g {group[0]} {group[1]} -frame {frame_path} -threshold {threshold} -sparsity {sparsity_method} -s {save_path}'
                        # os.system(f'python My_train.py -v {version} -net {net} -g {group[0]} {group[1]} -frame {frame_path}')
                        commands.append(command)

        n_ = np.ceil(len(commands) / 9000)
        data_split = np.array_split(np.array(commands), n_)
        for data_split_ in data_split:
            process = subprocess.Popen(['parallel', '-j', '4', '--gnu', ':::'] + data_split_.tolist())
            process.wait()


def func2():
    version_list = ['brainGB_GAT', 'brainGB_GCN', 'GAT', 'GCN', 'GIN',
                    'GlobalAttentionNet', 'GraphSAGE', 'SGCN_GCN', 'SGCN_GAT']
    task_dict = {'AD': [['CN', 'AD'], ['sMCI', 'pMCI']],
                 'ASD': [[1, 2]],
                 'MDD': [[0, 1]]}
    commands = []

    for task in list(task_dict.keys()):
        frame_path = f'DataTable_{task}.csv'

        save_path = f'results_{task}_model_comparisons'
        for group in task_dict[task]:
            for version in version_list:
                command = f'python train.py -v {version} -net SwinVIT_D32_P24 -g {group[0]} {group[1]} -frame {frame_path} -threshold {0.5} -sparsity percent -s {save_path}'
                commands.append(command)

    n_ = np.ceil(len(commands) / 9000)
    data_split = np.array_split(np.array(commands), n_)
    for data_split_ in data_split:
        process = subprocess.Popen(['parallel', '-j', '4', '--gnu', ':::'] + data_split_.tolist())
        process.wait()


if __name__ == '__main__':
    func1()
