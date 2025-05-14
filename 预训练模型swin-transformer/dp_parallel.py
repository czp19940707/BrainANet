import os
import numpy as np
import subprocess


def func1():
    # patch_size_list = [24, 32, 40]
    net_version = 'SwinVIT'
    # patch_size_list = [16, 24, 32, 40]
    patch_size_list = [16, 24, 32, 40]
    dim_list = [16, 32, 64]
    commands = []
    for dim in dim_list:
        for patch_size in patch_size_list:
            # os.system('python train.py -patch_size {} -b 8 -v {} -dim_hidden {}'.format(patch_size, net_version, dim))
            commands.append(
                'python train.py -patch_size {} -b 8 -v {} -dim_hidden {}'.format(patch_size, net_version, dim))
    n_ = np.ceil(len(commands) / 9000)
    data_split = np.array_split(np.array(commands), n_)
    for data_split_ in data_split:
        process = subprocess.Popen(['parallel', '-j', '2', '--gnu', ':::'] + data_split_.tolist())
        process.wait()


def func2():
    """
    推理AD所有数据

    """
    # import pandas as pd
    path_base = r'/media/shucheng/数程SSD_2T'
    patch_size_list = [24]
    dim_list = [32]
    # frame = pd.read_csv(frame_path)
    # frame = frame[frame['Phase']]
    # frame = frame[frame['Phase'] == 'AIBL']
    net = r'CNN'
    commands = []
    # Phase_list = ['MDD_ds003653', 'MDD_ds000171']
    Phase_list = ['ADNI', 'AIBL', 'NACC', 'OASIS3', 'PPMI']
    # Phase_list = ['ADNI', 'AIBL', 'NACC', 'OASIS3', 'MDD', 'MDD_ds003653', 'MDD_ds000171', 'PPMI', 'ABIDE']
    # Phase_list = ['MDD', 'MDD_ds003653', 'MDD_ds000171']
    for patch_size in patch_size_list:
        for dim in dim_list:
            for phase in Phase_list:
                path = os.path.join(path_base, phase, 'derivatives', 'cat12', 'mri')
                for file_name in os.listdir(path):
                    pa_ = os.path.join(path, file_name)
                    if file_name.endswith('.nii') and os.path.isfile(pa_):
                        save_path = f'/media/shucheng/数程SSD_2T/{phase}/derivatives'
                        os.makedirs(save_path, exist_ok=True)
                        command = f'python 模型推理.py -p {patch_size} -s {save_path} -i {pa_} -dim_hidden {dim} -v {net}'
                        commands.append(command)

            # for index_ in frame.index:
            #     # subject_information = frame.loc[index_, :]
            #     path = subject_information['Path']
            #     Phase = subject_information['Phase']
            #     subject_id = str(subject_information['Subject ID'])
            #     save_path = f'/media/shucheng/数程SSD_2T/{Phase}/derivatives'
            #     os.makedirs(save_path, exist_ok=True)
            #     # command = f'python 模型推理.py -p {patch_size} -subject_id {subject_id} -s {save_path} -i {path} -dim_hidden {dim} -v {net}'
            #     command = f'python 模型推理.py -p {patch_size} -s {save_path} -i {path} -dim_hidden {dim} -v {net}'
            #     commands.append(command)

    n_ = np.ceil(len(commands) / 9000)
    data_split = np.array_split(np.array(commands), n_)
    for data_split_ in data_split:
        process = subprocess.Popen(['parallel', '-j', '16', '--gnu', ':::'] + data_split_.tolist())
        process.wait()


def func3():
    """
    推理ICC数据

    """
    # patch_size_list = [16, 24, 32, 40]
    patch_size_list = [24]
    dim_list = [32]
    path = r'/media/shucheng/数程SSD_2T/HCP/derivatives/cat12/mri'
    # frame = frame[frame['Phase'] == 'AIBL']
    net = r'SwinVIT'
    Phase = 'HCP'
    commands = []
    for patch_size in patch_size_list:
        for dim in dim_list:
            for file_name in os.listdir(path):
                subject_id = file_name.replace('.nii', '')
                # suffix = file_name.split('_')[1]
                pa_ = os.path.join(path, file_name)
                save_path = f'/media/shucheng/数程SSD_2T/{Phase}/derivatives'
                os.makedirs(save_path, exist_ok=True)
                command = f'python 模型推理.py -p {patch_size} -s {save_path} -i {pa_} -dim_hidden {dim} -v {net}'
                commands.append(command)

    n_ = np.ceil(len(commands) / 9000)
    data_split = np.array_split(np.array(commands), n_)
    for data_split_ in data_split:
        process = subprocess.Popen(['parallel', '-j', '12', '--gnu', ':::'] + data_split_.tolist())
        process.wait()


if __name__ == '__main__':
    # func2(frame_path='DataTable_MDD.csv')
    func2()
    # func3()
