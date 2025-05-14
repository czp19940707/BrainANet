import subprocess
import os
import numpy as np


def func2():
    import random
    # random.seed(2024)
    commands = []
    # seeds = [random.randint(1, 10000) for _ in range(10)]
    path = r'/media/shucheng/数程SSD_2T/AD_multicenter2/derivatives'
    group_list = [['sMCI', 'pMCI'], ['CN', 'AD']]
    # group_list = [['sMCI', 'pMCI']]
    # group_list = [['Control', 'PD'], ['Control', 'Prodromal'], ['Prodromal', 'PD']]
    # group_list = [[0, 1]]
    methods_list = [
        'SwinVIT_D64_P24',
        'iRSSN',
        'R2SN',
        'CNN_D64_P24',
    ]
    classifier = 'SVM-L'
    save_path = 'results_AD_pearson'
    graph_name = 'pearson'
    for method in methods_list:
        method = method.replace('&', '\&')  # 转义&符号
        for group in group_list:
            # for seed in seeds:
            dp_ = os.path.join(path, method, f'{graph_name}.csv')
            command = f'python 主函数.py -data {dp_} -t {method} -g {group[0]} {group[1]} -s {save_path} -seed {2025} -classifier {classifier}'
            commands.append(command)

    process = subprocess.Popen(['parallel', '-j', '10', '--gnu', ':::'] + commands)
    process.wait()


def func6():
    commands = []
    path = r'/media/shucheng/数程SSD_2T/MDD_multicenter/derivatives'
    # group_list = [['CN', 'AD'], ['sMCI', 'pMCI']]
    # group_list = [['sMCI', 'pMCI']]
    # group_list = [['Control', 'PD'], ['Control', 'Prodromal'], ['Prodromal', 'PD']]
    # group_list = [[0, 1]]
    group_list = [[0, 1]]
    methods_list = [
        # 'SwinVIT_D64_P24',
        # 'iRSSN',
        # 'R2SN',
        # 'CNN_D32_P24',
        # 'SwinVIT_D32_P24',
        # 'R2SN.Wscore',

        'rest_fMRI_CA', 'rest_fMRI_CC', 'rest_fMRI_SCA', 'rest_fMRI_aal116', 'rest_fMRI_Dosenbach'

    ]

    # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold_list = [0.4]
    save_path = 'results_MDD'
    graph_name = 'pearson'
    for method in methods_list:
        method = method.replace('&', '\&')  # 转义&符号
        for group in group_list:
            for threshold in threshold_list:
                dp_ = os.path.join(path, method, f'{graph_name}.csv')
                command = f'python 主函数_稀疏度.py -data {dp_} -t {method} -g {group[0]} {group[1]} -s {save_path} -threshold {threshold}'
                commands.append(command)

    process = subprocess.Popen(['parallel', '-j', '12', '--gnu', ':::'] + commands)
    process.wait()


def func5():
    commands = []
    path = r'/media/shucheng/数程SSD_2T/AD_multicenter2/derivatives'
    group_list = [['CN', 'AD'], ['sMCI', 'pMCI']]
    # group_list = [['sMCI', 'pMCI']]
    # group_list = [['Control', 'PD'], ['Control', 'Prodromal'], ['Prodromal', 'PD']]
    # group_list = [[0, 1]]
    methods_list = [
        'SwinVIT_D64_P24',
        # 'iRSSN',
        # 'R2SN',
        # 'CNN_D64_P24',
    ]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    save_path = 'results_AD_pearson_mask'
    graph_name = 'pearson'
    for method in methods_list:
        method = method.replace('&', '\&')  # 转义&符号
        for group in group_list:
            for threshold in threshold_list:
                dp_ = os.path.join(path, method, f'{graph_name}.csv')
                command = f'python 主函数_阈值.py -data {dp_} -t {method} -g {group[0]} {group[1]} -s {save_path} -threshold {threshold}'
                commands.append(command)

    process = subprocess.Popen(['parallel', '-j', '4', '--gnu', ':::'] + commands)
    process.wait()


def func1():
    datasets = ['MDD']
    # datasets = ['AIBL']
    methods = [
        'iRSSN',
        'R2SN',
        # 'SwinVIT_16',
        # 'SwinVIT_24',
        # 'SwinVIT_32',
        # 'SwinVIT_40',
        # 'SwinVIT_16_finetune',
        # 'SwinVIT_24_finetune',
        # 'SwinVIT_32_finetune',
        # 'SwinVIT_40_finetune',
        # 'CNN_16',
        # 'CNN_24',
        # 'CNN_32',
        # 'CNN_40',

    ]
    path_base = r'/media/shucheng/数程SSD_2T'
    commands = []
    for dataset in datasets:
        table_ = os.path.join(path_base, dataset, 'derivatives', 'DataTable.csv')
        for method in methods:
            path_ = os.path.join(path_base, dataset, 'derivatives', method, 'data')
            s_ = os.path.join(path_base, dataset, 'derivatives', method)
            command = 'python 计算所有数据的相似性矩阵并且生成大表.py -table {} -path {} -s {} -method {}'.format(
                table_, path_, s_, method)
            commands.append(command)

    n_ = np.ceil(len(commands) / 9000)
    data_split = np.array_split(np.array(commands), n_)
    for data_split_ in data_split:
        process = subprocess.Popen(['parallel', '-j', '12', '--gnu', ':::'] + data_split_.tolist())
        process.wait()


def func3():
    """
        PD 结果展示

        """
    task1 = ['CNN_16_0_1_over_standard_Lasso_SVM-RBF',
             'CNN_24_0_1_over_standard_Lasso_SVM-RBF',
             'CNN_32_0_1_over_standard_Lasso_SVM-RBF',
             'CNN_40_0_1_over_standard_Lasso_SVM-RBF',
             ]
    f_ = open(r'show_results.txt', 'w')
    for task in task1:
        f_.write(task + '\n')
    f_.close()
    os.system(r'python 结果展示.py -t results_MDD -s "{}"'.format('CNN Patch_size Compare CN vs. MDD'))

    task2 = [i.replace('CNN', 'SwinVIT') for i in task1]
    f_ = open(r'show_results.txt', 'w')
    for task in task2:
        f_.write(task + '\n')
    f_.close()
    os.system(r'python 结果展示.py -t results_MDD -s "{}"'.format('SwinVIT Patch_size Compare CN vs. MDD'))

    task3 = [
        'R2SN_0_1_over_standard_Lasso_SVM-RBF',
        'iRSSN_0_1_over_standard_Lasso_SVM-RBF',
        'CNN_24_0_1_over_standard_Lasso_SVM-RBF',
        'SwinVIT_24_0_1_over_standard_Lasso_SVM-RBF'
    ]
    f_ = open(r'show_results.txt', 'w')
    for task in task3:
        f_.write(task + '\n')
    f_.close()
    os.system(r'python 结果展示.py -t results_MDD -s "{}"'.format('Net Compare CN vs MDD'))


def func4():
    t = 'results_AD'
    s_list = ['Net compare CN vs. AD', 'Net compare sMCI vs. pMCI']
    stage_list = ['train', 'test']
    # commands = []
    for stage in stage_list:
        for s in s_list:
            if s == 'Net compare CN vs. AD':

                task3 = [
                    'R2SN_CN_AD_over_standard_Lasso_SVM-RBF',
                    'iRSSN_CN_AD_over_standard_Lasso_SVM-RBF',
                    'CNN_D64_P24_CN_AD_over_standard_Lasso_SVM-RBF',
                    'SwinVIT_D64_P24_CN_AD_over_standard_Lasso_SVM-RBF'
                ]
                f_ = open(r'show_results.txt', 'w')
                for task in task3:
                    f_.write(task + '\n')
                f_.close()

            elif s == 'Net compare sMCI vs. pMCI':

                task3 = [
                    'R2SN_sMCI_pMCI_over_standard_Lasso_SVM-RBF',
                    'iRSSN_sMCI_pMCI_over_standard_Lasso_SVM-RBF',
                    'CNN_D64_P24_sMCI_pMCI_over_standard_Lasso_SVM-RBF',
                    'SwinVIT_D64_P24_sMCI_pMCI_over_standard_Lasso_SVM-RBF'
                ]
                f_ = open(r'show_results.txt', 'w')
                for task in task3:
                    f_.write(task + '\n')
                f_.close()

            if stage == 'train':
                command = 'python 结果展示_Phase.py -t {} -s "{}" -stage {} -Phase {}'.format(t, s, stage, 'ADNI')
                os.system(command)
                # commands.append(command)
            elif stage == 'test':
                for i in ['AIBL', 'OASIS3', 'NACC']:
                    command = 'python 结果展示_Phase.py -t {} -s "{}" -stage {} -Phase {}'.format(t, s, stage, i)
                    os.system(command)
                    # commands.append(command)

    t = 'results_MDD'
    s_list = ['Net compare CN vs. MDD']
    stage_list = ['train', 'test']
    # commands = []

    for stage in stage_list:

        for s in s_list:

            task3 = [
                'R2SN_0_1_over_standard_Lasso_SVM-RBF',
                'iRSSN_0_1_over_standard_Lasso_SVM-RBF',
                'CNN_D64_P24_0_1_over_standard_Lasso_SVM-RBF',
                'SwinVIT_D64_P24_0_1_over_standard_Lasso_SVM-RBF'
            ]
            f_ = open(r'show_results.txt', 'w')
            for task in task3:
                f_.write(task + '\n')
            f_.close()

            command = 'python 结果展示_Phase.py -t {} -s "{}" -stage {} -Phase {}'.format(t, s, stage, 'MDD')
            os.system(command)

            # commands.append(command)

    t = 'results_PD'
    s_list = ['Net compare Control vs. PD', 'Net compare Control vs. Prodromal', 'Net compare Prodromal vs. PD']
    stage_list = ['train', 'test']
    # commands = []
    for stage in stage_list:

        for s in s_list:
            if s == 'Net compare Control vs. PD':
                task3 = [
                    'R2SN_Control_PD_over_standard_Lasso_SVM-RBF',
                    'iRSSN_Control_PD_over_standard_Lasso_SVM-RBF',
                    'CNN_D64_P24_Control_PD_over_standard_Lasso_SVM-RBF',
                    'SwinVIT_D64_P24_Control_PD_over_standard_Lasso_SVM-RBF'
                ]
                f_ = open(r'show_results.txt', 'w')
                for task in task3:
                    f_.write(task + '\n')
                f_.close()
            elif s == 'Net compare Control vs. Prodromal':
                task3 = [
                    'R2SN_Control_Prodromal_over_standard_Lasso_SVM-RBF',
                    'iRSSN_Control_Prodromal_over_standard_Lasso_SVM-RBF',
                    'CNN_D64_P24_Control_Prodromal_over_standard_Lasso_SVM-RBF',
                    'SwinVIT_D64_P24_Control_Prodromal_over_standard_Lasso_SVM-RBF'
                ]
                f_ = open(r'show_results.txt', 'w')
                for task in task3:
                    f_.write(task + '\n')
                f_.close()

            elif s == 'Net compare Prodromal vs. PD':
                task3 = [
                    'R2SN_Prodromal_PD_over_standard_Lasso_SVM-RBF',
                    'iRSSN_Prodromal_PD_over_standard_Lasso_SVM-RBF',
                    'CNN_D64_P24_Prodromal_PD_over_standard_Lasso_SVM-RBF',
                    'SwinVIT_D64_P24_Prodromal_PD_over_standard_Lasso_SVM-RBF'
                ]
                f_ = open(r'show_results.txt', 'w')
                for task in task3:
                    f_.write(task + '\n')
                f_.close()

            command = 'python 结果展示_Phase.py -t {} -s "{}" -stage {} -Phase {}'.format(t, s, stage, 'PPMI')
            os.system(command)

            # commands.append(command)

    # for command in commands:
    #     os.system(command)


if __name__ == '__main__':
    func6()
    # func5()
