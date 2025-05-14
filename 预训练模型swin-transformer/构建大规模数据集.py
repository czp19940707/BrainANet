import os
import re
import pandas as pd

if __name__ == '__main__':
    dataset_list = ['AOMIC-ID1000', 'AOMIC-PIOP1', 'AOMIC-PIOP2', 'HCP', 'ENKI', 'IXI', 'SLIM', 'DLBS', 'MR-ART', 'SALD', 'CAB']
    path = r'/media/shucheng/数程SSD_2T'
    frame = pd.DataFrame()
    count_ = 0
    for dataset in dataset_list:
        pa_ = os.path.join(path, dataset, 'derivatives', 'cat12', 'mri')

        for subject_id in os.listdir(pa_):
            frame.loc[count_, 'Phase'] = dataset
            frame.loc[count_, 'Subject ID'] = subject_id[4:-4]
            frame.loc[count_, 'Path'] = os.path.join(pa_, subject_id)

            # if os.path.exists(path_data_table):
            #     if any(re.search(i) for i in frame_datatable['Subject ID'].tolist()):
            #         print()
            # frame.loc[count_, 'Age']
            count_ += 1


    """
    在frame中添加age，sex
    """
    for dataset in dataset_list:
        participants_table = os.path.join(path, dataset, 'participants.csv')
        frame_sub = frame[frame['Phase'] == dataset]
        if os.path.exists(participants_table):
            frame_datatable = pd.read_csv(participants_table)
            subjects = frame_datatable['Subject ID'].tolist()
            for index_1 in frame_datatable.index:
                subject_id = frame_datatable.loc[index_1, 'Subject ID']
                for index_2 in frame_sub.index:
                    frame_sub_information = frame_sub.loc[index_2, :]
                    if re.search(str(subject_id), frame_sub_information['Subject ID']):
                        frame.loc[index_2, 'Age'] = frame_datatable.loc[index_1, 'Age']
                        frame.loc[index_2, 'Sex'] = frame_datatable.loc[index_1, 'Sex']
                        continue

    frame.to_csv('DataTable.csv', index=False)