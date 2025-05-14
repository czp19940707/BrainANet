import pandas as pd
import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', default=r'DataTable_MDD2_new.csv', type=str)
    # parser.add_argument('-column_old', default=r'CNN', type=str)
    parser.add_argument('-net', default=r'SwinVIT_D32_P24', type=str)
    args = parser.parse_args()

    path_base = r'/media/shucheng/数程SSD_2T'

    count_ = 0
    frame = pd.read_csv(args.frame_path)
    frame = frame[~pd.isna(frame['Phase'])]
    for index_ in frame.index:
        subject_information = frame.loc[index_, :]
        subject_id = subject_information['Subject ID']
        Phase = subject_information['Phase']
        if Phase.startswith('ADNI'):
            Phase = 'ADNI'
        elif Phase.startswith('S'):
            Phase = 'MDD'
        else:
            Phase = Phase
        if args.net == 'iRSSN':
            pa_ = os.path.join(path_base, Phase, 'derivatives', args.net, 'data', f'mradiomics_mwp1{subject_id}.txt')
        elif args.net == 'R2SN':
            pa_ = os.path.join(path_base, Phase, 'derivatives', args.net, 'data', f'mradiomics_mwp1{subject_id}.csv')
        elif args.net.startswith('CNN_') or args.net.startswith('SwinVIT_'):
            pa_ = os.path.join(path_base, Phase, 'derivatives', args.net, 'data', f'Sem_mwp1{subject_id}.csv')
        else:
            raise ValueError(f'没有指定的网络:{args.net}')

        if os.path.exists(pa_):
            count_ += 1
            frame.loc[index_, args.net] = pa_

        # path_new = path_old.replace(args.column_old + '_D64_P24', args.column_new)

    frame.to_csv(args.frame_path[:-4] + '_new.csv', index=False)