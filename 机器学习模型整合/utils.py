import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np


def dataProc(args):
    if args.data.endswith('.xlsx'):
        frame = pd.read_excel(args.data)
    elif args.data.endswith('.csv'):
        frame = pd.read_csv(args.data)
    else:
        import sys
        print('表格文件必须以.csv或者.xlsx结尾，当前结尾{}'.format(os.path.split(args.data)[-1].split('.')[-1]))
        sys.exit()

    columns = frame.columns.tolist()
    """
    检查数据
    """
    if 'Group' not in columns:
        import sys
        print('表格内必须有Group列，用于标签！现在未找到！')
        sys.exit()

    if not all(item in frame['Group'].unique().tolist() for item in args.g):
        import sys
        group_now = ','.join([str(i) for i in frame['Group'].unique().tolist()])
        group_selected = ','.join([str(i) for i in [args.g]])
        print('存在未知分组\n现有：{}，选定{}'.format(group_now, group_selected))
        sys.exit()

    # frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    frame_task = frame[frame['Group'].isin(args.g)]
    dict_label = {i: num for num, i in enumerate(args.g)}
    column_data = [i for i in columns if i not in ['Subject ID', 'Group', 'Phase']]

    data = frame_task[column_data]
    label = frame_task['Group'].replace(dict_label)
    Id = frame_task['Subject ID']
    data.fillna(data.mean(), inplace=True)

    return train_test_split(data, label, Id, test_size=1 / args.f, random_state=args.seed)


def dataProc_multicenter(args, train_center='ADNI', test_center='NACC'):
    if args.data.endswith('.xlsx'):
        frame = pd.read_excel(args.data)
    elif args.data.endswith('.csv'):
        frame = pd.read_csv(args.data)
    else:
        import sys
        print('表格文件必须以.csv或者.xlsx结尾，当前结尾{}'.format(os.path.split(args.data)[-1].split('.')[-1]))
        sys.exit()

    columns = frame.columns.tolist()
    """
    检查数据
    """
    if 'Group' not in columns:
        import sys
        print('表格内必须有Group列，用于标签！现在未找到！')
        sys.exit()

    if not all(item in frame['Group'].unique().tolist() for item in args.g):
        import sys
        group_now = ','.join([str(i) for i in frame['Group'].unique().tolist()])
        group_selected = ','.join([str(i) for i in [args.g]])
        print('存在未知分组\n现有：{}，选定{}'.format(group_now, group_selected))
        sys.exit()

    # frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    frame_task = frame[frame['Group'].isin(args.g)]
    dict_label = {i: num for num, i in enumerate(args.g)}
    column_data = [i for i in columns if i not in ['Subject ID', 'Group', 'Phase', 'Stage']]

    train, test = frame_task[frame['Stage'] == train_center], frame_task[frame['Stage'] == test_center]
    X_train, X_test = train[column_data], test[column_data]
    y_train, y_test = train['Group'].replace(dict_label), test['Group'].replace(dict_label)
    id_train, id_test = train['Subject ID'], test['Subject ID']
    phase_train, phase_test = train['Phase'], test['Phase']
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    return X_train, X_test, y_train, y_test, id_train, id_test, phase_train, phase_test


def dataProc_multicenter_threshold(args, train_center='ADNI', test_center='NACC'):
    if args.data.endswith('.xlsx'):
        frame = pd.read_excel(args.data)
    elif args.data.endswith('.csv'):
        frame = pd.read_csv(args.data)
    else:
        import sys
        print('表格文件必须以.csv或者.xlsx结尾，当前结尾{}'.format(os.path.split(args.data)[-1].split('.')[-1]))
        sys.exit()

    columns = frame.columns.tolist()
    """
    检查数据
    """
    if 'Group' not in columns:
        import sys
        print('表格内必须有Group列，用于标签！现在未找到！')
        sys.exit()

    if not all(item in frame['Group'].unique().tolist() for item in args.g):
        import sys
        group_now = ','.join([str(i) for i in frame['Group'].unique().tolist()])
        group_selected = ','.join([str(i) for i in [args.g]])
        print('存在未知分组\n现有：{}，选定{}'.format(group_now, group_selected))
        sys.exit()

    # frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    frame_task = frame[frame['Group'].isin(args.g)]
    dict_label = {i: num for num, i in enumerate(args.g)}
    column_data = [i for i in columns if i not in ['Subject ID', 'Group', 'Phase', 'Stage']]

    train, test = frame_task[frame['Stage'] == train_center], frame_task[frame['Stage'] == test_center]
    X_train, X_test = train[column_data].abs().applymap(lambda x: x if x >= args.threshold else 0), test[
        column_data].abs().applymap(lambda x: x if x >= args.threshold else 0)
    y_train, y_test = train['Group'].replace(dict_label), test['Group'].replace(dict_label)
    id_train, id_test = train['Subject ID'], test['Subject ID']
    phase_train, phase_test = train['Phase'], test['Phase']
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    return X_train, X_test, y_train, y_test, id_train, id_test, phase_train, phase_test


def dataProc_multicenter_sparisty(args, train_center='ADNI', test_center='NACC'):
    if args.data.endswith('.xlsx'):
        frame = pd.read_excel(args.data)
    elif args.data.endswith('.csv'):
        frame = pd.read_csv(args.data)
    else:
        import sys
        print('表格文件必须以.csv或者.xlsx结尾，当前结尾{}'.format(os.path.split(args.data)[-1].split('.')[-1]))
        sys.exit()

    columns = frame.columns.tolist()
    """
    检查数据
    """
    if 'Group' not in columns:
        import sys
        print('表格内必须有Group列，用于标签！现在未找到！')
        sys.exit()

    if not all(item in frame['Group'].unique().tolist() for item in args.g):
        import sys
        group_now = ','.join([str(i) for i in frame['Group'].unique().tolist()])
        group_selected = ','.join([str(i) for i in [args.g]])
        print('存在未知分组\n现有：{}，选定{}'.format(group_now, group_selected))
        sys.exit()

    # frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    frame_task = frame[frame['Group'].isin(args.g)]
    dict_label = {i: num for num, i in enumerate(args.g)}
    column_data = [i for i in columns if i not in ['Subject ID', 'Group', 'Phase', 'Stage']]

    train, test = frame_task[frame['Stage'] == train_center], frame_task[frame['Stage'] == test_center]
    X_train, X_test = train[column_data].abs().apply(lambda row: keep_top_n_percent(row, args.threshold), axis=1), \
    test[column_data].abs().apply(lambda row: keep_top_n_percent(row, threshold=args.threshold), axis=1)
    y_train, y_test = train['Group'].replace(dict_label), test['Group'].replace(dict_label)
    id_train, id_test = train['Subject ID'], test['Subject ID']
    phase_train, phase_test = train['Phase'], test['Phase']
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    return X_train, X_test, y_train, y_test, id_train, id_test, phase_train, phase_test


def keep_top_n_percent(row, threshold):
    n = max(1, int(len(row) * threshold))  # 至少保留1个
    top_indices = row.nlargest(n).index  # 找到最大的前20%列
    return row.where(row.index.isin(top_indices), 0)


def dataProc_demon(args):
    from sklearn.datasets import fetch_california_housing
    california = fetch_california_housing()
    data = pd.DataFrame(california.data)
    label = pd.Series(california.target)
    Id = pd.Series(np.array([i for i in range(data.shape[0])]))
    return train_test_split(data, label, Id, test_size=1 / args.f, random_state=args.seed)


def dataProc_multicenter_reg(args):
    if args.data.endswith('.xlsx'):
        frame = pd.read_excel(args.data)
    elif args.data.endswith('.csv'):
        frame = pd.read_csv(args.data)
    else:
        import sys
        print('表格文件必须以.csv或者.xlsx结尾，当前结尾{}'.format(os.path.split(args.data)[-1].split('.')[-1]))
        sys.exit()

    columns = frame.columns.tolist()
    """
    检查数据
    """
    if 'Label' not in columns:
        import sys
        print('表格内必须有Label列，用于标签！现在未找到！')
        sys.exit()

    column_data = [i for i in columns if i not in ['Subject ID', 'Label', 'Phase']]
    data = frame[column_data]
    label = frame['Label']
    Id = frame['Subject ID']
    data.fillna(data.mean(), inplace=True)
    label.fillna(label.mean(), inplace=True)
    return train_test_split(data, label, Id, test_size=1 / args.f, random_state=args.seed)


def load_scaler(args):
    if args.scaler == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        sampler = MinMaxScaler()
    elif args.scaler == 'standard':
        from sklearn.preprocessing import StandardScaler
        sampler = StandardScaler()
    elif args.scaler == '':
        from sklearn.preprocessing import FunctionTransformer
        sampler = FunctionTransformer(lambda x: x)
    else:
        import sys
        print('标准化方法请从standard和minmax中选择！')
        sys.exit()
    return sampler


def load_selector(args):
    from sklearn.feature_selection import SelectFromModel
    if args.selector == 'Lasso':
        from sklearn.linear_model import Lasso
        toolbox = SelectFromModel(Lasso(alpha=0.01))
    elif args.selector == 'Ridge':
        from sklearn.linear_model import Ridge
        toolbox = SelectFromModel(Ridge(alpha=1.0))
    elif args.selector == 'Elastic':
        from sklearn.linear_model import ElasticNet
        toolbox = SelectFromModel(ElasticNet(alpha=0.01, l1_ratio=0.5))
    elif args.selector == 'Tree':
        from sklearn.ensemble import RandomForestClassifier
        toolbox = SelectFromModel(RandomForestClassifier(n_estimators=100))
    elif args.selector == 'Variance':
        from sklearn.feature_selection import VarianceThreshold
        toolbox = VarianceThreshold(threshold=0.1)
    elif args.selector == 'KBest':
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif  # 卡方检验
        toolbox = SelectKBest(f_classif, k=20)
    elif args.selector == 'Percentile':
        from sklearn.feature_selection import SelectPercentile
        from sklearn.feature_selection import f_classif  # ANOVA F-value
        toolbox = SelectPercentile(f_classif, percentile=20)
    elif args.selector == 'Generic':
        from sklearn.feature_selection import GenericUnivariateSelect
        from sklearn.feature_selection import f_classif
        toolbox = GenericUnivariateSelect(score_func=f_classif, mode='percentile', param=20)
    elif args.selector == 'RFE':
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        toolbox = RFE(LogisticRegression(), n_features_to_select=10, step=1)
    elif args.selector == 'RFECV':
        from sklearn.feature_selection import RFECV
        from sklearn.ensemble import RandomForestClassifier
        toolbox = RFECV(RandomForestClassifier(), step=1, cv=args.f)
    else:
        import sys
        sys_ = ','.join(
            ['Lasso', 'Ridge', 'Elastic', 'Tree', 'Variance', 'KBest', 'Percentile', 'Generic', 'RFE', 'RFECV'])
        print('请在{}中选择特征选择工具！'.format(sys_))
        sys.exit()
    return toolbox


def load_classifier(args):
    if args.classifier == 'Reg':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif args.classifier == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif args.classifier == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3)
    elif args.classifier == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif args.classifier == 'SVM-L':
        from sklearn.svm import SVC
        model = SVC(kernel='linear', max_iter=10000, probability=True)
    elif args.classifier == 'SVM-RBF':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', max_iter=10000, probability=True)
    elif args.classifier == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif args.classifier == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
    elif args.classifier == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
    elif args.classifier == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)
    elif args.classifier == 'QDA':
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis()
    elif args.classifier == 'Xgboost':
        import xgboost as xgb
        model = xgb.XGBClassifier(random_state=args.seed)
    elif args.classifier == 'Catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_state=args.seed, silent=True)
    else:
        import sys
        sys_ = ','.join(
            ['Reg', 'LDA', 'KNN', 'NaiveBayes', 'SVM-L', 'SVM-RBF', 'DecisionTree', 'RandomForest', 'GradientBoosting',
             'MLP', 'QDA', 'Xgboost', 'Catboost'])
        print('请在{}中选择分类器！'.format(sys_))
        sys.exit()
    return model


def load_regressor(args):
    if args.regressor == 'Linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif args.regressor == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()  # 线性判别分析可用于回归
    elif args.regressor == 'KNN':
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=3)
    elif args.regressor == 'SVR-L':
        from sklearn.svm import SVR
        model = SVR(kernel='linear')
    elif args.regressor == 'SVR-RBF':
        from sklearn.svm import SVR
        model = SVR(kernel='rbf')
    elif args.regressor == 'DecisionTree':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
    elif args.regressor == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100)
    elif args.regressor == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor()
    elif args.regressor == 'MLP':
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(128,), max_iter=300)
    elif args.regressor == 'Xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(random_state=args.seed)
    elif args.regressor == 'Catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_state=args.seed, silent=True)
    else:
        import sys
        sys_ = ','.join(
            ['Linear', 'LDA', 'KNN', 'SVR-L', 'SVR-RBF', 'DecisionTree', 'RandomForest', 'GradientBoosting',
             'MLP', 'Xgboost', 'Catboost'])
        print('请在{}中选择回归器！'.format(sys_))
        sys.exit()
    return model


def load_sampler(args):
    if args.sampler == 'under':
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(random_state=args.seed)
    elif args.sampler == 'over':
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=args.seed)
    elif args.sampler == '':
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(lambda x: x)
    else:
        import sys
        sys_ = ','.join(
            ['under', 'over'])
        print('请在{}中选择sampler！'.format(sys_))
        sys.exit()


def save_results(save_path, y_train, pred_train, prob_train, y_test, pred_test, prob_test, id_train, id_test,
                 loss_train, loss_test, phase_train, phase_test):
    import pandas as pd
    y_train_frame = pd.DataFrame(y_train, columns=['Group']).reset_index(drop=True)
    pred_train_frame = pd.DataFrame(pred_train, columns=['Pred']).reset_index(drop=True)
    prob_train_frame = pd.DataFrame()
    for n in range(prob_train.shape[-1]):
        prob_train_frame[f'Prob_{n}'] = prob_train[:, n]
    prob_train_frame = prob_train_frame.reset_index(drop=True)
    id_train_frame = pd.DataFrame(id_train, columns=['Subject ID']).reset_index(drop=True)
    phase_train_frame = pd.DataFrame(phase_train, columns=['Phase']).reset_index(drop=True)

    loss_train_frame = pd.DataFrame(loss_train, columns=['Loss']).reset_index(drop=True)
    frame_train = pd.concat(
        [id_train_frame, phase_train_frame, y_train_frame, pred_train_frame, prob_train_frame, loss_train_frame],
        axis=1)

    y_test_frame = pd.DataFrame(y_test, columns=['Group']).reset_index(drop=True)
    pred_test_frame = pd.DataFrame(pred_test, columns=['Pred']).reset_index(drop=True)
    prob_test_frame = pd.DataFrame()
    for n in range(prob_test.shape[-1]):
        prob_test_frame[f'Prob_{n}'] = prob_test[:, n]
    prob_test_frame = prob_test_frame.reset_index(drop=True)
    id_test_frame = pd.DataFrame(id_test, columns=['Subject ID']).reset_index(drop=True)
    phase_test_frame = pd.DataFrame(phase_test, columns=['Phase']).reset_index(drop=True)
    loss_test_frame = pd.DataFrame(loss_test, columns=['Loss']).reset_index(drop=True)
    frame_test = pd.concat(
        [id_test_frame, phase_test_frame, y_test_frame, pred_test_frame, prob_test_frame, loss_test_frame], axis=1)

    frame_train.to_csv(os.path.join(save_path, 'result_train.csv'), index=False)
    frame_test.to_csv(os.path.join(save_path, 'result_test.csv'), index=False)
    # import numpy as np
    # np.save(os.path.join(save_path, 'cls_train.npy'), y_train)
    # np.save(os.path.join(save_path, 'pred_train.npy'), pred_train)
    # np.save(os.path.join(save_path, 'prob_train.npy'), prob_train)
    # np.save(os.path.join(save_path, 'cls_test.npy'), y_test)
    # np.save(os.path.join(save_path, 'pred_test.npy'), pred_test)
    # np.save(os.path.join(save_path, 'prob_test.npy'), prob_test)


def save_results_reg(save_path, y_train, pred_train, loss_train, y_test, pred_test, id_train, id_test, loss_test):
    import pandas as pd
    y_train_frame = pd.DataFrame(y_train, columns=['Label']).reset_index(drop=True)
    pred_train_frame = pd.DataFrame(pred_train, columns=['Pred']).reset_index(drop=True)
    id_train_frame = pd.DataFrame(id_train, columns=['Subject ID']).reset_index(drop=True)
    loss_train_frame = pd.DataFrame(loss_train, columns=['Loss']).reset_index(drop=True)
    frame_train = pd.concat([id_train_frame, y_train_frame, pred_train_frame, loss_train_frame], axis=1)

    y_test_frame = pd.DataFrame(y_test, columns=['Label']).reset_index(drop=True)
    pred_test_frame = pd.DataFrame(pred_test, columns=['Pred']).reset_index(drop=True)
    id_test_frame = pd.DataFrame(id_test, columns=['Subject ID']).reset_index(drop=True)
    loss_test_frame = pd.DataFrame(loss_test, columns=['Loss']).reset_index(drop=True)
    frame_test = pd.concat([id_test_frame, y_test_frame, pred_test_frame, loss_test_frame], axis=1)

    frame_train.to_csv(os.path.join(save_path, 'result_train.csv'), index=False)
    frame_test.to_csv(os.path.join(save_path, 'result_test.csv'), index=False)


def save_coefficient(save_path, name, coef):
    import pandas as pd
    frame_coef = pd.DataFrame()
    for index_, (c, n) in enumerate(zip(coef, name)):
        frame_coef.loc[index_, 'Feature Name'] = n
        frame_coef.loc[index_, 'coef'] = c
    frame_coef.to_csv(os.path.join(save_path, 'Coefficient.csv'), index=False)


feature_selection_param_grids = {
    'Lasso': {
        'feature_selection__estimator__alpha': [0.01, 0.05, 0.1, 0.5, 1]
    },
    'Ridge': {
        'feature_selection__estimator__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    },
    'Elastic': {
        'feature_selection__estimator__alpha': [0.01, 0.1, 1.0, 10.0],
        'feature_selection__estimator__l1_ratio': [0.1, 0.5, 0.9]
    },
    'Tree': {
        'feature_selection__estimator__n_estimators': [50, 100, 200],
        'feature_selection__estimator__max_depth': [3, 4, 5]
    },
    'Variance': {
        'feature_selection__threshold': [0, 0.01, 0.1, 1.0]
    },
    'KBest': {
        'feature_selection__k': [1, 2, 3, 4, 5]
    },
    'Percentile': {
        'feature_selection__percentile': [10, 20, 30, 40, 50]
    },
    'Generic': {
        'feature_selection__mode': ['percentile', 'k_best', 'fpr', 'fdr', 'fwe'],
        'feature_selection__param': [1, 2, 3, 10, 20, 30, 40, 50]
    },
    'RFE': {
        'feature_selection__n_features_to_select': [1, 2, 3, 4, 5],
        'feature_selection__step': [1, 2, 0.5]
    },
    'RFECV': {
        'feature_selection__step': [1, 2, 0.5],
        'feature_selection__cv': [2, 3, 5]
    }
}

classifier_param_grids = {
    'Reg': {
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'saga']
    },
    'LDA': {
        'classifier__solver': ['svd', 'lsqr', 'eigen']
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'NaiveBayes': {
        'classifier__var_smoothing': np.logspace(-9, 0, 10)
    },
    'SVM-L': {
        'classifier__C': [0.1, 1.0, 5.0, 10.0]
    },
    'SVM-RBF': {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__gamma': ['scale', 'auto']
    },
    'DecisionTree': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'MLP': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__activation': ['tanh', 'relu'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive']
    },
    'QDA': {
        'classifier__reg_param': [0.0, 0.1, 0.5, 1.0]
    },
    'Xgboost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 4, 5],
        'classifier__subsample': [0.7, 0.8, 0.9]
    },
    'Catboost': {
        'classifier__iterations': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__depth': [3, 4, 5]
    }
}


def grid_search_func(args, pipeline, X, y, cv, save_path):
    from sklearn.model_selection import GridSearchCV
    param_grid = {**feature_selection_param_grids[args.selector], **classifier_param_grids[args.classifier]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)
    frame_best_params = pd.DataFrame()
    for index_, param in enumerate(grid_search.best_params_.items()):
        frame_best_params.loc[index_, 'ParamName'] = param[0]
        frame_best_params.loc[index_, 'ParamValue'] = param[1]
    frame_best_params.to_csv(os.path.join(save_path, 'grid_search_best_params.csv'), index=False)
    return grid_search.best_estimator_


def nll_loss_no_reduction(log_softmax_output, targets):
    # log_softmax_output: 形状 (batch_size, num_classes)
    # targets: 形状 (batch_size,) 每个元素是 [0, num_classes-1] 之间的整数，表示类别

    # 获取每个样本的对应的 log-softmax 值
    log_probs = log_softmax_output[np.arange(log_softmax_output.shape[0]), targets.astype(int)]
    # 计算每个样本的 NLL Loss (负对数似然损失)
    loss_per_sample = -log_probs
    return loss_per_sample


def use_loss_no_reduction(predictions, targets):
    loss_per_sample = (targets - predictions) ** 2
    return loss_per_sample


def save_names(save_path, name):
    import pandas as pd
    frame_name = pd.DataFrame()
    for index_, n in enumerate(name):
        frame_name.loc[index_, 'Feature Name'] = n
    frame_name.to_csv(os.path.join(save_path, 'Coefficient.csv'), index=False)
