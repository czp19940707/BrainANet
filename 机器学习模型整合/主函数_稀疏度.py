import argparse
from utils import dataProc_multicenter_sparisty as dataProc
from utils import load_scaler, load_selector, load_classifier, save_results, save_coefficient, load_sampler, \
    nll_loss_no_reduction, save_names
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import os


# /home/shucheng/python_files/ML_classifier/data/lh.ThickAvg.aparc.a2009s.csv
#

def parse_str_or_int(value):
    try:
        # 尝试将输入转换为整数
        return int(value)
    except ValueError:
        # 如果转换失败，则保留为字符串
        return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default=r'/media/shucheng/数程SSD_2T/ABIDE_multicenter/derivatives/CNN_D32_P24/pearson.csv',
                        type=str,
                        help='数据存放表格绝对路径(.csv或者.xlsx)，这个表格必须包含Subject ID、Group列和特征列（特征列数量>=1）SwinVIT_D64_P24')
    parser.add_argument('-f', default=5, type=int, help='交叉验证折数')
    parser.add_argument('-g', nargs='+', default=[1, 2], type=parse_str_or_int, help='分组（sMCI_pMCI），可以是整型')
    parser.add_argument('-seed', default=2024, type=int, help='随机打乱种子')
    parser.add_argument('-scaler', default='standard', type=str,
                        help='标准化方法，从standard和minmax中选，也可以选择空不进行标准化')
    parser.add_argument('-selector', default='Lasso', type=str,
                        help='特征选择工具，从[Lasso, Ridge, Elastic, Tree, Variance, KBest, Percentile, Generic, RFE, RFECV]中选择')
    parser.add_argument('-classifier', default='SVM-RBF', type=str,
                        help='分类器，从[Reg, LDA, KNN, NaiveBayes, SVM-L, SVM-RBF, DecisionTree, RandomForest, GradientBoosting, MLP, QDA, Xgboost, Catboost]中选择')
    parser.add_argument('-sampler', default='over', type=str, help='采样方法，从[under, over]中选择，也可以是空')
    parser.add_argument('-save_coef', default=True, action='store_true', help='是否保存特征选择后的系数矩阵')
    parser.add_argument('-t', default='CNN_D32_P24', type=str,
                        help='保存文件夹名称(例如：ABIDE.aseg.Volume_mm3)')  # AD_multicenter.CNN.HCP&Amsterdam_16.aal
    parser.add_argument('-grid', default=False, action='store_true', help='是否使用网格寻优')
    parser.add_argument('-alpha', default=0.01, type=float, help='alpha lasso')
    parser.add_argument('-s', default='results_ASD', type=str, help='保存文件夹')
    parser.add_argument('-threshold', default=0.5, type=float, help='网络阈值')
    args = parser.parse_args()

    save_path = os.path.join(args.s, args.t + '_' + str(args.threshold) + '_' + '_'.join([str(i) for i in
                                                                                          args.g]) + '_' + args.sampler + '_' + args.scaler + '_' + args.selector + '_' + args.classifier)

    if os.path.exists(save_path):
        import sys

        print(f'{save_path}文件已存在，请删除根目录下的{save_path}文件夹，或者重新命名-t属性，确保结果不会被覆盖')
        sys.exit()
    else:
        os.makedirs(save_path)

    """
    载入数据
    """
    X_train, X_test, y_train, y_test, id_train, id_test, phase_train, phase_test = dataProc(args, train_center='Train',
                                                                                            test_center='Test')
    """
    载入sampler
    """
    sampler = load_sampler(args)
    """
    初始化标准化工具
    """
    scaler = load_scaler(args)
    """
    初始化特征选择工具
    """
    toolbox = load_selector(args)
    """
    初始化分类器
    """
    model = load_classifier(args)

    """
    构建 Pipeline

    """
    pipeline = Pipeline([
        ('sampler', sampler),  # 采样
        ('scaler', scaler),  # 数据标准化
        ('feature_selection', toolbox),  # 特征选择
        ('classifier', model)  # 分类器
    ])

    train_cv = StratifiedKFold(n_splits=args.f, shuffle=True, random_state=args.seed)

    if args.grid:
        from utils import grid_search_func

        pipeline = grid_search_func(args, pipeline, X_train, y_train, cv=train_cv, save_path=save_path)

    # 交叉验证以评估模型
    pred_train = cross_val_predict(pipeline, X_train, y_train, cv=train_cv, method='predict')
    prob_train = cross_val_predict(pipeline, X_train, y_train, cv=train_cv, method='predict_proba')
    loss_train = nll_loss_no_reduction(prob_train, y_train)

    # pred_train = pipeline.predict(X_train)
    # prob_train = pipeline.predict_proba(X_train)

    # 在训练数据上拟合模型
    pipeline.fit(X_train, y_train)
    # 在测试数据上进行预测
    pred_test = pipeline.predict(X_test)
    prob_test = pipeline.predict_proba(X_test)
    loss_test = nll_loss_no_reduction(prob_test, y_test)

    save_results(save_path=save_path, y_train=y_train, pred_train=pred_train, prob_train=prob_train,
                 prob_test=prob_test, pred_test=pred_test, y_test=y_test, id_train=id_train, id_test=id_test,
                 loss_train=loss_train, loss_test=loss_test, phase_train=phase_train, phase_test=phase_test
                 )

    if args.save_coef:
        import numpy as np

        name = np.array(X_train.columns.tolist())[pipeline.named_steps['feature_selection'].get_support()]
        try:
            # 尝试获取 feature_importances_
            coef = pipeline.named_steps['classifier'].feature_importances_.squeeze(axis=0)
            save_coefficient(save_path=save_path, name=name, coef=coef)
        except AttributeError:
            try:
                # 尝试获取 coef_
                coef = pipeline.named_steps['classifier'].coef_.squeeze(axis=0)
                save_coefficient(save_path=save_path, name=name, coef=coef)
            except AttributeError:
                import sys

                save_names(save_path=save_path, name=name)
                sys.exit("Neither feature_importances_ nor coef_ are available in the model.")
