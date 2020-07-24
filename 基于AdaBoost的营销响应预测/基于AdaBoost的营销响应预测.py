"""

OneHotEncoder：将分类变量和顺序变量转换为二值化标志变量
StratifiedKFold,cross_val_score：交叉验证，前者用来将数据分为训练集和测试集；后者用来交叉检验。
StratifiedkFold 能结合样本标签做数据集分割，而不是完全的随机选择和分割
SelectPercentile,f_classif：前者用来做特征选择的数量控制，后者用来确定特征选择的得分计算标准
AdaBoostClassifier：集成算法，用来做分类模型训练
Pipeline：将不同的环节结合起来（本案例中，将特征选择和集成算法结合起来形成一个”管道对象“，然后针对该对象
训练不同参数下交叉检验的结果）


"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder         # 导入二值化标志转化库
from sklearn.model_selection import StratifiedKFold, cross_val_score    # 导入交叉检验算法
from sklearn.feature_selection import SelectPercentile, f_classif       # 导入特征选择方法
from sklearn.ensemble import AdaBoostClassifier         # 导入集成算法
from sklearn.pipeline import Pipeline                   # 导入Pipeline库
from sklearn.metrics import accuracy_score              # 准确率指标



def set_summary(df):

    """
    查看数据
    :return:
    """

    print('Data Overview')
    print('Records: {0}\tDimension{1}'.format(df.shape[0], (df.shape[1]-1)))

    # 打印数据集 X 的形状
    print('-' * 30)
    print(df.head(2))       # 打印前两条数据
    print('-' * 30)
    print('Data DESC')
    print(df.describe())    # 打印数据描述信息
    print('Data Dtypes')
    print(df.dtypes)        # 打印数据类型
    print('-' * 60)



def na_summary(df):

    """
    查看数据的缺失
    :param df:
    :return:
    """

    na_cols = df.isnull().any(axis=0)   # 每一列是否有缺失值
    print('NA Cols:')
    print(na_cols)                      # 查看具有缺失值的列
    print('valid records for each Cols:')
    print(df.count())                   # 每一列非NA的记录数
    na_lines = df.isnull().any(axis=1)  # 每一行是否有缺失值
    print('Total number of NA lines is: {0}'.format(na_lines.sum()))    # 查看缺失值的行总记录数
    print('-' * 30)



def label_summary(df):

    """
    查看每个类的样本量分布
    :param df:
    :return:
    """
    print('Label sample count:')
    print(df['value_level'].groupby(df['respone']).count())     # 以response为分类汇总维度对value_level列计数统计
    print('-' * 60)



def type_con(df):

    """
    转换目标列的数据为特定数据类型
    :param df:
    :return:
    """

    val_list = {'edu': 'int32',
                'user_level': 'int32',
                'industry': 'int32',
                'value_level': 'int32',
                'act_level': 'int32',
                'sex': 'int32',
                'region': 'int32'}      # 字典：定义要转换的列及其数据类型(key：列名，velue：新数据类型)
    for var, type in val_list.items():
        df[var] = df[var].astype(type)
    print('Data Dtypes:')
    print(df.dtypes)
    return df



def na_replace(df):

    """
    将NA值使用自定义方法得替换
    :param df:
    :return:
    """

    na_rules = {'age': df['age'].mean(),
                'total_pageviews': df['total_pageviews'].mean(),
                'edu': df['edu'].median(),      # median()：中值
                'edu_ages': df['edu_ages'].median(),
                'user_level': df['user_level'].median(),
                'industry': df['industry'].median(),
                'act_level': df['act_level'].median(),
                'sex': df['sex'].median(),
                'red_money': df['red_money'].median(),
                'region': df['region'].median()
                }
    df = df.fillna(na_rules)        # 使用指定方法填充缺失值
    print('Check NA exists:')
    print(df.isnull().any().sum())  # 查看是否还有缺失值
    return df



def symbol_con(df, enc_object=None, train=True):

    """
    将分类和顺序变量转换为二值化的标志变量
    :param df:
    :param enc_object:
    :param train:
    :return:
    """

    convert_cols = ['edu', 'user_level', 'industry', 'value_level', 'act_level', 'sex', 'region']       # 选择要做标志转换的列(教育程度，用户等级，用户企业划分，用户价制度划分，用户活跃度划分，性别，地区)
    df_con = df[convert_cols]       # 选择要做标志转换的数据
    df_org = df[['age', 'total_pageviews', 'edu_ages', 'blue_money', 'red_money', 'work_hours']].values   # 设置不做标志转换的列（df.column.values：以array形式返回指定column的所有取值）
    if train == True:               # 如果数据处于训练阶段
        enc = OneHotEncoder()       # 建立标志转换模型对象
        enc.fit(df_con)             # 训练模型
        df_con_new = enc.transform(df_con).toarray()    # 转换数据并输出为数组格式
        new_matrix = np.hstack(df_con_new, df_con)      # 将未转换的数据与转换后的数据合并
        return new_matrix, enc
    else:
        df_con_new = enc_object.transform(df_con).toarray()     # 使用训练阶段获得转换对象转换数据并输出数组格式
        new_matrix = np.hstack((df_con_new, df_org))    # 将未转换的数据与转换的数据合并
        return new_matrix


"""
定义需转换的列形成数据框df_con
定义无需转换的列形成数据矩阵(数据框的值，通过values获得)df_org
df_org是Numpy矩阵而非Pandas数据框，接下来使用Numpy方法做数据矩阵合并，原因是OneHotEncoder转换后的对象也是Numpy矩阵


接下来根据train的状态做判断，
为True时，用OneHotEncoder方法建立转换对象enc，然后用fit方法做训练，接着用transform方法做转换
（这里fit和transform方法分开操作是enc对象要在fit之后传给预测集使用）
最后将未转换的数据与转换后的数据合并。
为False时，直接用训练阶段获得的对象enc做transform，然后将结果合并。

"""



def get_best_model(X, y):
    """
    结合交叉检验得到不同参数下的分类模型结果
    :param X: 输入X
    :param y: 输出y
    :return: 特征选择模型对象
    """
    transform = SelectPercentile(f_classif, percentile=50)  # 使用f_classif方法选择特征最明显的50%数量的特征
    model_adaboost = AdaBoostClassifier()                   # 建立AdaBoostClassifier模型对象
    model_pipe = Pipeline(steps=[('ANOVA', transform), ('model_adaboost', model_adaboost)]) # 建立由特征选择和分类模型构成的“管道”对象
    cv = StratifiedKFold(5)             # 设置交叉验证次数
    n_estimators = [20, 50, 80, 100]    # 设置模型参数
    score_methods = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']    # 设置交叉检验指标
    mean_list = list()                  # 建立空列表用于存放不同参数方法、交叉检验评估指标的均值列表
    std_list = list()                   # 建立空列表用于存放不同参数方法、交叉检验评估指标的标准差列表
    for parameter in n_estimators:      # 循环读出每个参数值
        t1 = time.time()                # 记录训练开始的时间
        score_list = list()             # 建立空列表用于存放不同交叉检验下各个评估指标的详细数据
        print ('set parameters: %s' % parameter)    # 打印当前模型使用的参数
        for score_method in score_methods:          # 循环读出每个交叉检验指标
            model_pipe.set_params(model_adaboost__n_estimators=parameter)               # 通过“管道”设置分类模型参数
            score_tmp = cross_val_score(model_pipe, X, y, scoring=score_method, cv=cv)  # 使用交叉检验计算指定指标的得分
            score_list.append(score_tmp)            # 将交叉检验得分存储到列表
        score_matrix = pd.DataFrame(np.array(score_list), index=score_methods)          # 将交叉检验详细数据转换为矩阵
        score_mean = score_matrix.mean(axis=1).rename('mean')   # 计算每个评估指标的均值
        score_std = score_matrix.std(axis=1).rename('std')      # 计算每个评估指标的标准差
        score_pd = pd.concat([score_matrix, score_mean, score_std], axis=1)             # 将原始详细数据和均值、标准差合并
        mean_list.append(score_mean)    # 将每个参数得到的各指标均值追加到列表
        std_list.append(score_std)      # 将每个参数得到的各指标标准差追加到列表
        print (score_pd.round(2))       # 打印每个参数得到的交叉检验指标数据，只保留2位小数
        print ('-' * 60)
        t2 = time.time()    # 计算每个参数下算法用时
        tt = t2 - t1        # 计算时间间隔
        print ('time: %s' % str(tt))        # 打印时间间隔
    mean_matrix = np.array(mean_list).T     # 建立所有参数得到的交叉检验的均值矩阵
    std_matrix = np.array(std_list).T       # 建立所有参数得到的交叉检验的标准差矩阵
    mean_pd = pd.DataFrame(mean_matrix, index=score_methods, columns=n_estimators)      # 将均值矩阵转换为数据框
    std_pd = pd.DataFrame(std_matrix, index=score_methods, columns=n_estimators)        # 将均值标准差转换为数据框
    print ('Mean values for each parameter:')
    print (mean_pd)         # 打印输出均值矩阵
    print ('Std values for each parameter:')
    print (std_pd)          # 打印输出标准差矩阵
    print ('-' * 60)
    return transform





"""
在该函数的实现过程中，首先做数据降维，这里使用特征选择方法，没有使用PCA、LDA做数据转换，
因为在实际应用中考虑业务方可能对结果的特征重要性做分析，所以不使用转换方法。
"""


"""
SelectPercentile：移除指定的最高得分百分比之外的所有特征
f_classif：计算方差
"""




# 加载数据集
raw_data = pd.read_excel('order.xlsx', sheet_name=0)    # 第一个sheet
X = raw_data.drop('response', axis=1)                   # 分割X
y = raw_data['response']                                # 分割y




X_t1 = na_replace(X)        # 替换缺失值
X_t2 = type_con(X_t1)       # 数据类型转换
X_new, enc = symbol_con(X_t2, enc_object=None, train=True)      # 将分类和顺序数据转换为标志

# 分类模型训练
transform = get_best_model(X_new, y)    # 获得最佳分类模型参数信息
transform.fit(X_new, y)                 # 应用特征选择对象选择要参与建模的特征变量
X_final = transform.transform(X_new)    # 获得具有显著性特征的特征变量
final_model = AdaBoostClassifier(n_estimators=100)      # 从打印的参数均值和标准差信息中确定参数并建立分类模型对象
final_model.fit(X_final, y) # 训练模型

# 新数据集做预测
new_data = pd.read_excel('order.xlsx', sheetname=1)     # 读取要预测的数据集
final_reponse = new_data['final_response']              # 获取最终的目标变量值
new_data = new_data.drop('final_response', axis=1)      # 获得预测的输入变量X
set_summary(new_data)   # 基本状态查看
na_summary(new_data)    # 缺失值审查
new_X_t1 = na_replace(new_data)     # 替换缺失值
new_X_t2 = type_con(new_X_t1)       # 数据类型转换
new_X_t3 = symbol_con(new_X_t2, enc_object=enc, train=False)    # 将分类和顺序数据转换为标志
new_X_final = transform.transform(new_X_t3)             # 对数据集做特征选择

# 输出预测值以及预测概率
predict_labels = pd.DataFrame(final_model.predict(new_X_final), columns=['labels'])     # 获得预测标签
predict_labels_pro = pd.DataFrame(final_model.predict_proba(new_X_final), columns=['pro1', 'pro2'])     # 获得预测概率
predict_pd = pd.concat((new_data, predict_labels, predict_labels_pro), axis=1)          # 将预测标签、预测数据和原始数据X合并
print ('Predict info')
print (predict_pd.head(2))
print ('-' * 60)



# 后续--与实际效果的比较
print ('final accuracy: {0}'.format(accuracy_score(final_reponse, predict_labels)))







