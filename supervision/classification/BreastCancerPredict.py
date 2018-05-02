# Author: Charse
'''
线性分类器（分类）
是一种假设特征与分类结果存在线性关系的模型，这个模型通过累加每一个维度的特征
与各自权重的乘积来帮助决策
'''

import pandas
import numpy
# from sklearn.cross_validation import train_test_split   0.18的版本中cross_validation已经被废弃 所以使用下面的模块
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report   # 导入性能分析模块

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']
# 使用 pandas.read_cav 函数从互联网读取指定数据
data = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                       names=column_names)
# 将?替换为标准缺失值 空
data = data.replace(to_replace='?', value=numpy.nan)
# 丢弃带有缺失值的数据(只要有一个维度有缺失)
data = data.dropna(how='any')
# 输出data的数据量和维度
print(data.shape)


# 由于原始数据没有提供对应的测试样本用于评估模型性能, 因此需要堆带有标记的数据进行分割
# 通常情况下， 25%的数据会作为测试集，其余75%的数据用于训练

# 使用sklearn.model_selection里的train_test_split模块用于分割数据
# 随机采样25%的数据用于测试, 剩下的75%用于构建训练集合
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                    test_size=0.25, random_state=33)
#print("X_train:", X_train)
#print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

# 检查训练样本的数量和类别分布 共512条 (344条良性肿瘤数据, 168条恶性肿瘤数据)
print(y_train.value_counts())
# 检查测试样本的数量和类别分布 共171条 (100条良性肿瘤数据, 71条恶性肿瘤数据)
print(y_test.value_counts())
'''
Name: Class, Length: 171, dtype: int64
2    344
4    168
Name: Class, dtype: int64
2    100
4     71
Name: Class, dtype: int64
'''

## 接下来, 我们使用 逻辑斯蒂回归 和 随机梯度参数估计 两种方法对上述处理后的训练数据进行学习,并根据测试样本特征进行预测

# 标准化数据，保证每个维度的特征数据方差为1, 均值为0, 使得预测结果不会被某些维度过大的特征值而主导
stand = StandardScaler()

X_train = stand.fit_transform(X_train)
X_test = stand.transform(X_test)

# 初始化 逻辑斯蒂回归
lr = LogisticRegression()
# 初始化 随机梯度参数估计
sgdc = SGDClassifier()

# 调用 LogisticRegression 中的 fit 行数/模块用来训练模型参数
lr.fit(X_train, y_train)
#  使用训练好的模型lr对X_test 进行预测, 结果存储在变量lr_y_predict中
lr_y_predict = lr.predict(X_test)

print("LogisticRegression 预测lr_y_predict:", lr_y_predict)

# 调用SDGClassifier 中的fit函数/模型用来训练模型参数
sgdc.fit(X_train, y_train)
# 使用训练好的模型 sgdc 对X_test进行预测，结果存储在变量sgdc_y_predict 中
sgdc_y_predict = sgdc.predict(X_test)

print("SDGClassifier 预测sgdc_y_predict:", sgdc_y_predict)

## 性能测评
# 由于171 条测试样本拥有正确的标记, 并记录在变量 y_test中, 因此非常直观的做法是对比预测结果和原本正确标记, 计算171
# 条测试言本中, 预测正确百分比，我们把这个百分比称作准确性(Accuracy), 并且将器作为评估分类模型的一个重要性能指标

'''
恶性肿瘤： 阳性(Positive)     真阳性(True Positive)
良性肿瘤： 阴性(Negative)     真阴性(True Negative) 
良性肿瘤被诊断为恶性:         假阳性(False Positive)
恶性肿瘤被诊断为良性:         假阴性(False Negative)

准确性： Accuracy  = ( #(True Positive) + #(True Negative) ) / ( #(True Positive) + #(True Negative) + #(False Positive) + #(False Negative) )
召回率： Recall  = #(True Positive) / ( #(True Positive) + #(False Negative) )
准确率： Precision =  #(True Positive) /  ( #(True Positive) + #(False Positive) )

为了综合考量召回率  和 精确率
我们计算这两个指标的调和平均数，得到F1 指标(F1 measure)

F1 measure = 2 /( 1/Precision + 1/Recall )
之所以使用调和平均数,是因为它除了具备平均功能外，还会对那些召回率和精确率更加接近的模型给予更高的分数，而
这个也是我们所期待的，因为那些召回率和精确率差距过大的学习 模型，往往没有足够的使用价值

'''

## 对两个模型的性能 从准确性, 召回率, 精确率以及F1指标的表现进行分析

# 使用逻辑斯蒂回归模型自带的评分函数score获取模型在测试集上的准确性结果
print("Accuracy of LR Classifier:", lr.score(X_test, y_test))

# 利用calssification_reqport 模块获得 LogisticRegression 其他三个指标的结果
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

# 使用随机梯度下降模型自带的评分函数,score 获得模型在测试集上的准确性结果
print("Accuracy of  SGDC Classifier:", sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))


## 特点分析
'''
LogisticRegression 比较器SGDClassifier 在测试集上表现有更高的准确性(Accuracy) 这个是因为
Scikit-learn 中采用解析的方式精确计算 LogisticRegression 的参数, 而使用梯度法估计SGDClassifiter 
的参数

特点分析
线性分类器可以说是最为基本和常用的机器学习模型, 尽管其受限于数据特征与分类目标之间的线性假设
我们仍然可以在科学研究与工程实践中把线性分类器表现性能作为基准
这里所使用的模型包括 LogisticRegression  与 SGDCClassifier 

相比之下，前者堆参数的计算采用精确解析的方式，计算时间长，但是模型性能略; 后者采用随机梯度上升算法估计
模型参数，计算时间短，但是产出的模型性能略低

一般而言，度与训练数据规模在10万量级以上的数据，考虑到时间的耗用， 更加推荐随机梯度算法堆模型参数进行估计

'''





















