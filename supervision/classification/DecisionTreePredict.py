# Author: Charse
'''
决策树:
    模型介绍： 逻辑斯蒂回归和支持向量机模型，都是在某种程度上要求被学习的数据
    特征和目标之间遵照线性假设，然而，在许多显实现场景下，这种假设是不存在的


'''

import pandas
from sklearn.model_selection import train_test_split
# 使用scikit-learn.feature_extraction 中的特征转换器
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

titanic = pandas.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#print("titanic head:", titanic.head())

#print("titanic info:", titanic.info)

X = titanic[['pclass','age','sex']]

y = titanic['survived']

#print(X.info())

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 3 columns):
pclass    1313 non-null object
age       633 non-null float64
sex       1313 non-null object
dtypes: float64(1), object(2)

通过上面的输出，我们设计如下的几个数据处理的任务
1. age 这个数据列， 只有633个，需要补全
2. sex 与pclass 两个数据列的值都是类别型， 需要转换wier数值特征， 用0/1 惊醒替代

'''

# 首先我们补全 age里的数据, 使用平均数或者 中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(), inplace=True)

# 堆补全的数据重新查看
print("X.info:", X.info())

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)
# 转换特征之后， 我们发现凡是类别型特征都单独剥离出来，独成一列特征，数值型的则保持不变

X_train = vec.fit_transform(X_train.to_dict(orient='record'))

print(vec.feature_names_)

# 同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

# 从sklearn.tree 中导入决策模型分类器
# 使用默认配置初始化决策数分类器
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测
y_predict = dtc.predict(X_test)

# 性能测评
print(classification_report(y_predict, y_test, target_names=['died', 'survived']))


## 特点分析
'''
相比于其他学习模型, 决策树在模型描述上有着巨大的优势, 决策树的推断逻辑非常直观, 具有清晰的可解释性
也方便了模型的可视化，这些特征同时也保证在使用决策树模型时，是无须考虑对数据的量化甚至标准化的，
并且，与前一节K近邻模型不同, 决策树仍然属于有参数模型、需要花费更多的时间在训练数据上
'''




