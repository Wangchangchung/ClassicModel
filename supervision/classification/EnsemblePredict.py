# Author: Charse
'''
集成模型(分类)
模型介绍：
    集成(Ensemble)分类模型便是综合考量多个分类器的预测结果，从而作出决策，只是这种"综合考量"
    的方式大体上分为两种
    1. 利用相同的训练数据同时搭建多个独立的分类模型，然后通过投票的方式，以少数服从多数的原则作出从最终 的分类决策
    比较具有代表性的模型为随机森林分类器(Random Forest Classifier) 即在相同训练数据上同时搭建多棵决策树(Decision Tree)

'''
import pandas
from sklearn.model_selection import train_test_split
# 使用scikit-learn.feature_extraction 中的特征转换器
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

titanic = pandas.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X = titanic[['pclass','age','sex']]

y = titanic['survived']


X['age'].fillna(X['age'].mean(), inplace=True)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)
# 转换特征之后， 我们发现凡是类别型特征都单独剥离出来，独成一列特征，数值型的则保持不变

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# 同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树进行集成模型的训练以及预测分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)


## 性能测试

# 单一决策树
print("The accuracy of decision tree is", dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

# 输出随机森林
print("The accuracy of random forest classifiter is ", rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

# 梯度提升决策树在测试集上的分类准确性
print("The accuray of gradient tree boosting is ", gbc.score(X_test, y_test))
print(classification_report(gbc_y_pred, y_test))


## 特点分析
'''
集成模型可以是实战应用中最为常见的，相比于其他单一的学习模型, 集继模型可以整合多种模型
或者多次就一种类型的模型进行建模，由于模型估计参数的过程也同样受到概率的影响，具有一定的不确定性
因此继承模型虽然在训练过程中需要消耗更多的时间，但是得到的综合模型往往具有更高的表现性能和更好的稳定性
'''
