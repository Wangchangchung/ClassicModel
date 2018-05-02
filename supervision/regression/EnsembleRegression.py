# Author: Charse
'''

集成模型(回归)

普通随机森林 和提升树模型的回归器版本
随机模型的另一种变种: 极端随机森林( Extremely Randomized Trees)
与普通的随机森林模型(Random Forests)不同的是;  极端随机森林在每当构建一棵树的分裂节点的时候,
不会任意地选取特征,而是先随机收集一部分特征,然后利用信息熵(Information Gain)
和基尼不纯性(Gini Impurity) 等指标挑选最佳的节点特征
'''
import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# 从sklearn.ensemble中导入RandomForestRegressor、ExtraTreesGressor以及GradientBoostingRegressor。
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

# 从读取房价数据村粗在变量中
boston = load_boston()

print("boston.DESCR:", boston)

X = boston.data
y = boston.target

print("boston.data:", X)
print("boston.target:", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

# 分析回归目标值的差异
print("The max target value is", numpy.max(boston.target))
print("The min target value is", numpy.min(boston.target))
print("The average target value is", numpy.mean(boston.target))

# 分别初始化对特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理。
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量rfr_y_predict中。
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 使用ExtraTreesRegressor训练模型，并对测试数据做出预测，结果存储在变量etr_y_predict中。
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中。
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)

# 使用R-squared、MSE以及MAE指标对默认配置的随机回归森林在测试集上进行性能评估。
print ('R-squared value of RandomForestRegressor:', rfr.score(X_test, y_test))
print ('The mean squared error of RandomForestRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print ('The mean absoluate error of RandomForestRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))


# 使用R-squared、MSE以及MAE指标对默认配置的极端回归森林在测试集上进行性能评估。
print ('R-squared value of ExtraTreesRegessor:', etr.score(X_test, y_test))
print ('The mean squared error of  ExtraTreesRegessor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print ('The mean absoluate error of ExtraTreesRegessor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))

# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度。
print (numpy.sort(zip(etr.feature_importances_, boston.feature_names), axis=0))


# 使用R-squared、MSE以及MAE指标对默认配置的梯度提升回归树在测试集上进行性能评估。
print ('R-squared value of GradientBoostingRegressor:', gbr.score(X_test, y_test))
print ('The mean squared error of GradientBoostingRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print ('The mean absoluate error of GradientBoostingRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))

'''
特点分析
许多在业界从事商业分析系统开发和搭建的工作者更加青睐集成模型,并且经常以这些模型的性能变现为基准,与新设计
的其他模型性能进行比较,虽然这些集成模型在训练过程中要消耗更多的时间,但是往往可以提供更高的表现性能和更好的稳定性

'''