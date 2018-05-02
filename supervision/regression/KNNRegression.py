# Author: Charse
import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weights='uniform'。
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights='distance'。
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# 使用R-squared、MSE以及MAE三种指标对平均回归配置的K近邻模型在测试集上进行性能评估。
print ('R-squared value of uniform-weighted KNeighorRegression:', uni_knr.score(X_test, y_test))
print ('The mean squared error of uniform-weighted KNeighorRegression:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print ('The mean absoluate error of uniform-weighted KNeighorRegression', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))

# 使用R-squared、MSE以及MAE三种指标对根据距离加权回归配置的K近邻模型在测试集上进行性能评估。
print ('R-squared value of distance-weighted KNeighorRegression:', dis_knr.score(X_test, y_test))
print ('The mean squared error of distance-weighted KNeighorRegression:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print ('The mean absoluate error of distance-weighted KNeighorRegression:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))

'''
特点分析:
K 近邻(回归)与K近邻(分类)一样,均属于无参数模型(Nonparametric model)  同样
没有参数训练的过程,但是由于其模型的计算方法非常直观,因此深受广大初学者的喜爱, 这个例子讨论了两种根据数据
样本的相似程度预测回归值的方法,并且验证采用K近邻加权平均的回归策略可以获得较高的模型性能.
'''