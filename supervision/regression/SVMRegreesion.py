# Author: Charse
'''支持向量机（回归）

'''

import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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


# 使用线性核函数配置的支持向量进行回归训练,并且对测试样本进行预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练,并且对测试样本进行预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练,并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)


# 线性核函数
print("R-squared value of linear SVR is", linear_svr.score(X_test, y_test))

print("The mean squared error of linear SVR is",
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print("The mean absolute error of linear SVR is",
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

# 多项式核函数
print("R-squared value of linear Poly SVR is", poly_svr.score(X_test, y_test))

print("The mean squared error of Poly SVR is",
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print("The mean absolute error of Poly SVR is",
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))


# 径向基核函数
print("R-squared value of linear rbf SVR is", rbf_svr.score(X_test, y_test))

print("The mean squared error of rbf SVR is",
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))

print("The mean absolute error of rbf SVR is",
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))

## 性能测评
'''
使用径向基(Radial basis function)核函数对特征进行非线性映射之后,支持向量机展现了最佳的回归性能
这个例子展示了不同配置模型在相同数据上所表现的性能差异,该模型还可以通过配置不同的核函数来改变模型性能,因此
建议读者在使用时多尝试几种配置进而获得更好的预测性能

核函数时一项非常有用的特征技巧,同时在数学描述上也略为复杂,因此在本书中不做过度引申
简单一些理解,便是通过某种函数计算,将原有的特征映射到更高维度空间,从而尽可能达到新的高纬度特征线性可分的程度
结合支持向量机的特点,这种高纬度线性可分的数据特征恰好可以发挥其模型的优势
'''







