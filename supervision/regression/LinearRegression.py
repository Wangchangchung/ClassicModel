# Author: Charse

'''
线性回归器：
线性模型中也可用用于分类，之前的分类模型中也使用了线性模型
在线性分类器中重点介绍了用于分类的线性模型，其中为了便于将原来在实数域
上计算结果映射到（0,1）区间，映入了逻辑斯蒂函数，而在线性回归问题中，由于预测目标直接是
实数域上的数值，因此优化目标就更加简单，即最小化预测结果与真实值之间的差异
'''
import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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



'''
对数据进行初步检查发现，预测目标房价直接爱呢的差异较大，因此需要对特征以及目标值进行标准化处理
'''
stand_X = StandardScaler()
stand_y = StandardScaler()

X_train = stand_X.fit_transform(X_train)
X_test = stand_X.transform(X_test)

y_train = stand_X.fit_transform(numpy.array(y_train).reshape(1, -1))
print(y_test)
y_test = stand_X.transform(y_test[:1,:])



# 使用最简单的线性回归模型 LinearRegression和SGDRegressor 分别对波士顿房价数据进行训练学习以及预测
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)


sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgd_y_predict = sgdr.predict(X_test)

## 性能测评
'''
# 不同于类别预测，我们不能苟求回归预测的数值结果要严格与真实值相同，
# 一般情况下，我们希望权衡预测值与真实值之间的差距，因此可以通过多种测评函数进行评价
# 其中最为直观的评价指标包括 平均绝对误差(Mean Absolute Error, MAE) 以及均方误差(Mean Squared Error, MSE)
# 因为这也是线性回归，模型所需要优化的目标

然而，差值的绝对值或者平方，都会随着不同的预测问题而变化巨大，欠缺在不同问题中的可比性
因此，我们要考虑到测评指标需要具备某些统计学含义，类似于分类问题评价中的准确指标
回归问题也有R-request 这样的评价方式，既考量了回归指与真实值的差异，同时也兼顾问题本身真实值的变动

'''
# 使用 LinearRegression 模型自带的评估模块，并输出评估结果
print("The value of default measurement of LinearRegression is:", lr.score(X_test, y_test))

# 使用r_score 模块进行评估

print("The value of R-squared  LinearRegression is:", r2_score(y_test, lr_y_predict))

# 使用mean_squared_error 模块进行评估
print("The mean squared error of LinearRegression is:",
      mean_squared_error(stand_y.inverse_transform(y_test), stand_y.inverse_transform(lr_y_predict)))

# 使用mean_absolute_error 模块
print("The mean absolute error of LinearRegression is:",
      mean_absolute_error(stand_y.inverse_transform(y_test), stand_y.inverse_transform(lr_y_predict)))


## 使用SGDRegressor模型自带的评估模块

# 使用LinearRegression模型自带的评估模块，并输出评估结果
print("The value of default measurement of SGDRegressor is:", sgdr.score(X_test, y_test))

# 使用r_score 模块进行评估
print("The value of R-squared f SGDRegressor is:", r2_score(y_test, sgd_y_predict))

# 使用mean_squared_error 模块进行评估
print("The mean squared error of SGDRegressor is:",
      mean_squared_error(stand_y.inverse_transform(y_test), stand_y.inverse_transform(sgd_y_predict)))

# 使用mean_absolute_error 模块
print("The mean absolute error of SGDRegressor is:",
      mean_absolute_error(stand_y.inverse_transform(y_test), stand_y.inverse_transform(sgd_y_predict)))


'''
分析, 使用随机梯度下降参数的方法SDGRegressor在性能表现上不及使用解析方法的  ；
但是如果面对训练规模十分庞大的任务，随机梯度法无论是在分类还是回归问题上都表现得十分高效，可以在不损失过多性能
的前提下，节省大量计算时间。

根据Scikit-lenrn官网的建议，如果数据规模超过10万， 推荐使用随机梯度法估计参数模型（SGD Classifier/Regressor）


特点分析

线性回归器时最为简单，易用的回归模型，正是因为其对特征与回归目标之间的线性假设，从某种程度
上说也局限了其应用范围，特别是，现实生活中许多实例数据的各个特征与回归目标之间，绝大多数不能保证严格的线性关系
尽管如此，在不清楚特征之间关系的前提下，我们仍然可以使用线性回归模型作为大多数科学领域试验的基线系统（Baseline system）

'''