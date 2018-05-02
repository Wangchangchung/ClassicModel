# Author: Charse
'''
回归树 :
回归树在选择不同特征特征作为分裂节点的策略上,与 之前的分类决策树的思路类似,不同之处在于,回归树叶节点
节点的数据类型不是离散型,而是连续型,决策树每个节点依照训练数据表现的概率倾向决定了其最终的预测类别
而回归树的叶节点确实一个个具体的值,从预测值连续这个意义上严格的讲,回归树不能称为"回归算法"
因为回归树的叶节点返回的是 一团 训练数据的均值,而不是具体的,连续的预测值

'''

import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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

dtr = DecisionTreeRegressor()
# 用波士顿房价的训练数据构建回归树
dtr.fit(X_train, y_train)
# 使用默认配置的单一回归树对测试数据进行预测,并将预测值存储在变量中
dtr_y_predict = dtr.predict(X_test)


# 使用R-squared、MSE以及MAE指标对默认配置的回归树在测试集上进行性能评估。
print ('R-squared value of DecisionTreeRegressor:', dtr.score(X_test, y_test))
print ('The mean squared error of DecisionTreeRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print ('The mean absoluate error of DecisionTreeRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))

'''
特点分析
在系统地介绍了决策(分类)树与回归树之后,可以总结这类树模型的优点:
1, 树模型可以解决非线性特征的问题
2, 树模型不要求对特征标准化和统一量化,即数值型和类别型特征都可以直接被应用在树模型的构建和预测过程中
3, 因为上述原因,树模型也可以直观的输出决策过程,使得预测结果具有可解释性
同时,树模型也有一些显著的缺陷
1.  正是因为树模型可以解决复杂的非线性拟合问题,所以更加容易因为模型搭建过于复杂而丧失对新数据预测的精度(泛化力)
2.  树型模型从上至下的预测流程会因为数据细微的更改而发生较大的结构变化,因此预测稳定性较差
3. 依托训练数据构建最佳的树型模型时NP难问题,即在有限事件内无法找到最优解的问题,因此我们所使用类似贪婪算法的解法只能找到
一些次优解,这也是为什么我们经常借助集成模型,在多个次优解中寻觅更高的模型性能

'''
