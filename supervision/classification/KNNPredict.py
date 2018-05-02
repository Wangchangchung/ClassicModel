# Author: Charse
'''
K 近邻(分类)
模型介绍: K 近邻模型本身非常直观且容易理解

'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

iris = load_iris()

print("iris.data.shape:", iris.data.shape)

print("iris.DESCR:", iris.DESCR)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

print("iris.target:", iris.target)

stand = StandardScaler()
X_train = stand.fit_transform(X_train)
X_test = stand.transform(X_test)

knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

print("The accuracy of K-Nearest Neighbor Classifier is:", knc.score(X_test, y_test))

print(classification_report(y_test, y_predict, target_names=iris.target_names))

'''
特点分析
K近邻(分类) 是非常直观的机器学习模型, 因此深受广大初学者的喜爱, 读多教科书常常以此模型为例抛砖引玉
便足以看出其不仅特别，而且尚有弊端之处，
K近邻算法与其他模型最大的不同在于： 该模型没有参数训练过程，也就是说，我们并没有通过任何学习算法分析训练数据
而只是根据测试样本在训练数据的分布直接作出分类决策, 因此，K近邻属于无参数模型(Nonparametric model)中非常简单一种
然而，正是这样的决策算法，导致了其非常高的计算复杂度和内存消耗，因为该模型每处理一个测试样本，都需要对所有
预先加载在内存的训练样本进行遍历，逐一计算相似度，排序并且选取k个最近邻训练样本的标记，进而作出分类决策，
这个是平方级别的算法复杂度，一旦数据规模稍大，使用者需要权衡根多计算时间的代价
'''





