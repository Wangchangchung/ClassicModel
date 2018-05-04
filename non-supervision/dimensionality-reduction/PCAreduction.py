# Author: Charse
'''
在特征降维中, 主成分分析(Principal Componment Analysis)
是最为经典个实用的特征降维技术,特别时在辅助图像识别方面有突出的表现
'''

import pandas
import numpy
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

digits_train = pandas.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)

digits_test = pandas.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)

X_digits = digits_train[numpy.arange(64)]
y_digits = digits_train[64]


estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):   # python  3 中的用法时时直接使用range遍历, 但是在2 中使用的是xrange
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(numpy.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel("Second Principal Component")
    plt.show()

plot_pca_scatter()


X_train = digits_train[numpy.arange(64)]
y_train = digits_train[64]

X_test = digits_test[numpy.arange(64)]
y_test = digits_test[64]

svc = LinearSVC()
svc.fit(X_train, y_train)

y_predict = svc.predict(X_test)

estimator = PCA(n_components=20)

# 利用训练特征决定fit 20个正交维度的方向, 并转换(transform) 原训练特征
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转换(transform)
pca_X_test = estimator.transform(X_test)


#  使用默认配置车不是花LinearSVC, 对压缩过后的二十维度特征的寻;训练数据进行建模
# 并在测试数据上做出预测, 存储在pca_y_predict 中
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)


# 对使用原始图像高纬度特征训练的支持向量机分类器的性能做出评估
print(svc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=numpy.arange(10).astype(str)))


# 使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能做出评估
print(pca_svc.score(pca_X_test, y_test))
print(classification_report(y_test, pca_y_predict, target_names=numpy.arange(10).astype(str)))


'''
特点分析:

降维/压缩问题则是选取数据具有代表性的特征,在保持数据多样性(Variance) 的基础上,规避掉大量的
特征冗余和噪声,不过这个过程也很有可能会损失一些有用的模式信息,经过大量的实践证明
相较于损失的少部分模型性能,维度压缩能够节省大量用于模型训练的时间,这样一来,使得PCA所带来的模型综合效率变得更加划算

'''




