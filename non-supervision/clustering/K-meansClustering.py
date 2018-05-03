# Author: Charse
'''
无监督学习(Unsupervised Learning) 着重与发现数据本身的分布特点
与监督学习(Supervised Learning)不同 无监督学习不需要对数据进行标记,这样在节省大量人工的同时
也让可以利用的数据规模变得不可限量

从功能角度讲,无监督学习模型可以帮助我们发现数据"群落"
同时也可以寻找"离群"的样本,另外,,对于特征维度非常高的数据样本,我们同样可以通过无监督学习对数据
进行降维,保留最具有区分性的低维度特征,这些都是在海量数据处理中是非常使用的技术

K 均值算法:
数据聚类时无监督学习的主流应用之一,最为经典并且易用的聚类模型,当属 K 均值(K-means)算法
该算法要求我们预先设定聚类的个数,然后不断的更新聚类中心,经过几轮这样的迭代,最后的目标就是
要让所有数据点到其所属聚类中心距离的平方和趋近稳定

'''
import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score

digits_train = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                               header=None)
digits_test = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                               header=None)

X_train = digits_train[numpy.arange(64)]
y_train = digits_train[64]

print(X_train)
print(y_train)

print('-----------------------------------------')

X_test = digits_test[numpy.arange(64)]
y_test = digits_test[64]

# 初始化 KMeans模型, 并设置聚类中心数量为10
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)

# 逐条判断每个测试图像所属的聚类中心
y_predict = kmeans.predict(X_test)
# 0.6592893679369013,  0.6621773801044615
print(metrics.adjusted_rand_score(y_test, y_predict))


plt.subplot(3, 2, 1)

x1 = numpy.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = numpy.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])

print(x1)
print(x2)
X = numpy.array(list(zip(x1, x2))).reshape(len(x1), 2)
print(X)

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title("Instances")
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []


for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)

    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
        plt.title("K=%s, silhouette coefficient=%0.03f" % (t, sc_score))

plt.figure()


print("======================")
print(clusters)
print(sc_scores)


plt.plot(clusters, sc_scores, "")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient Score")
plt.show()

'''
如果被用于评估的数据没有所属类别,那么我们习惯使用轮廓系数(Silhouette Coefficient)
来度量聚类结果的质量,轮廓系数同时兼顾了聚类的凝聚度(Cohesion)和分离度(Seqaration) 
用于评估聚类的效果并且取值范围为[-1, 1].  轮廓系数值越大,表示聚类效果越好,具体的计算
步骤如下: 
1. 对于已经聚类数据中第 i个 样本 x^i,   计算x^i 与其同一个类簇内的所有其他样本距离的平均值,记做
a^i
'''