# Author: Charse
'''
朴素贝叶斯（分类）

模型介绍： 朴素贝叶斯是一个非常简单, 但是实用性很强的分类模型,不过,
和 上述两个基于线性假设的模型(线性分类器和支持向量机分类器)不同，朴素贝叶斯分类器的构造基础是贝叶斯理论
抽象一些说： 贝叶斯分类器会单独考量每一维度特征被分类的条件概率，进而综合这些概率并对其所在的特征向量
作出分类预测，因此则和个模型的基本数学假设是： 各个维度上的特征被分类的条件概率之间是相互独立的

'''
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
# 导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 从sklearn.naive_bayes 里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 性能分析报告
from sklearn.metrics import classification_report


news = fetch_20newsgroups(subset='all')

print("news.data len:",len(news.data))
print("news.data[0]:", news.data[0])


X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
print("news.target", news.target)

# 首先将文本转化为特征向量，然后利用朴素贝叶斯模型从训练数据集中估计参数, 最后利用这些概率
# 参数对同样转换为特征向量的测试新闻样本进行类别预测
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 从使用默认配置初始化贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对测试样本进行类别预测, 结果存储在变量中
y_predict = mnb.predict(X_test)

## 性能测评
'''
与线性分类器的评价指标一样,我们使用准确性,召回率，精确率和F1指标 这4个维度对贝叶斯
模型在20类新闻文本分类任务上的性能进行评估
'''

print("The accuracy of Naive Bayes Classifier is:", mnb.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))

## 特点分析
'''
贝叶斯模型被广泛应用于 海量互联网文本任务, 由于其较强的特征条件独立假设， 
使得模型预测所需要估计的参数规模从幂指数级向线性量级减少，极大节约了内存消耗和计算时间
但是，正是受这种强假设的限制，模型训练是无法将各个特征之间的联系考虑在内，使得该模型在其他数据
特征关联性较强的分类任务上表现不佳
'''

