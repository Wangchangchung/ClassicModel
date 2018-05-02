# Author: Charse
'''
支持向量机（分类）模型

'''

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 从sklearn.datasets里导入手写数字加载器  scikit-learn内部集成了手写体数字图片数据集
digits = load_digits()

# 检查数据规模和特征维度 (1797L, 64L)
# 表示该手写体的数码图像数据共有1797条数据，并且每一幅图片是由 8x8 = 64 的像素矩阵表示
# 在模型使用这些像素矩阵的时候，我们习惯将2D的图片像素矩阵逐行首尾拼接为1D的像素特征向量。
# 这样做也许损失一些数据本身的结构信息，但是很遗憾的是，我们当下所介绍的经典模型都没有堆结构信息
# 进行学习的能力

print("digits.data.shapr:", digits.data.shape)

# 随机选取75%的数据作为训练样本, 其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

# 检查训练集与测试集的数据规模
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

# 需要堆训练和测试的特征数据进行标准化
stand = StandardScaler()
X_train = stand.fit_transform(X_train)
X_test = stand.fit_transform(X_test)

# 初始化线性假设的支持向量机分类器 LinearSVC
lsvc = LinearSVC()
# 进行模型训练
lsvc.fit(X_train, y_train)
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果存储在变量中
y_predict = lsvc.predict(X_test)


## 性能测评

# 使用模型自带的评估函数进行准确性测评
print("The Accuracy of Linear SVC is:", lsvc.score(X_test, y_test))
# 依然使用sklearn.metrics里面的classification_report 模块堆预测结果做更加详细的分析
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))


'''
召回率，准确率和F1 指标最先适用于二分类任务,  但是在本实例中, 我们的分类目标有10个类别， 即0-9的10个数字
因此无法直接计算上诉三个指标，通常的做法是，逐一评估某个类别的这三个性能指标：
我们把所有其他的类别看作阴性(负)样本，这样一来，就创造了10个二分类任务，事实上，学习模型在对待多分类任务时
就是这样做的
'''

## 特点分析
'''
支持向量机模型曾经在机器学习领域繁荣发展了很长一段时间，主要原因在于其精妙的模型假设
可以帮助我们在海量甚至高维度的数据中, 筛选对预测任务最为有效的少数训练样本，这样不仅节省了模型学习所需要的数据内存
同时页提高了模型预测的预测性能，然而，要获得如此的优势就必然要付出更多的计算代价(CPU资源和计算时间)
'''





