
'''
Created on 2017-10-19
@author:Cyan-du
说明：在约会网站上使用k-近邻算法，用sklearn来实现
'''
from sklearn import neighbors
import matplotlib.pyplot as plt
from numpy import *
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

def file2matrix(filename):
    """
    导入训练数据
    """
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵，三个特征 所以行数为numberOflines，列数为3
    # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


if __name__ == '__main__':
    datingDataMat,datingLabels = file2matrix('/Users/dudu/Desktop/MachineLearning-master/input/2.KNN/datingTestSet2.txt')
    X_train, X_test, y_train, y_test = train_test_split(datingDataMat, datingLabels, test_size=0.66, random_state=4)
    n_neighbors = 2

    #可用交叉验证来选择最为合适的n_neighbors
    k_range = range(1,31)
    k_scores = []
    # 在今后的测试当中，也可以通过换不同的model来测试哪个model比较好
    # for k in range(1,31):
    #     clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    #     loss = -cross_val_score(clf, datingDataMat, datingLabels, cv=5, scoring='mean_squared_error')  # for regression
    #     scores = cross_val_score(clf,datingDataMat,datingLabels,cv=10,scoring='accuracy')#for classification
    #     k_scores.append(loss.mean())

    # plt.plot(k_range,k_scores)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.show()
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X_train,y_train)

    #这一步采取了交叉验证，score输出了五组成绩
    scores = cross_val_score(clf,datingDataMat,datingLabels,cv=5,scoring='accuracy')
    print(scores)
    #输出结果为：[ 0.76732673  0.79104478  0.83417085  0.7638191   0.81909548]
    print(scores.mean())
    print(clf.score(X_train,y_train))
    print(clf.score(X_test,y_test))