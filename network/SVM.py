from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)
# data = pd.read_csv('../data/data_test_new.txt', delimiter=',', header=None)
data = pd.read_csv('../data/dataset/all_data.csv')
print(data)


# #11特征 5+6
data_set = data.iloc[:, :5]

label_set = data.iloc[:, -1]

print(data_set,label_set)
#训练集与测试集切分
train_data,test_data,train_label,test_label = train_test_split(data_set,label_set,test_size=0.4)

# 归一化操作  minmax
scaler = MinMaxScaler()
train_data_tf = scaler.fit_transform(train_data)
test_data_tf = scaler.fit_transform(test_data)

# clf=SVC(kernel='rbf',C=5).fit(train_data,train_label)
clf=SVC(kernel='linear').fit(train_data,train_label)
pred = clf.predict(test_data)
acc = accuracy_score(pred,test_label)
print('acc',acc)




# bp = MLPClassifier(hidden_layer_sizes=(256,128,64), activation='relu',
#                    solver='sgd', alpha=0.0001, max_iter=30,
#                    learning_rate='constant')
# bp.fit(train_data_tf, train_label.astype('int'))
# y_predict = bp.predict(test_data_tf)
# acc = accuracy_score(y_predict,test_label)
# print('acc',acc)
#
# y_test1 = test_label.tolist()
# y_predict = list(y_predict)
# # print(int(y_test1[1]))
# for i in range(len(y_test1)):
#     y_test1[i] = int(y_test1[i])
# print('BP网络：\n', classification_report(test_label.astype('int'), y_predict))
# print("真实数据：\t", y_test1)
# print("预测数据：\t", y_predict)
# print("混淆矩阵：\t",confusion_matrix(y_predict,test_label))

