
# #confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# classes = ['A','B','C','D','E']
# confusion_matrix = np.array([(9,1,3,4,0),(2,13,1,3,4),(1,4,10,0,13),(3,1,1,17,0),(0,0,0,1,14)],dtype=np.float64)
#
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
# plt.title('confusion_matrix')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes)
# plt.yticks(tick_marks, classes)
#
# thresh = confusion_matrix.max() / 2.
# #iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# #ij配对，遍历矩阵迭代器
# iters = np.reshape([[[i,j] for j in range(5)] for i in range(5)],(confusion_matrix.size,2))
# for i, j in iters:
#     plt.text(j, i, format(confusion_matrix[i, j]))   #显示对应的数字
#
# plt.ylabel('Real label')
# plt.xlabel('Prediction')
# plt.tight_layout()
# plt.show()



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

y_pred = ['0','1','2','3','4','5','6','7','8','9','0','2','2','3','4','5','6','7','8','9'] # ['2','2','3','1','4'] # 类似的格式
y_true = ['0','1','2','3','4','5','6','7','8','9','0','1','2','3','4','5','6','7','8','9'] # ['0','1','2','3','4'] # 类似的格式


cnf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
print(cnf_matrix, cnf_matrix.shape, cnf_matrix.dtype)
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
# plt.matshow(cnf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
for i in range(len(cnf_matrix)):
    for j in range(len(cnf_matrix)):
        plt.annotate(cnf_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.xticks(np.arange(10))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Elas-hand Confusion Matrix')
# plt.savefig('Elas-hand_cm.png', format='png')
plt.show()
