
#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
classes = ['A','B','C','D','E']
confusion_matrix = np.array([(9,1,3,4,0),(2,13,1,3,4),(1,4,10,0,13),(3,1,1,17,0),(0,0,0,1,14)],dtype=np.float64)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(5)] for i in range(5)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]))   #显示对应的数字

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()
