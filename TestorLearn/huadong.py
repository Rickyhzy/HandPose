import numpy as np
from random import randrange


def sliding_window(data, sw_width=3, in_start=0):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
    '''
    X, y = [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if in_end < len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end, 0]
            train_seq = train_seq.reshape((len(train_seq), 1))
            X.append(train_seq)
            # y.append(data[in_end:out_end, 0])
        in_start += 1

    return np.array(X), np.array(y)


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.randint(0, 10, size=(10, 2))
    print(data)
    X, y = sliding_window(data)
    print(X, '\n')
