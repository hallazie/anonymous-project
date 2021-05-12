# coding:utf-8

def f1_score(label, pred):
    tp, fp, fn = 0, 0, 0
    for l, p in zip(label, pred):
        l, p = set(l), set(p)
        tp += len(l & p)
        fp += len(p - l)
        fn += len(l - p)
    prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.
    reca = float(tp) / (tp + fn) if (tp + fp) > 0 else 0.
    f1sc = (2 * prec * reca) / (prec + reca) if (prec + reca) > 0 else 0.
    return f1sc


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    a = [
        [1, 0, 2, 2, 0],
        [1, 1, 0, 2, 4],
        [3, 0, 0, 1, 4],
        [3, 0, 1, 4, 0]
    ]
    a = np.array(a)
    b = a == 1
    plt.imshow(b)
    plt.show()
    # print(b)

