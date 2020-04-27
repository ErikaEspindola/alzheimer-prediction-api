from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np

much_data = np.load('muchdata-50-50-30-normalizado.npy', allow_pickle=True)
X = [data[0].flatten() for data in much_data]
y = [data[1] for data in much_data]

Y = []

for i in y:
    label = ''.join([str(j) for j in i.tolist()])
    if label == '001':
        Y.append(1)
    elif label == '010':
        Y.append(2)
    else:
        Y.append(3)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

ct = 0
for i in range(0, len(y_test)):
    if y_test[i] == y_pred[i]:
        ct += 1

print(ct)
print((y_test == y_pred).sum())

print("Accuracy: ", (100*((y_test == y_pred).sum()/len(X_test))))

print(confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None))