from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np

# I'm normilzing the inputs between 0 and 1
lines = [ lines for lines in open("a2-train-data.txt", "r").readlines() ]

# Keep track of the min and max values
maximums = []
minimums = []

for _, data in enumerate(lines):
    x = [ float(d) for d in data.split() ]
    maximums.append(max(x))
    minimums.append(min(x))

scale = max(abs(min(minimums)), max(maximums))

# I'm going to be really lazy with this
lines = [ lines for lines in open("a2-train-data.txt", "r").readlines() ]
X_train = []

for _, data in enumerate(lines):
    x = [ float(d)/scale for d in data.split() ]
    # x.append(1)
    X_train.append(np.array(x))

X_train = np.array(X_train)
Y_train = np.ravel(np.array([ [float(y)] for y in open("a2-train-label.txt", "r").readlines()]))

# Testing data
lines = [ lines for lines in open("a2-test-data.txt", "r").readlines() ]
X_test = []

for _, data in enumerate(lines):
    x = [ float(d)/scale for d in data.split() ]
    # x.append(1)
    X_test.append(np.array(x))

X_test = np.array(X_test)
Y_test = np.ravel(np.array([ [float(y)] for y in open("a2-test-label.txt", "r").readlines()]))

# I'm using the same net design as my HW 2
mlp = MLPClassifier(hidden_layer_sizes=(1000,100,50,1), activation='tanh', max_iter=500)
mlp.fit(X_train, Y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Training Report:")
print(classification_report(Y_train,predict_train))

print("Testing Report:")
print(classification_report(Y_test,predict_test))