import numpy as np
import matplotlib.pyplot as plt

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
X = []

for _, data in enumerate(lines):
    x = [ float(d)/scale for d in data.split() ]
    # x.append(1)
    X.append(np.array(x))

X = np.array(X)
Y = np.array([ [float(y)] for y in open("a2-train-label.txt", "r").readlines() ])

# Testing data
lines = [ lines for lines in open("a2-test-data.txt", "r").readlines() ]
TEST_X = []

for _, data in enumerate(lines):
    x = [ float(d)/scale for d in data.split() ]
    # x.append(1)
    TEST_X.append(np.array(x))

TEST_X = np.array(TEST_X)
TEST_Y = np.array([ [float(y)] for y in open("a2-test-label.txt", "r").readlines() ])

# set up weights
input_size = 1000
hidden_layer_size_1 = 100
hidden_layer_size_2 = 50
out_size = 1

params = {}

params["W1"] = np.random.randn(input_size, hidden_layer_size_1)
params["W2"] = np.random.randn(hidden_layer_size_1, hidden_layer_size_2)
params["W3"] = np.random.randn(hidden_layer_size_2, out_size)

def forward(input):
    params["A0"] = input

    params["Z1"] = np.dot(params["A0"], params["W1"])
    params["A1"] = np.tanh(params["Z1"])

    params["Z2"] = np.dot(params["A1"], params["W2"])
    params["A2"] = np.tanh(params["Z2"])

    params["Z3"] = np.dot(params["A2"], params["W3"])
    params["A3"] = np.tanh(params["Z3"])

    return params["A3"]

def error(X, Y):
    yHat = forward(X)
    return 0.5*sum((Y-yHat)**2)

def tanh_prime(z):
    t = np.tanh(z)
    return 1 - t * t

def backward(Y):
    d3 = np.multiply(-(Y - params["A3"]), tanh_prime(params["Z3"]))
    dw3 = np.dot(params["A2"].T, d3)

    d2 = np.dot(d3, params["W3"].T) * tanh_prime(params["Z2"])
    dw2 = np.dot(params["A1"].T, d2)

    d1 = np.dot(d2, params["W2"].T) * tanh_prime(params["Z1"])
    dw1 = np.dot(params["A0"].T, d1)

    return dw3, dw2, dw1

def preformance(X, Y):
    yHat = forward(X)

    count = 0
    for i in range(len(yHat)):
        if np.sign(yHat[i]) == Y[i]:
            count += 1
    
    return count / float(len(yHat))

# Train idk 100 times
training_loss = []
testing_loss = []

training_acc = []
testing_acc = []

for i in range(500):
    # for fun plot the error in the testing set
    y = forward(TEST_X)
    testing_loss.append(0.5*sum((TEST_Y-y)**2) / len(TEST_Y))
    testing_acc.append(preformance(TEST_X, TEST_Y))

    # forward prop
    y = forward(X)
    training_loss.append(0.5*sum((Y-y)**2) / len(Y))
    training_acc.append(preformance(X, Y))

    # calculate gradient
    dw3, dw2, dw1 = backward(Y)
    w3, w2, w1 = params["W3"], params["W2"], params["W1"]

    # make the biggest working change
    params["W2"] = w3 - dw3 * 0.05
    params["W2"] = w2 - dw2 * 0.05
    params["W1"] = w1 - dw1 * 0.05

print("Training accuracy:", preformance(X, Y))
print("Testing accuracy:", preformance(TEST_X, TEST_Y))

with open("a2-test-predictions.txt", "w") as f:
    o = [str(np.sign(y[0])) for y in forward(X)]
    f.write(" ".join(o))

plt.plot(training_acc)
plt.plot(testing_acc)
plt.title("accuracy vs iterations")
plt.show()