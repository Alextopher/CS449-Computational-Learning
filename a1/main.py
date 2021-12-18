#!python3

import numpy as np
import sys

# Number of dimensions
N = 1024

# read data from file
pairs = []
with open(sys.argv[1]) as f:
    for line in f.readlines():
        split = line.split(' ')
        
        # convert string numbers to floats
        v = np.array([float(n) for n in split[:N]])

        # normalize inputs vectors to have magnitude 1
        # this doesn't change what side of the plane the points lie on
        # as long as our peceptron is going through the origin
        v = v / np.linalg.norm(v)

        # extract the boolean 
        is_yes = split[-2] == 'Y'

        pairs.append((v, is_yes))


## Simple binary perceptron algorithm
# error function
def error(weights, pairs):
    return sum([(np.dot(weights, x) >= 0) == b for (x, b) in pairs]) / float(len(pairs))

# Weights all start at 0
weights = np.zeros(N)
t = 0
time_since_error = 0

print("maximum interations: " + str(len(pairs) * 20))
for (x, y) in pairs * 20:
    # w * x > 0
    prediction = np.dot(weights, x) >= 0
    t += 1

    # if we guessed right there is no change to make
    if prediction != y:
        # if the correct answer is positive or negative
        if y:
            weights += x
        else:
            weights -= x

        time_since_error = 0
    else:
        time_since_error += 1

    # check if we've correctly predicted every training point
    if time_since_error == len(pairs):
        break

# Save weights in Tino's format
outfile = sys.argv[2] if len(sys.argv) >= 3 else 'weights.txt'
with open(outfile, 'w') as f:
    f.write(" ".join([str(w) for w in weights]))

print("wrote weight to " + outfile)
print("correctness: " + str(error(weights, pairs) * 100) + "%")
print("iterations (how many dot products): " + str(t))