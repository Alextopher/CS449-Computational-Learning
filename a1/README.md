# binary_perceptron
Binary Perceptron written for CS449 at Clarkson University. The uses the simple perceptron algorithm that can solves problems where there is a "perfect" solution.

## Run

`python3 main.py training-data.txt`

## Dependencies 

This was written using `Python 3.9.7` and `numpy 1.20.1`. It should work in most scenarios.

## Format

Each point in the train set is represented as space-seperated floats followed by a `Y` or `N` and then a `\n` newline.

For example in the 3d case:

```
-187.52495221 1226.75921175 1817.26741526 Y
1012.31895965 5.14374519193 337.286686965 N
```
