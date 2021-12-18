# Assigment 4

> Install OpenAI gym and test run this basic driver program. Feel free to tweak or change some of the parameters.
> It contains two simple-minded policies to balance the pole. Which strategy is better? 

Of the 2 included learners the naive approach is on than the random approach. It's possible the random approach could do better but I would bet against it something around 80% of the time. After 100 episodes of each here are the results.

Random:
```
mean: 21.23 
std: 10.806345358168041 
min: 9.0 
max: 63.0
```

Naive:
```
mean: 42.95
std: 8.703303970332186 
min: 25.0
max: 63.0
```

> Design a policy that is better than the two given in the above program.
> Use any ML technique of your own choice. Justify that your policy is better.
> Consider using scikit-learn or tensorflow to implement your ML algorithm. 

