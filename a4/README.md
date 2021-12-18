# Assigment 4

## How to run
Within test-gym.py scroll to the bottom and edit the `if __name__ == "__main__":` block.

```

```

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

Well I looked up a guide and used Deep Q Learning. I think someone covered this in their presentation but Deep Q Learning is just Q learning but the observation vs action table is modeled by a deep neural network. This means the network is free to learn the non linear relationship between the pole and movement unlike the given example. I'd like to continue playing around with the algorithm to make some novel tweaks but trainning takes too long even with my graphics card.

Results:

After 100 episodes I got the following performance:
```
mean: 73.47
std: 34.22497772095696
min: 43.0 
max: 200.0
```

The model hasn't learn to go infinite (maybe this isn't possible in cartpole?) but it is doing better than the naive approach. It seems to have learned to move in the direction of the falling pole with some confidence. From what I've seen it hasn't learned to catch and recenter itself. I wonder if you tweaked the reward function and rewarded it somewhat for staying close to the middle if it would learn better habbits.

It's also interesting how wide the std is, not sure why.