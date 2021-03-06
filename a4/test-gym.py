# Playing with OpenAI Gym: CartPole-v0

import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

##################################################################################################
# policies

def naive_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

def random_policy(obs):
	angle = obs[2]
	return 0 if np.random.uniform() < 0.5 else 1

##################################################################################################

def naive_main( policy ):
	env = gym.make("CartPole-v0")
	obs = env.reset()
	#env.render()

	# episodic reinforcement learning
	totals = []
	for episode in range(100):
		episode_rewards = 0
		obs = env.reset()
		while True:
			action = policy(obs)
			obs, reward, done, info = env.step(action)
			#env.render()
			episode_rewards += reward
			if done:
				#env.render()
				break
		totals.append(episode_rewards)
		print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

# DQN Adapted from https://github.com/gsurma/cartpole MIT LICENSE
GAMMA = 0.95
LEARNING_RATE = 0.001

BATCH_SIZE = 10

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:
    def __init__(self, observation_space, action_space, model=""):

        self.action_space = action_space
        self.memory = []

        if model == "":
            self.exploration_rate = EXPLORATION_MAX

            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(6, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        else:
            self.exploration_rate = 0

            self.model = load_model('model')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(0, self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

# Trains a deep q network
def dqn():
    env = gym.make("CartPole-v0")

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    totals = []
    for _ in range(100):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        episode_rewards = 0
        while True:
            action = dqn_solver.act(state)
            obs, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            state_next = np.reshape(obs, [1, observation_space])
            #env.render()
            episode_rewards += reward
            dqn_solver.remember(state, action, reward, state_next, done)
            dqn_solver.experience_replay()
            state = state_next

            if done:
                #env.render()
                break

        totals.append(episode_rewards)
        print(_, episode_rewards, "|", np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

        # save the model weights
        dqn_solver.model.save("model")

def run():
    env = gym.make("CartPole-v0")

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, "model")
    totals = []
    
    for episode in range(100):
        episode_rewards = 0
        obs = env.reset()
        obs = np.reshape(obs, [1, observation_space])
        while True:
            action = dqn_solver.act(obs)
            obs, reward, done, info = env.step(action)
            obs = np.reshape(obs, [1, observation_space])
            env.render()
            episode_rewards += reward
            if done:
                env.render()
                break
        totals.append(episode_rewards)
        print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

##################################################################################################

if __name__ == "__main__":
    # Uncomment which of these you want to run
    #naive_main( random_policy )
    #naive_main( naive_policy )
	#dqn() # Train a DQN ( and each episode save the weights to ./model )
    run() # Run the saved DQN

##################################################################################################

