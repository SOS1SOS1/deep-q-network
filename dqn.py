# Author: Shanti Mickens
# Description: This is an implementation of a DQN using PyTorch.
# Date: January 2022

# Deep Q-Network (DQN)
# can learn successful policies directly from high-dimensional sensory inputs using end-to-end RL
# https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

# combines RL with a deep neural network, specifically a deep convolutional network, which has layers
# of tiled convolutional filters, to approximate the optimal action-value function Q*(s, a)

# Q*(s, a) = maximium future discounted reward from starting in state s, taking action a, and then
# following policy pi thereafter

# typically, using a nonlinear function approximator for Q, like a neural network, would lead to RL being
# unstable or possibly diverging, but this is prevented by using a variant of Q-learning
	# - experience replay: randomizes over the data to remove correlations in the observation sequence and smootihing over changes in the data distribution
	# - iterative update that adjusts action-values towards target values that are only periodically updated (every C updates)

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
from numpy import savetxt
from numpy import asarray

from replay_memory import ReplayMemory

## REFERENCE THIS BLOG: https://www.saashanair.com/dqn-code/
# take a look at open ai baselines package
# https://github.com/jasonbian97/Deep-Q-Learning-Atari-Pytorch/blob/master/DQNs.py

# TODO: logging reward and timesteps per episode
class Net(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Net, self).__init__()

		relu = nn.ReLU(True)

		conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
		conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

		fc1 = nn.Linear(2*9*9*32, 256)
		fc2 = nn.Linear(256, output_dim)

		self.cnn = nn.Sequential(conv1, relu, conv2, relu)
		self.classifier = nn.Sequential(fc1, relu, fc2)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

	def forward(self, x):
		x = self.cnn(x)
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = self.classifier(x)
		return x

class DQN():
	N = 10000 # size of replay memory

	def __init__(self, state_space, action_space, gamma, epsilon, num_episodes):
		self.state_space = state_space
		self.action_space = action_space

		self.gamma = gamma
		self.epsilon = epsilon
		self.num_episodes = num_episodes

		# initialize replay memory to capacity N
		self.memory = ReplayMemory(DQN.N)

		# initialize action-value function Q with random weights theta (online q-network)
		self.q = Net(state_space._shape, action_space.n)

		# initialize target action-value function Q_hat with weights theta_- = theta (target q-network)
		self.q_hat = Net(state_space._shape, action_space.n)
		self.q_hat.eval() # because no training is done on this network

		self.counter = 0
		self.C = 200

		self.rewards = []
		self.timesteps_per_episode = []

	def select_action(self, screen_state):
		# STEP 1 - agent selects and executes actions according to an epsilon-greedy policy based on Q 	
		p = random.random() # randome value between 0 and 1
		if p < self.epsilon:
			# exploit - follows the greedy policy with probability 1-epsilon
			with torch.no_grad():
				action = self.q(self._get_processed_screen(screen_state))
				a = torch.argmax(action).item()
		else:
			# explore - selects a random action with probability epsilon
			a = self.action_space.sample()

		# slowly reduce epsilon as the agent learns
		self.epsilon = max(0.01, self.epsilon*0.9999)

		return a		

	def store_memory(self, e_t, time_per_episode):
		# STEP 2 - store the agent's experience at each time-step into a replay memory D
		# e_t: tuple of agent's experience (s_t, a_t, r_t, s_(t+1))
		self.memory.add(e_t)


	def _sample_memories(self, batch_size=32):
		# STEP 3 - Apply Q-learning updates, or minibatch updates, to samples of experience drawn at random from the pool of stored samples
			# improves sample efficiency, reduces variance, and helps break correlations in the data
		return self.memory.sample(batch_size)

	def learn(self):
		# sample random minibatch of transitions from D to use to update the targets
		states, actions, rewards, next_states, dones = self._sample_memories()

		# get q-values for actions taken
		# must reshape actions vector to be (n, 1)
		q_pred = self.q(states.squeeze(1))
		# print("q pred")
		# print(q_pred.shape)

		# STEP 4 - Use a separate network for generating the targets y_j in the Q-learning update
		q_target = self.q_hat(next_states.squeeze(1)).max(dim=1).values
		# print("q target")
		# print(q_target.shape)

		# q-value for terminal states (places where done = true)
		q_target[dones] = 0.0

		y_j = rewards + (self.gamma * q_target)
		y_j = y_j.view(-1, 1)

		# perform gradient descent with respect to the network parameters
		self._gradient_descent(y_j, q_pred)

		# every C updates clone the network Q to obtain a target network Q_hat and use Q_hat to generate the 
		# Q-learning targets for the following C updates to Q
		if self.counter == self.C:
			self._update_target_network()
		
		self.counter += 1

	def _update_target_network(self):
		# generating the targets using an older set of parameters helps make the algorithm more stable, by adding 
		# a delay between the time an update to Q is maade and the time the update affects the targets y_j
		self.counter = 0
		self.q_hat.load_state_dict(self.q.state_dict())

	def _gradient_descent(self, y_j, q_pred):
    	# STEP 5 - perform gradient descent on (y_j - Q)^2 WRT network parameters theta
			# clipping this error term between -1 and 1 corresponds to using an absolute value loss function for 
			# errors outside of (-1, 1) which helps further improve the algorithm's stability
		# zero the parameter gradients
		self.q.optimizer.zero_grad()
		n = list(q_pred.size())[1]
		loss = F.mse_loss(y_j.repeat(1, n), q_pred).mean()
		loss.backward()
		self.q.optimizer.step()
		
	def _get_processed_screen(self, screen):
		# PyTorch expects CHW
		screen = screen.transpose((2, 0, 1))
		transformed_screen = self._transform_screen_data(screen) #shape is [1,1,110,84]
		return transformed_screen

	def _crop_screen(self, screen):
		bbox = [34,0,160,160] #(x,y,delta_x,delta_y)
		screen = screen[:, bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]] #BZX:(CHW)
		return screen

	def _transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
		screen = (np.ascontiguousarray(screen, dtype=np.float32) / 255)
		screen = torch.from_numpy(screen)
		screen = self._crop_screen(screen)
		# Use torchvision package to compose image transforms
		resize = T.Compose([
            T.ToPILImage()
            , T.Grayscale()
            , T.Resize((84, 84)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
            , T.ToTensor()
        ])
		# add a batch dimension (BCHW)
		screen = resize(screen)
		
		return screen.unsqueeze(0)   # BZX: Pay attention to the shape here. should be [1,1,84,84]

	def save_model(self, path):
		torch.save(self.q.state_dict(), path)

	def load_model(self, path):
		self.q.load_state_dict(torch.load(path))
		self.q.eval()

	def load_model_to_continue_training(self, path):
		self.q.load_state_dict(torch.load(path))

	def save_data(self):
		print(self.rewards)
		print(self.timesteps_per_episode)
		# save rewards and timesteps per episode to csv file
		savetxt('rewards.csv', asarray(self.rewards), delimiter=',')
		savetxt('timesteps_per_episode.csv', asarray(self.timesteps_per_episode), delimiter=',')

 


			













