import torch
import numpy as np

class ReplayMemory():

	def __init__(self, N):
		self.N = N
		self.index = 0
		
		self.state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.new_state_memory = []
		self.done_memory = []

	def add(self, e_t):
		# treat end of array as the front, that is store the most recent memory at the end
		# e_t: tuple of agent's experience (s_t, a_t, r_t, s_(t+1), done) where done is True if the episode is over

		# only keeps the N most recent experience tupes, but this isn't necessarily the best option
		if self.index >=self.N:
			self.state_memory[self.index] = e_t[0]
			self.action_memory[self.index] = e_t[1]
			self.reward_memory[self.index] = e_t[2]
			self.new_state_memory[self.index] = e_t[3]
			self.done_memory[self.index] = e_t[4]
		else:
			self.state_memory.append(e_t[0])
			self.action_memory.append(e_t[1])
			self.reward_memory.append(e_t[2])
			self.new_state_memory.append(e_t[3])
			self.done_memory.append(e_t[4])
		
		self.index = self.index + 1 if self.index + 1 < self.N else 0

	def sample(self, batch_size):
		# randomly samples x experiences from the replay memory
		random_sample_indices = np.random.permutation(len(self.state_memory))[:batch_size]

		states = torch.from_numpy(np.array(self.state_memory)[random_sample_indices]).float()
		actions = torch.from_numpy(np.array(self.action_memory)[random_sample_indices])
		rewards = torch.from_numpy(np.array(self.reward_memory)[random_sample_indices]).float()
		next_states = torch.from_numpy(np.array(self.new_state_memory)[random_sample_indices]).float()
		dones = torch.from_numpy(np.array(self.done_memory)[random_sample_indices])

		return (states, actions, rewards, next_states, dones)
