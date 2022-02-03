import gym
from ale_py import ALEInterface
from dqn import DQN

ale = ALEInterface()

from ale_py.roms import Breakout
env = gym.make('Breakout-v0', obs_type='rgb', frameskip=5, mode=0, difficulty=0, repeat_action_probability=0.25, full_action_space=True, render_mode=None)
ale.loadROM(Breakout)

# from ale_py.roms import Pong
# env = gym.make('Pong-v0', obs_type='rgb', frameskip=5, mode=0, difficulty=0, repeat_action_probability=0.25, full_action_space=True, render_mode=None)
# ale.loadROM(Pong)

MAX_STEPS = 5000

dqn = DQN(env.observation_space, env.action_space, gamma=0.99, epsilon=1, num_episodes=5)

for episode_i in range(100):
  # start a new epsiode
  observation = env.reset()
  total_reward = 0

  for t in range(MAX_STEPS):
    if episode_i % 10 == 0:
      env.render()

    # pick an action using epsilon-greedy policy
    action = dqn.select_action(env.render('rgb_array'))

    # execute that action, then observe the reward and new state
    new_observation, reward, done, info = env.step(action)
    total_reward += reward

    screen_state = dqn._get_processed_screen(env.render('rgb_array')).numpy()
    new_screen_state = dqn._get_processed_screen(env.render('rgb_array')).numpy()

    # store the agent's experience in the replay memory
    dqn.store_memory([screen_state, action, reward, new_screen_state, done], str(t+1))
    observation = new_observation

    dqn.learn()

    if done:
      print("Episode " + str(episode_i) + " finished after " + str(t+1) + " timesteps with score " + str(total_reward))
      break

  dqn.rewards.append(total_reward)
  dqn.timesteps_per_episode.append(t+1)
  print("   - Epsilon: " + str(dqn.epsilon))

env.close()

dqn.save_model("weights_2.pth")
dqn.save_data()

# weights_1.pth 
# [2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 2.0, 3.0, 1.0, 0.0, 2.0, 3.0, 5.0, 2.0, 1.0, 3.0]
# [243, 98, 121, 136, 99, 159, 122, 183, 208, 185, 139, 128, 117, 110, 137, 160, 143, 203, 107, 224, 148, 135, 121, 167, 117, 120, 199, 141, 187, 169, 164, 177, 197, 176, 153, 126, 166, 215, 114, 114, 160, 177, 123, 113, 188, 214, 279, 161, 128, 199]

