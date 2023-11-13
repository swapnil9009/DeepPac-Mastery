

# Starting with the importing neccessary libraries and packages
!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

import os
import numpy as np
import torch
import random
import torch.nn as neural_nets
import torch.optim as optim
import torch.nn.functional as func
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

# Now creating the architecture of our deep convolutional q learning model
class Architecture(neural_nets.Module):
  def __init__(self,action_size,seed = 42):
    super(Architecture,self).__init__()
    self.seed = torch.manual_seed(seed)
    self.cnv_layer1 = neural_nets.Conv2d(3,32, kernel_size= 8, stride = 4)
    self.batch_norm_layer1 = neural_nets.BatchNorm2d(32)
    self.cnv_layer2 = neural_nets.Conv2d(32,64, kernel_size = 4, stride = 2)
    self.batch_norm_layer2 = neural_nets.BatchNorm2d(64)
    self.cnv_layer3 = neural_nets.Conv2d(64,64, kernel_size= 3, stride = 1)
    self.batch_norm_layer3 = neural_nets.BatchNorm2d(64)
    self.cnv_layer4 = neural_nets.Conv2d(64,128, kernel_size = 3, stride = 1)
    self.batch_norm_layer4 = neural_nets.BatchNorm2d(128)
    self.full_connected_layer1 = neural_nets.Linear(10 * 10 * 128, 512)
    self.full_connected_layer2 = neural_nets.Linear(512, 256)
    self.full_connected_layer3 = neural_nets.Linear(256, action_size)

  def forward(self,state):
    x = func.relu(self.batch_norm_layer1(self.cnv_layer1(state)))
    x = func.relu(self.batch_norm_layer2(self.cnv_layer2(x)))
    x = func.relu(self.batch_norm_layer3(self.cnv_layer3(x)))
    x = func.relu(self.batch_norm_layer4(self.cnv_layer4(x)))
    x = x.view(x.size(0), -1)
    x = func.relu(self.full_connected_layer1(x))
    x = func.relu(self.full_connected_layer2(x))
    return self.full_connected_layer3(x)


# Setting up the environment

import gymnasium as gym
env = gym.make('MsPacmanDeterministic-v0', full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
total_actions = env.action_space.n
state_shape, state_size, total_actions

# initialising parameter
lr = 5e-4
batch_size = 64
gamma = 0.99

from PIL import Image
from torchvision import transforms

def image_preprocessing(frame):
  frame = Image.fromarray(frame)
  image_process = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
  return image_process(frame).unsqueeze(0)

# Now creating the AI Player class

class Player():
  def __init__(self,action_size):
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.action_size = action_size
    self.local_network = Architecture(state_size,action_size).to(self.device)
    self.main_network = Architecture(state_size,action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)
    self.memory = deque(maxlen=10000)

  def step(self, state, action, rewards, next_states, confirmations):
    state = image_preprocessing(state)
    next_states = image_preprocessing(next_states)
    self.memory.append((state, action, rewards, next_states, confirmations))
    if len(self.memory) > batch_size:
      experiences = random.sample(self.memory, k = batch_size)
      self.fit(experiences,gamma)

  def act(self,states,epsilon = 0.):
    states = image_preprocessing(states).to(self.device)
    self.local_network.eval()
    with torch.no_grad():
      action_values = self.local_network(states)
    self.local_network.train()
    if random.random() > epsilon:
      print("Action values:", action_values.cpu().data.numpy())
      return np.argmax(action_values.cpu().data.numpy())
    else:
      print("Action values:", action_values.cpu().data.numpy())
      return random.choice(np.arange(self.action_size))

  def fit(self,experiences, gamma):
    states, actions, rewards, next_states, confirmations = zip(*experiences)
    states = torch.from_numpy(np.vstack(states)).float().to(self.device)
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    confirmations = torch.from_numpy(np.vstack(confirmations).astype(np.uint8)).float().to(self.device)
    q_targets_next = self.main_network(next_states).detach().max(1)[0].unsqueeze(1)
    normal_q_targets = rewards + gamma * q_targets_next * (1 - confirmations)
    predicted_actions = self.local_network(states).gather(1,actions)
    loss = func.mse_loss(normal_q_targets,predicted_actions)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

total_actions

player = Player(total_actions)

# Training the model
total_episodes = 2000
tota_timesteps_per_episode = 10000
epsilon_starting  = 1.0
epsilon_ending  = 0.01
epsilon_decay  = 0.995
epsilon = epsilon_starting
scores_on_100_episodes = deque(maxlen = 100)

for episode in range(1, total_episodes + 1):
  state, _ = env.reset()
  cumulative_rewards = 0
  for time in range(tota_timesteps_per_episode):
    action = player.act(state,epsilon)
    print("Selected action:", action)
    next_state, reward, confirmation, _, _ = env.step(action)
    print("Environment step result:", next_state, reward, confirmation)
    player.step(state,action,reward,next_state,confirmation)
    state = next_state
    cumulative_rewards += reward
    if confirmation:
      break
  scores_on_100_episodes.append(cumulative_rewards)
  epsilon = max(epsilon_ending,epsilon_decay*epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 600:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    torch.save(Player.local_network.state_dict(), 'parameters_detail.pth')
    break

# to show the video
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(player, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = player.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(player, 'MsPacmanDeterministic-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()