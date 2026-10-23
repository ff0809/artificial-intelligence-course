# dqn_planner.py
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Set, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from rl_environment import GridWorldEnv

class DQN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim=2, action_dim=4, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 memory_size=20000, batch_size=64, target_update_freq=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.policy_net = DQN(state_dim, 128, action_dim)
        self.target_net = DQN(state_dim, 128, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def act(self, state, eval_mode=False):
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def plan_with_dqn(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                  episodes: int = 1000, max_steps: int = 300,
                  render_training: bool = False) -> Tuple[Optional[List[Tuple[int, int]]],
                  List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]]]:
    env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
    agent = DQNAgent(state_dim=2, action_dim=4)

    for ep in range(episodes):
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            prev_dist = abs(state[0]-goal[0]) + abs(state[1]-goal[1])
            new_dist = abs(next_state[0]-goal[0]) + abs(next_state[1]-goal[1])
            shaped_reward = reward + 0.2 * (prev_dist - new_dist)
            agent.remember(state, action, shaped_reward, next_state, done)
            agent.replay()
            state = next_state
            step += 1
        if render_training and (ep+1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes}, Epsilon: {agent.epsilon:.3f}")

    state = env.reset()
    done = False
    path = [tuple(state)]
    frames = []
    visited = {tuple(state)}
    step_count = 0
    max_rollout = 500

    while not done and step_count < max_rollout:
        action = agent.act(state, eval_mode=True)
        next_state, _, done, _ = env.step(action)
        next_pos = tuple(next_state)
        if next_pos not in visited:
            visited.add(next_pos)
            path.append(next_pos)
        current = tuple(state)
        neighbors = set()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = current[0]+dr, current[1]+dc
            if env._is_free((nr, nc)):
                neighbors.add((nr, nc))
        frontier = neighbors - visited
        frames.append((current, visited.copy(), frontier))
        state = next_state
        step_count += 1

    if done and tuple(state) == goal:
        if path[-1] != goal:
            path.append(goal)
        return path, frames
    return None, frames