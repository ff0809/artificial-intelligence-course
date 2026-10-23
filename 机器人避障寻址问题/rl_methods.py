import numpy as np
import random
from typing import List, Tuple, Set, Optional
from rl_environment import GridWorldEnv

def train_sarsa(env, episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.2):
    q_table = np.zeros((env.observation_space.high[0]+1, env.observation_space.high[1]+1, 4))
    epsilon_decay = 0.998
    min_epsilon = 0.05
    
    for _ in range(episodes):
        state = env.reset()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state[0], state[1]])
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state[0], next_state[1]])
            td_target = reward + gamma * q_table[next_state[0], next_state[1], next_action]
            q_table[state[0], state[1], action] += alpha * (td_target - q_table[state[0], state[1], action])
            state, action = next_state, next_action
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return q_table


def train_double_q(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.3):
    q1 = np.zeros((env.observation_space.high[0]+1, env.observation_space.high[1]+1, 4))
    q2 = np.zeros_like(q1)
    epsilon_decay = 0.998
    min_epsilon = 0.05
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q1[state[0], state[1]] + q2[state[0], state[1]])
            next_state, reward, done, _ = env.step(action)
            if random.random() < 0.5:
                best_next_a = np.argmax(q1[next_state[0], next_state[1]])
                td_target = reward + gamma * q2[next_state[0], next_state[1], best_next_a]
                q1[state[0], state[1], action] += alpha * (td_target - q1[state[0], state[1], action])
            else:
                best_next_a = np.argmax(q2[next_state[0], next_state[1]])
                td_target = reward + gamma * q1[next_state[0], next_state[1], best_next_a]
                q2[state[0], state[1], action] += alpha * (td_target - q2[state[0], state[1], action])
            state = next_state
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return (q1 + q2) / 2


def plan_with_sarsa(grid, start, goal, episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.2, max_steps=300):
    env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
    q_table = train_sarsa(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)

    state = env.reset()
    done = False
    path = [tuple(state)]
    frames = []
    visited = {tuple(state)}
    step_count = 0
    max_rollout = 500

    while not done and step_count < max_rollout:
        action = int(np.argmax(q_table[state[0], state[1]]))
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


def plan_with_double_q(grid, start, goal, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.3, max_steps=300):
    env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
    q_table = train_double_q(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)

    state = env.reset()
    done = False
    path = [tuple(state)]
    frames = []
    visited = {tuple(state)}
    step_count = 0
    max_rollout = 500

    while not done and step_count < max_rollout:
        action = int(np.argmax(q_table[state[0], state[1]]))
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