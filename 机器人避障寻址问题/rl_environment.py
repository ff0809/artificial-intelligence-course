import numpy as np
import gym
from gym import spaces
from typing import Dict, Optional, Tuple, Any, List, Set

class GridWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 grid: np.ndarray,
                 start: Tuple[int, int],
                 goal: Tuple[int, int],
                 max_steps: int = 200):
        super().__init__()
        self._grid = np.asarray(grid, dtype=np.int8).copy()
        self._start = self._find_nearest_free_cell(start, "start")
        self._goal = self._find_nearest_free_cell(goal, "goal")
        self._validate_positions(self._start, self._goal)
        self._agent_pos = self._start
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([self._grid.shape[0] - 1, self._grid.shape[1] - 1], dtype=np.int32),
            dtype=np.int32,
        )

    @property
    def grid(self) -> np.ndarray:
        return self._grid.copy()

    @property
    def start(self) -> Tuple[int, int]:
        return self._start

    @property
    def goal(self) -> Tuple[int, int]:
        return self._goal

    @property
    def agent_pos(self) -> Tuple[int, int]:
        return self._agent_pos

    def _validate_positions(self, start, goal):
        if self._grid.ndim != 2:
            raise ValueError("grid must be a 2D array")
        for name, pos in (("start", start), ("goal", goal)):
            if not (0 <= pos[0] < self._grid.shape[0] and 0 <= pos[1] < self._grid.shape[1]):
                raise ValueError(f"{name} position {pos} is out of bounds")
            if self._grid[pos] != 0:
                raise ValueError(f"{name} position {pos} must be on an empty cell")

    def _find_nearest_free_cell(self, pos: Tuple[int, int], name: str = "position") -> Tuple[int, int]:
        from collections import deque
        r, c = int(pos[0]), int(pos[1])
        if (0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1] and self._grid[r, c] == 0):
            return (r, c)
        queue = deque([(r, c)])
        visited = {(r, c)}
        while queue:
            cr, cc = queue.popleft()
            if (0 <= cr < self._grid.shape[0] and 0 <= cc < self._grid.shape[1] and self._grid[cr, cc] == 0):
                return (cr, cc)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        raise ValueError(f"Cannot find a free cell near {name} position {pos}")

    def _is_free(self, position: Tuple[int, int]) -> bool:
        r, c = position
        return (0 <= r < self._grid.shape[0] and
                0 <= c < self._grid.shape[1] and
                self._grid[r, c] == 0)

    def _move(self, action: int) -> Tuple[int, int]:
        r, c = self._agent_pos
        if action == 0:   return r-1, c
        if action == 1:   return r+1, c
        if action == 2:   return r, c-1
        if action == 3:   return r, c+1
        raise ValueError(f"Invalid action: {action}")

    def _get_observation(self) -> np.ndarray:
        return np.array(self._agent_pos, dtype=np.int32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        super().reset(seed=seed)
        self._agent_pos = self._start
        self.current_step = 0
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be in [0,1,2,3], got {action}")

        self.current_step += 1
        next_pos = self._move(action)
        reward = -1.0
        if self._is_free(next_pos):
            self._agent_pos = next_pos
        else:
            reward = -10.0

        done = False
        if self._agent_pos == self._goal:
            reward = 100.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, {
            "is_success": self._agent_pos == self._goal,
            "step": self.current_step,
        }

    def render(self, mode: str = "human") -> None:
        rows, cols = self._grid.shape
        symbols = np.full((rows, cols), " ", dtype='<U1')
        symbols[self._grid == 1] = "█"
        symbols[self._start] = "S"
        symbols[self._goal] = "G"
        symbols[self._agent_pos] = "A"
        print("+" + "-"*cols + "+")
        for row in symbols:
            print("|" + "".join(row) + "|")
        print("+" + "-"*cols + "+")

    def close(self) -> None:
        pass


def train_qlearning(env: GridWorldEnv, episodes: int = 1000, alpha: float = 0.1,
                    gamma: float = 0.9, epsilon: float = 0.1, epsilon_decay: float = 0.995,
                    min_epsilon: float = 0.01) -> Tuple[np.ndarray, List[float]]:
    state_shape = (env.observation_space.high[0] + 1,
                   env.observation_space.high[1] + 1,
                   env.action_space.n)
    q_table = np.zeros(state_shape, dtype=np.float32)
    episode_rewards: List[float] = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state[0], state[1]]))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            old_q = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] = old_q + alpha * (reward + gamma * next_max - old_q)
            state = next_state
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    return q_table, episode_rewards


def plan_with_qlearning(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
    episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2,
    max_steps: int = 300
) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]]]:
    env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
    q_table, _ = train_qlearning(env, episodes=episodes, alpha=alpha, gamma=gamma,
                                 epsilon=epsilon, epsilon_decay=0.998, min_epsilon=0.05)

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
    else:
        return None, frames