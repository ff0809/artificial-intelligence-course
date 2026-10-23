# artificial_potential_field.py
import numpy as np
from scipy.ndimage import distance_transform_edt, sobel
from typing import List, Tuple, Set, Optional

class APFPlanner:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 eta: float = 40.0, xi: float = 5.0, rho0: float = 3.0,
                 step_size: float = 0.5, max_iter: int = 500):
        self.grid = grid.astype(np.int8)
        self.dist_to_obs = distance_transform_edt(1 - (self.grid == 1))
        self.grad_row = sobel(self.dist_to_obs, axis=0)
        self.grad_col = sobel(self.dist_to_obs, axis=1)
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.eta, self.xi, self.rho0 = eta, xi, rho0
        self.step = step_size
        self.max_iter = max_iter

    def _get_force(self, pos: np.ndarray) -> np.ndarray:
        diff = self.goal - pos
        dist_g = np.linalg.norm(diff)
        f_att = self.xi * diff / (dist_g + 0.1)
        f_rep = np.zeros(2)
        r, c = int(round(pos[0])), int(round(pos[1]))
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            d = self.dist_to_obs[r, c]
            if 0 < d < self.rho0:
                grad = np.array([self.grad_row[r, c], self.grad_col[r, c]])
                mag = self.eta * (1.0/d - 1.0/self.rho0) * (1.0/(d**2))
                f_rep = mag * (grad / (np.linalg.norm(grad) + 1e-6))
        return f_att + f_rep

    def plan(self):
        pos = self.start.copy()
        path = [tuple(pos.astype(int))]
        visited = {path[0]}
        frames = []
        recent_pos = []
        stuck_counter = 0

        for _ in range(self.max_iter):
            curr_cell = (int(round(pos[0])), int(round(pos[1])))
            if np.linalg.norm(pos - self.goal) < 0.8:
                if path[-1] != tuple(self.goal.astype(int)):
                    path.append(tuple(self.goal.astype(int)))
                return path, frames

            force = self._get_force(pos)

            recent_pos.append(curr_cell)
            if len(recent_pos) > 10:
                recent_pos.pop(0)
                if len(set(recent_pos)) < 3:
                    return None, frames

            mag = np.linalg.norm(force)
            if mag > 1e-6:
                direction = force / mag
                if abs(direction[0]) > abs(direction[1]):
                    dr = 1 if direction[0] > 0 else -1
                    dc = 0
                else:
                    dr = 0
                    dc = 1 if direction[1] > 0 else -1
                new_pos = pos + np.array([dr, dc]) * self.step
            else:
                new_pos = pos

            new_cell = (int(round(new_pos[0])), int(round(new_pos[1])))
            if not (0 <= new_cell[0] < self.grid.shape[0] and 0 <= new_cell[1] < self.grid.shape[1]) \
               or self.grid[new_cell] == 1:
                stuck_counter += 1
                if stuck_counter > 5:
                    return None, frames
                force += np.random.uniform(-2, 2, 2)
                continue
            else:
                stuck_counter = 0

            pos = new_pos
            if new_cell != path[-1]:
                path.append(new_cell)
                visited.add(new_cell)
            frames.append((new_cell, visited.copy(), set()))

        return None, frames