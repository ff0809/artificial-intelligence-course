import numpy as np
from typing import List, Tuple, Set, Optional

class Bug2Planner:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start, self.goal = start, goal
        self.A = goal[0] - start[0]
        self.B = start[1] - goal[1]
        self.C = goal[1]*start[0] - goal[0]*start[1]
        self.denom = np.sqrt(self.A**2 + self.B**2)

    def _on_m_line(self, pos: Tuple[int, int]) -> bool:
        if self.denom < 1e-6:
            return True
        dist = abs(self.A*pos[1] + self.B*pos[0] + self.C) / self.denom
        return dist < 0.5

    def plan(self):
        curr = self.start
        path = [curr]
        visited = {curr}
        frames = []
        mode = "MTG"
        hit_point = None
        max_steps = 2000
        steps = 0

        while steps < max_steps:
            if curr == self.goal:
                return path, frames

            if mode == "MTG":
                dr = self.goal[0] - curr[0]
                dc = self.goal[1] - curr[1]
                if abs(dr) >= abs(dc):
                    step = (1 if dr > 0 else -1, 0)
                else:
                    step = (0, 1 if dc > 0 else -1)
                next_cell = (curr[0] + step[0], curr[1] + step[1])
                if (0 <= next_cell[0] < self.grid.shape[0] and
                    0 <= next_cell[1] < self.grid.shape[1] and
                    self.grid[next_cell] == 0):
                    curr = next_cell
                    path.append(curr)
                    visited.add(curr)
                    frames.append((curr, visited.copy(), set()))
                else:
                    mode = "WF"
                    hit_point = curr
            else:
                neighbors = []
                for dr, dc in [(-1,0),(0,1),(1,0),(0,-1)]:
                    nb = (curr[0]+dr, curr[1]+dc)
                    if (0 <= nb[0] < self.grid.shape[0] and
                        0 <= nb[1] < self.grid.shape[1] and
                        self.grid[nb] == 0):
                        neighbors.append(nb)
                if not neighbors:
                    return None, frames
                unvisited = [n for n in neighbors if n not in visited]
                next_cell = unvisited[0] if unvisited else neighbors[0]
                curr = next_cell
                path.append(curr)
                visited.add(curr)
                frames.append((curr, visited.copy(), set()))
                if hit_point is not None and self._on_m_line(curr):
                    dist_curr = np.linalg.norm(np.array(curr) - np.array(self.goal))
                    dist_hit = np.linalg.norm(np.array(hit_point) - np.array(self.goal))
                    if dist_curr < dist_hit - 0.5:
                        mode = "MTG"
                        hit_point = None
            steps += 1
        return None, frames