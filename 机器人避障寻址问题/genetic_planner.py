import numpy as np
import random
from collections import deque
from typing import List, Tuple, Set

class GAPathPlanner:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 pop_size: int = 200, max_gen: int = 300,
                 mutation_rate: float = 0.2, elite_size: int = 10):
        self.grid = np.asarray(grid, dtype=np.int8).copy()
        self.start = start
        self.goal = goal
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_path_len = 25

    def _random_point(self) -> Tuple[int, int]:
        rows, cols = self.grid.shape
        while True:
            point = (random.randint(0, rows-1), random.randint(0, cols-1))
            if self.grid[point] == 0:
                return point

    def _create_individual(self) -> List[Tuple[int, int]]:
        n_points = random.randint(2, self.max_path_len)
        indiv = [self.start]
        for _ in range(n_points):
            indiv.append(self._random_point())
        indiv.append(self.goal)
        return indiv

    def _bresenham(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2-x1)
        dy = abs(y2-y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        points = [(x, y)]
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
                points.append((x, y))
        else:
            err = dy / 2.0
            while y != y2:
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                points.append((x, y))
        return points

    def _fitness(self, path: List[Tuple[int, int]]) -> float:
        length = 0.0
        collision_penalty = 0.0
        for i in range(len(path)-1):
            p1, p2 = path[i], path[i+1]
            points = self._bresenham(p1, p2)
            for pt in points:
                if self.grid[pt] == 1:
                    collision_penalty += 500.0
            length += np.linalg.norm(np.array(p1)-np.array(p2))
        dist_to_goal = np.linalg.norm(np.array(path[-1])-np.array(self.goal))
        return 1.0 / (length + collision_penalty + 0.1*dist_to_goal + 1e-6)

    def _crossover(self, p1, p2):
        if len(p1) < 3 or len(p2) < 3:
            return p1[:], p2[:]
        cut1 = random.randint(1, len(p1)-2)
        cut2 = random.randint(1, len(p2)-2)
        child1 = p1[:cut1] + p2[cut2:]
        child2 = p2[:cut2] + p1[cut1:]
        child1[0], child1[-1] = self.start, self.goal
        child2[0], child2[-1] = self.start, self.goal
        return child1, child2

    def _mutate(self, indiv):
        if len(indiv) > 2 and random.random() < self.mutation_rate:
            idx = random.randint(1, len(indiv)-2)
            indiv[idx] = self._random_point()
        return indiv

    def _tournament_select(self, population, fitnesses, k=3):
        participants = random.sample(range(len(population)), min(k, len(population)))
        winner = max(participants, key=lambda idx: fitnesses[idx])
        return population[winner]

    def _bfs_path(self, start, goal):
        queue = deque([start])
        parent = {start: None}
        visited = {start}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (current[0]+dr, current[1]+dc)
                if (0 <= nb[0] < self.grid.shape[0] and
                    0 <= nb[1] < self.grid.shape[1] and
                    self.grid[nb] == 0 and nb not in visited):
                    visited.add(nb)
                    parent[nb] = current
                    queue.append(nb)
        if goal not in parent:
            return []
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def _connect_path(self, waypoints):
        full_path = []
        for i in range(len(waypoints)-1):
            segment = self._bfs_path(waypoints[i], waypoints[i+1])
            if not segment:
                return []
            if full_path:
                segment = segment[1:]
            full_path.extend(segment)
        return full_path

    def evolve(self):
        population = [self._create_individual() for _ in range(self.pop_size)]
        best_individual = population[0]
        best_fitness = -np.inf
        history = []
        for _ in range(self.max_gen):
            fitnesses = [self._fitness(ind) for ind in population]
            elite_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:self.elite_size]
            elites = [population[i][:] for i in elite_idx]
            max_fit = max(fitnesses)
            if max_fit > best_fitness:
                best_fitness = max_fit
                best_individual = population[int(np.argmax(fitnesses))][:]
            history.append(best_fitness)
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            population = new_pop
        best_path = self._connect_path(best_individual)
        if not best_path:
            best_path = best_individual[:]
        frames = []
        visited = set()
        for i, node in enumerate(best_path):
            visited.add(node)
            frontier = set()
            if i+1 < len(best_path):
                frontier.add(best_path[i+1])
            frames.append((node, visited.copy(), frontier.copy()))
        return best_path, history, frames