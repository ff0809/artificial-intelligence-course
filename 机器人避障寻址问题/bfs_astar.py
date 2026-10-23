import numpy as np
from collections import deque
import heapq
from typing import Dict, List, Optional, Tuple, Set


def get_neighbors(grid: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c = pos
    neighbors = []
    directions_orth = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions_orth:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(parent: Dict[Tuple[int, int], Tuple[int, int]],
                     current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path

def bfs_search(grid: np.ndarray,
               start: Tuple[int, int],
               end: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]]]:
    queue = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    frames: List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]] = []

    while queue:
        current = queue.popleft()
        frontier: Set[Tuple[int, int]] = set(queue)
        frames.append((current, visited.copy(), frontier.copy()))

        if current == end:
            return reconstruct_path(parent, current), frames

        for nb in get_neighbors(grid, current):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = current
                queue.append(nb)

    return None, frames

def heuristic(a: Tuple[int, int], b: Tuple[int, int], method: str = 'manhattan') -> float:
    if method == 'manhattan':
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    elif method == 'euclidean':
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5
    elif method == 'diagonal':
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
    else:
        raise ValueError(f"Unknown heuristic method: {method}")

def a_star_search(grid: np.ndarray,
                  start: Tuple[int, int],
                  end: Tuple[int, int],
                  heuristic_method: str = 'manhattan') -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]]]:
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, end, heuristic_method)}

    open_set_hash: Set[Tuple[int, int]] = {start}
    closed_set: Set[Tuple[int, int]] = set()
    frames: List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]] = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current not in open_set_hash:
            continue
        open_set_hash.remove(current)
        frames.append((current, closed_set.copy(), open_set_hash.copy()))

        if current == end:
            return reconstruct_path(came_from, current), frames

        closed_set.add(current)

        for nb in get_neighbors(grid, current):
            if nb in closed_set:
                continue

            tentative_g = g_score[current] + 1.0
            if nb not in open_set_hash or tentative_g < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score[nb] = tentative_g + heuristic(nb, end, heuristic_method)
                if nb not in open_set_hash:
                    heapq.heappush(open_set, (f_score[nb], nb))
                    open_set_hash.add(nb)

    return None, frames