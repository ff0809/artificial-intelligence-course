# dijkstra.py
import heapq
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


def reconstruct_path(parent: Dict[Tuple[int, int], Tuple[int, int]],
                     current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def dijkstra_search(grid: np.ndarray,
                    start: Tuple[int, int],
                    end: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]]]:
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed_set = set()
    frames = []

    while open_set:
        current_g, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        frames.append((current, closed_set.copy(), {item[1] for item in open_set}))

        if current == end:
            return reconstruct_path(came_from, current), frames

        closed_set.add(current)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                neighbor = (nr, nc)
                if neighbor in closed_set:
                    continue
                tentative_g = current_g + 1.0
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g, neighbor))

    return None, frames