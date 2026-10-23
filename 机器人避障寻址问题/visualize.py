import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Set, Tuple


def animate_search(grid: np.ndarray,
                   start: Tuple[int, int],
                   end: Tuple[int, int],
                   frames: List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]],
                   algo_name: str,
                   path: Optional[List[Tuple[int, int]]] = None,
                   save_gif: bool = False,
                   interval: int = 500,
                   fps: int = 10):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"{algo_name} Search Process")

    cmap = plt.cm.colors.ListedColormap(['white', 'royalblue', 'lightskyblue', 'gold', 'red', 'gray'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    display_map = grid.copy().astype(float)
    im = ax.imshow(display_map, cmap=cmap, norm=norm, origin='upper')

    ax.set_xlabel('col')
    ax.set_ylabel('row')
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='minor', length=0)

    ax.scatter(start[1], start[0], c='red', s=120, marker='o', edgecolors='black', label='Start')
    ax.scatter(end[1], end[0], c='gray', s=120, marker='s', edgecolors='black', label='End')
    current_marker = ax.scatter([], [], c='orange', s=120, marker='o', edgecolors='black', linewidths=1, zorder=5)

    text_info = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame_idx: int):
        if frames is None:
            return im, text_info

        if frame_idx < len(frames):
            current, visited, frontier = frames[frame_idx]
            layer = grid.copy().astype(float)
            for v in visited:
                if v != start and v != end:
                    layer[v] = 2
            for f in frontier:
                if f != start and f != end:
                    layer[f] = 3
            layer[start] = 4
            layer[end] = 5

            im.set_data(layer)
            if current != start and current != end:
                current_marker.set_offsets([[current[1], current[0]]])
                current_marker.set_visible(True)
            else:
                current_marker.set_visible(False)
            text_info.set_text(f"Step: {frame_idx+1}/{len(frames)}\n"
                               f"Current (row, col): {current}\n"
                               f"Visited: {len(visited)}\n"
                               f"Frontier: {len(frontier)}")
        else:
            layer = grid.copy().astype(float)
            if path is not None:
                for p in path[1:-1]:
                    layer[p] = 2
            layer[start] = 4
            layer[end] = 5
            im.set_data(layer)
            if path is not None:
                text_info.set_text(f"Path Found! Length: {len(path)-1:.2f}\nPath Nodes: {len(path)}")
            else:
                text_info.set_text("No path found.")
            current_marker.set_visible(False)

        return im, text_info

    total_frames = len(frames) + (1 if path is not None else 0)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval,
                                  blit=False, repeat=False)

    if save_gif:
        safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in algo_name)
        filename = f"{safe_name}_search.gif"
        ani.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved as {filename}")
    else:
        plt.show()

    return ani


def animate_path_with_frontier(grid: np.ndarray,
                 start: Tuple[int, int],
                 end: Tuple[int, int],
                 frames: List[Tuple[Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]],
                 algo_name: str,
                 path: Optional[List[Tuple[int, int]]] = None,
                 save_gif: bool = False,
                 interval: int = 400,
                 fps: int = 10):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"{algo_name} Path Building Process")

    cmap = plt.cm.colors.ListedColormap(['white', 'royalblue', 'lightskyblue', 'gold', 'red', 'gray'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    display_map = grid.copy().astype(float)
    im = ax.imshow(display_map, cmap=cmap, norm=norm, origin='upper')

    ax.set_xlabel('col')
    ax.set_ylabel('row')
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='minor', length=0)

    ax.scatter(start[1], start[0], c='red', s=120, marker='o', edgecolors='black', label='Start')
    ax.scatter(end[1], end[0], c='gray', s=120, marker='s', edgecolors='black', label='End')
    current_marker = ax.scatter([], [], c='yellow', s=120, marker='o', edgecolors='black', linewidths=1, zorder=5)

    text_info = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update(frame_idx: int):
        if frame_idx < len(frames):
            current, visited, frontier = frames[frame_idx]
            layer = grid.copy().astype(float)
            for v in visited:
                if v != start and v != end:
                    layer[v] = 2
            for f in frontier:
                if f != start and f != end:
                    layer[f] = 3
            layer[start] = 4
            layer[end] = 5

            im.set_data(layer)
            if current != start and current != end:
                current_marker.set_offsets([[current[1], current[0]]])
                current_marker.set_visible(True)
            else:
                current_marker.set_visible(False)
            text_info.set_text(f"Step: {frame_idx+1}/{len(frames)}\n"
                               f"Current: {current}\n"
                               f"Visited: {len(visited)}, Frontier: {len(frontier)}")
        else:
            layer = grid.copy().astype(float)
            if path is not None:
                for p in path[1:-1]:
                    layer[p] = 2
            layer[start] = 4
            layer[end] = 5
            im.set_data(layer)
            if path is not None:
                text_info.set_text(f"Path Found! Length: {len(path)-1}\nPath Nodes: {len(path)}")
            else:
                text_info.set_text("Path construction complete.")
            current_marker.set_visible(False)

        return im, text_info

    total_frames = len(frames) + 1
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval,
                                  blit=False, repeat=False)

    if save_gif:
        safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in algo_name)
        filename = f"{safe_name}_path.gif"
        ani.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved as {filename}")
    else:
        plt.show()

    return ani



def draw_final_path(grid: np.ndarray,
                    start: Tuple[int, int],
                    end: Tuple[int, int],
                    path: List[Tuple[int, int]],
                    algo_name: str,
                    save_png: bool = False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"{algo_name} Final Path")

    cmap = plt.cm.colors.ListedColormap(['white', 'royalblue', 'lightskyblue', 'gold', 'red', 'gray'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    display_map = grid.copy().astype(float)
    for p in path[1:-1]:
        display_map[p] = 2
    display_map[start] = 4
    display_map[end] = 5

    im = ax.imshow(display_map, cmap=cmap, norm=norm, origin='upper')
    ax.set_xlabel('col')
    ax.set_ylabel('row')
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='minor', length=0)

    ax.scatter(start[1], start[0], c='red', s=120, marker='o', edgecolors='black', label='Start')
    ax.scatter(end[1], end[0], c='gray', s=120, marker='s', edgecolors='black', label='End')
    ax.legend(loc='upper right')
    if save_png:
        safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in algo_name)
        filename = f"{safe_name}_path.png"
        fig.savefig(filename, dpi=150)
        print(f"Path image saved as {filename}")
    plt.show()

    return fig
