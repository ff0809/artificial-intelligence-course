# main.py
import time
import numpy as np
from map_loader import get_map
from bfs_astar import bfs_search, a_star_search
from dijkstra import dijkstra_search
from artificial_potential_field import APFPlanner
from bug2_planner import Bug2Planner
from genetic_planner import GAPathPlanner
from rl_environment import plan_with_qlearning
from rl_methods import plan_with_sarsa, plan_with_double_q
from dqn_planner import plan_with_dqn
from visualize import animate_search, animate_path_with_frontier

MAP_OPTIONS = {
    '1': ('original', '原始地图'),
    '2': ('u_trap', 'U形陷阱地图'),
    '3': ('custom_maze', '自定义迷宫地图'),
}

def choose_map():
    print("请选择要加载的地图：")
    for key, (_, label) in MAP_OPTIONS.items():
        print(f"  {key}. {label}")
    choice = input("请输入地图编号(1-3，默认1)：").strip()
    selected = MAP_OPTIONS.get(choice, MAP_OPTIONS['1'])
    print(f"已选择：{selected[1]}\n")
    return selected[0]

def run_comparison(map_name='original'):
    grid, start, goal = get_map(map_name)
    print(f"地图尺寸: {grid.shape}, 起点: {start}, 终点: {goal}")

    algorithms = [
        ("BFS", lambda: bfs_search(grid, start, goal)),
        ("A*", lambda: a_star_search(grid, start, goal)),
        ("Dijkstra", lambda: dijkstra_search(grid, start, goal)),
        ("APF", lambda: APFPlanner(grid, start, goal).plan()),
        ("GA", lambda: GAPathPlanner(grid, start, goal, pop_size=200, max_gen=300).evolve()),
        ("Bug2", lambda: Bug2Planner(grid, start, goal).plan()),
        ("Q-learning", lambda: plan_with_qlearning(grid, start, goal, episodes=1000)),
        ("SARSA", lambda: plan_with_sarsa(grid, start, goal, episodes=1000)),
        ("Double Q", lambda: plan_with_double_q(grid, start, goal, episodes=1000)),
        ("DQN", lambda: plan_with_dqn(grid, start, goal, episodes=1000)),
    ]

    results = {}
    for name, func in algorithms:
        print(f"\n运行 {name} ...")
        t0 = time.time()
        if name == "GA":
            path, history, frames = func()
        else:
            path, frames = func()
        elapsed = time.time() - t0
        success = path is not None and (len(path) == 0 or path[-1] == goal)
        results[name] = {
            'success': success,
            'path_length': len(path)-1 if success else None,
            'time': elapsed
        }
        if success:
            print(f"{name} 成功，路径长度 {len(path)-1}，耗时 {elapsed:.2f}s")
            if name in ["BFS", "A*", "Dijkstra"]:
                animate_search(grid, start, goal, frames, name, path=path, interval=400)
            else:
                animate_path_with_frontier(grid, start, goal, frames, name, path=path, interval=400)
        else:
            print(f"{name} 失败，耗时 {elapsed:.2f}s")

    print("算法性能综合分析")
    print("\n1. 成功情况:")
    for algo, data in results.items():
        status = "成功" if data['success'] else "失败"
        print(f"   {algo:<12}: {status}")
    print("\n2. 路径长度与耗时:")
    for algo, data in results.items():
        if data['success']:
            print(f"   {algo:<12}: 长度 {data['path_length']}, 耗时 {data['time']:.3f}s")
        else:
            print(f"   {algo:<12}: 失败, 耗时 {data['time']:.3f}s")

if __name__ == "__main__":
    map_name = choose_map()
    run_comparison(map_name)