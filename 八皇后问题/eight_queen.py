import itertools
import time
import random
from kanren import run, var, eq, conde, lall

# 1. 基础工具函数
def get_walk(v, s):
    while isinstance(v, var) and v in s:
        v = s[v]
    return v

def success(s): yield s
def fail(s): return

# 2. 稳定的成员关系
def membero_stable(item, collection):
    return conde(*[[eq(item, val)] for val in collection])

# 3. 棋盘可视化函数
def print_board(queens, title=""):
    if title:
        print(f"   [{title}]")
    print("   " + " ".join([str(i) for i in range(8)])) 
    print("  " + "-" * 17)
    for r in range(8):
        row_str = f"{r}|"
        for c in range(8):
            if queens[r] == c:
                row_str += " Q"
            else:
                row_str += " ."
        print(row_str + "|")
    print("  " + "-" * 17)

# 4. 核心约束逻辑（在part 2,3中使用）
def no_conflict_goal(idx1, idx2, q1, q2):
    def goal(s):
        v1, v2 = get_walk(q1, s), get_walk(q2, s)
        if not isinstance(v1, var) and not isinstance(v2, var):
            if v1 == v2 or abs(v1 - v2) == abs(idx1 - idx2):
                return 
        yield s
    return goal

# 穷举法 + 逻辑筛选
def solve_part1():
    all_perms = list(itertools.permutations(range(8)))
    q = var()
    
    def logic_diag_filter(q_var):
        def goal(s):
            queens = get_walk(q_var, s)
            if not isinstance(queens, var):
                for i in range(8):
                    for j in range(i + 1, 8):
                        if abs(queens[i] - queens[j]) == abs(i - j):
                            return
                yield s
        return goal

    results = []
    chunk_size = 5000
    for i in range(0, len(all_perms), chunk_size):
        chunk = all_perms[i:i + chunk_size]
        chunk_res = run(0, q, membero_stable(q, chunk), logic_diag_filter(q))
        results.extend(list(chunk_res))
    return results

# 纯逻辑编程 
def solve_part2():
    qs = [var() for _ in range(8)]
    domain = list(range(8))
    
    def build_goals(n):
        if n == 0:
            return [membero_stable(qs[0], domain)]
        current_goals = [membero_stable(qs[n], domain)]
        for j in range(n):
            current_goals.append(no_conflict_goal(n, j, qs[n], qs[j]))
        return current_goals

    all_goals = []
    for i in range(8):
        all_goals.extend(build_goals(i))
    
    return list(run(0, qs, lall(*all_goals)))

#  手动实现 neq 逻辑关系
def solve_part3():
    def my_neq(x, y):
        def goal(s):
            v1, v2 = get_walk(x, s), get_walk(y, s)
            if not isinstance(v1, var) and not isinstance(v2, var):
                if v1 == v2: return
            yield s
        return goal

    qs = [var() for _ in range(8)]
    domain = list(range(8))
    
    all_goals = []
    for i in range(8):
        all_goals.append(membero_stable(qs[i], domain))
        for j in range(i):
            all_goals.append(my_neq(qs[i], qs[j]))
            def diag_goal(idx1, idx2, q1, q2):
                def goal(s):
                    v1, v2 = get_walk(q1, s), get_walk(q2, s)
                    if not isinstance(v1, var) and not isinstance(v2, var):
                        if abs(v1 - v2) == abs(idx1 - idx2): return
                    yield s
                return goal
            all_goals.append(diag_goal(i, j, qs[i], qs[j]))
            
    return list(run(0, qs, lall(*all_goals)))

# DFS 算法
def solve_part4():
    solutions = []
    board = [-1] * 8
    def is_safe(row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True
    def backtrack(row):
        if row == 8:
            solutions.append(tuple(board))
            return
        for col in range(8):
            if is_safe(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1
    backtrack(0)
    return solutions

if __name__ == "__main__":
    print("="*25, "八皇后问题", "="*25)
    
    tasks = [
        ("Part 1: 穷举+逻辑筛选", solve_part1),
        ("Part 2: 纯逻辑编程", solve_part2),
        ("Part 3: 自定义 neq 求解", solve_part3),
        ("Part 4: DFS 算法", solve_part4)
    ]
    
    for name, func in tasks:
        print(f"\n[任务] {name} 正在执行 ... ", end="", flush=True)
        t_start = time.time()
        try:
            ans = func()
            t_end = time.time()
            print("完成！")
            print(f"  - 找到解总数: {len(ans)}")
            print(f"  - 消耗时间: {t_end - t_start:.4f} 秒")
            
            if len(ans) >= 3:
                sample_indices = random.sample(range(len(ans)), 3)
                print(f"  - 随机抽取3个示例展示:")
                for i, idx in enumerate(sample_indices):
                    print_board(ans[idx], title=f"示例 {i+1} (索引:{idx})")
            elif len(ans) > 0:
                print_board(ans[0], title="示例 1")
                
        except Exception as e:
            print(f"\n  [!!] 运行失败: {e}")
        print("-" * 60)