
# 手动实现 DBSCAN：RegionQuery + ExpandCluster（队列BFS扩展）
# labels: 0=未分配, -1=噪声, 1..K=簇ID
from __future__ import annotations
import numpy as np

NOISE = -1
UNASSIGNED = 0

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def get_neighbors(X: np.ndarray, idx: int, eps: float) -> list[int]:
    """
    RegionQuery：返回 eps 邻域内点索引（含自身）。
    论文定义通常用 dist <= eps，这里用 <= 与论文一致。
    """
    neighbors = []
    for j in range(len(X)):
        if euclidean_distance(X[idx], X[j]) <= eps:
            neighbors.append(j)
    return neighbors

def my_dbscan(X: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """
     DBSCAN 主过程：
    - visited 控制“是否已经做过邻域查询”
    - 先到先得：点一旦归属某簇，就不被后续簇覆盖（噪声点可被“招安”为边界点）
    """
    X = np.asarray(X, dtype=float)
    n = len(X)

    labels = np.full(n, UNASSIGNED, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = get_neighbors(X, i, eps)

        # 核心点判定：|N_eps(i)| >= MinPts
        if len(neighbors) < min_pts:
            labels[i] = NOISE
            continue

        # 创建新簇
        cluster_id += 1
        labels[i] = cluster_id

        # ExpandCluster：用队列BFS扩展，避免 neighbors.extend() 的重复膨胀
        queue = list(neighbors)
        in_queue = np.zeros(n, dtype=bool)
        in_queue[queue] = True

        q = 0
        while q < len(queue):
            p = queue[q]
            q += 1

            # 噪声点可被吸收为边界点
            if labels[p] == NOISE:
                labels[p] = cluster_id

            if not visited[p]:
                visited[p] = True
                p_neighbors = get_neighbors(X, p, eps)
                if len(p_neighbors) >= min_pts:
                    # 只有核心点才扩展其邻域
                    for r in p_neighbors:
                        if not in_queue[r]:
                            queue.append(r)
                            in_queue[r] = True

            # 未分配点加入当前簇
            if labels[p] == UNASSIGNED:
                labels[p] = cluster_id

    return labels
