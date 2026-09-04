from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphInfo:
    city: str
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray
    node_count: int
    edge_count: int
    connected_components: int


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def pairwise_haversine(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def build_knn_adjacency(coords: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n == 1:
        return np.ones((1, 1), dtype=np.float32)

    distances = pairwise_haversine(coords)
    adjacency = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        sorted_indices = np.argsort(distances[i])
        neighbors = [idx for idx in sorted_indices if idx != i][: min(k, n - 1)]
        adjacency[i, neighbors] = 1.0

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    degree = np.where(degree == 0, 1.0, degree)
    d_inv_sqrt = np.diag(np.power(degree, -0.5))
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


def count_connected_components(adjacency: np.ndarray) -> int:
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    components = 0

    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.where(adjacency[node] > 0)[0]
            for nxt in neighbors:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))

    return int(components)


def build_city_graph(city: str, coords: np.ndarray, k_neighbors: int) -> GraphInfo:
    adjacency = build_knn_adjacency(coords, k=k_neighbors)
    normalized = normalize_adjacency(adjacency)
    components = count_connected_components(adjacency)
    node_count = int(adjacency.shape[0])
    edge_count = int((adjacency.sum() - np.trace(adjacency)) / 2)

    return GraphInfo(
        city=city,
        adjacency=adjacency.astype(np.float32),
        normalized_adjacency=normalized.astype(np.float32),
        node_count=node_count,
        edge_count=edge_count,
        connected_components=components,
    )
