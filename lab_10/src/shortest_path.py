"""
Модуль для алгоритмов на графах.
Реализует алгоритм Дейкстры и топологическую сортировку.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple

from graph_representation import AdjacencyList
from graph_traversal import GraphTraversal


class GraphAlgorithms:
    """Класс для реализации алгоритмов на графах."""

    @staticmethod
    def dijkstra_adjacency_list(
        graph: AdjacencyList, start_vertex: str
    ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Алгоритм Дейкстры для поиска кратчайших путей во взвешенном графе.

        Args:
            graph: Взвешенный граф в виде списка смежности
            start_vertex: Стартовая вершина

        Returns:
            Кортеж (расстояния, родители)

        Сложность: O((V + E) log V)
        """
        distances: Dict[str, float] = {}
        parents: Dict[str, Optional[str]] = {}

        for vertex in graph.get_vertices():
            distances[vertex] = float('inf')
            parents[vertex] = None

        distances[start_vertex] = 0

        priority_queue: List[Tuple[float, str]] = []
        heapq.heappush(priority_queue, (0, start_vertex))

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor in graph.get_adjacent_vertices(current_vertex):
                edge_weight = graph.get_edge_weight(current_vertex, neighbor)
                if edge_weight is None:
                    continue

                new_distance = current_distance + edge_weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        return distances, parents

    @staticmethod
    def topological_sort_adjacency_list(graph: AdjacencyList) -> List[str]:
        """
        Топологическая сортировка для ориентированного ациклического графа.

        Args:
            graph: Ориентированный граф в виде списка смежности

        Returns:
            Топологически упорядоченный список вершин

        Сложность: O(V + E)
        """
        if not graph.directed:
            raise ValueError(
                'Данная сортировка применима только к ориентированным графам'
            )

        in_degree: Dict[str, int] = {}
        result: List[str] = []
        queue: List[str] = []

        for vertex in graph.get_vertices():
            in_degree[vertex] = 0

        for vertex in graph.get_vertices():
            for neighbor in graph.get_adjacent_vertices(vertex):
                in_degree[neighbor] += 1

        for vertex, degree in in_degree.items():
            if degree == 0:
                queue.append(vertex)

        while queue:
            vertex = queue.pop(0)
            result.append(vertex)

            for neighbor in graph.get_adjacent_vertices(vertex):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(graph.get_vertices()):
            raise ValueError(
                'Граф содержит циклы, топологическая сортировка невозможна'
            )

        return result

    @staticmethod
    def solve_maze_shortest_path(
        maze_grid: List[List[int]], start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Поиск кратчайшего пути в лабиринте с использованием BFS.

        Args:
            maze_grid: Матрица лабиринта (0 - проходимо, 1 - стена)
            start: Начальная позиция (row, col)
            end: Конечная позиция (row, col)

        Returns:
            Список позиций кратчайшего пути или None, если путь не найден
        """
        rows = len(maze_grid)
        if rows == 0:
            return None
        cols = len(maze_grid[0])

        start_invalid = (start[0] < 0 or start[0] >= rows or
                         start[1] < 0 or start[1] >= cols)
        end_invalid = (end[0] < 0 or end[0] >= rows or
                       end[1] < 0 or end[1] >= cols)

        if start_invalid or end_invalid:
            return None

        if (maze_grid[start[0]][start[1]] == 1 or
                maze_grid[end[0]][end[1]] == 1):
            return None

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        queue: List[Tuple[int, int]] = [start]
        visited: Set[Tuple[int, int]] = {start}
        parents_dict: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {
            start: None
        }

        while queue:
            current = queue.pop(0)

            if current == end:
                path: List[Tuple[int, int]] = []
                current_pos = current
                while current_pos is not None:
                    path.append(current_pos)
                    current_pos = parents_dict[current_pos]
                path.reverse()
                return path

            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                valid_position = (0 <= nr < rows and 0 <= nc < cols)
                if (valid_position and maze_grid[nr][nc] == 0 and
                        neighbor not in visited):
                    visited.add(neighbor)
                    parents_dict[neighbor] = current
                    queue.append(neighbor)

        return None


if __name__ == '__main__':
    print('Демонстрация алгоритмов на графах')

    print('\nАлгоритм Дейкстры')
    weighted_graph = AdjacencyList()
    weighted_graph.add_edge('A', 'B', 4)
    weighted_graph.add_edge('A', 'C', 2)
    weighted_graph.add_edge('B', 'C', 1)
    weighted_graph.add_edge('B', 'D', 5)
    weighted_graph.add_edge('C', 'D', 8)
    weighted_graph.add_edge('C', 'E', 10)
    weighted_graph.add_edge('D', 'E', 2)

    dijkstra_results = GraphAlgorithms.dijkstra_adjacency_list(
        weighted_graph, 'A'
    )
    dijkstra_distances, dijkstra_parents = dijkstra_results
    print('Расстояния Дейкстры:', dijkstra_distances)
    print('Родители:', dijkstra_parents)

    path_to_e = GraphTraversal.get_shortest_path(dijkstra_parents, 'E')
    print('Кратчайший путь к E:', path_to_e)

    print('\nТопологическая сортировка')
    dag = AdjacencyList(directed=True)
    dag.add_edge('A', 'B')
    dag.add_edge('A', 'C')
    dag.add_edge('B', 'D')
    dag.add_edge('C', 'D')
    dag.add_edge('D', 'E')

    try:
        topological_order = GraphAlgorithms.topological_sort_adjacency_list(
            dag
        )
        print('Топологический порядок:', topological_order)
    except ValueError as e:
        print(f'Ошибка: {e}')

    print('\nРешение лабиринта')
    maze_example = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]

    start_pos = (0, 0)
    end_pos = (4, 4)

    maze_path = GraphAlgorithms.solve_maze_shortest_path(
        maze_example, start_pos, end_pos
    )
    maze_height = len(maze_example)
    maze_width = len(maze_example[0])
    print(f'Лабиринт {maze_height}x{maze_width}')
    print(f'Путь от {start_pos} до {end_pos}: {maze_path}')
