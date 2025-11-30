"""
Модуль для обхода графов.
Реализует BFS и DFS алгоритмы.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_representation import AdjacencyList


class GraphTraversal:
    """Класс для реализации алгоритмов обхода графов."""

    @staticmethod
    def bfs(graph: Any, start_vertex: str) -> Tuple[Dict[str, int],
                                                    Dict[str, Optional[str]]]:
        """
        Универсальный поиск в ширину для любого представления графа.

        Args:
            graph: Граф (матрица смежности или список смежности)
            start_vertex: Стартовая вершина

        Returns:
            Кортеж (расстояния, родители)

        Сложность: O(V + E) для списка смежности, O(V²) для матрицы
        """
        distances_result: Dict[str, int] = {}
        parents_result: Dict[str, Optional[str]] = {}
        visited: Set[str] = set()
        queue: deque[str] = deque()

        for vertex in graph.get_vertices():
            distances_result[vertex] = -1
            parents_result[vertex] = None

        distances_result[start_vertex] = 0
        queue.append(start_vertex)
        visited.add(start_vertex)

        while queue:
            current_vertex = queue.popleft()
            current_distance = distances_result[current_vertex]

            for neighbor in graph.get_adjacent_vertices(current_vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances_result[neighbor] = current_distance + 1
                    parents_result[neighbor] = current_vertex
                    queue.append(neighbor)

        return distances_result, parents_result

    @staticmethod
    def dfs_recursive(graph: Any, start_vertex: str) -> Dict[str, int]:
        """
        Универсальный рекурсивный поиск в глубину.

        Args:
            graph: Граф (матрица смежности или список смежности)
            start_vertex: Стартовая вершина

        Returns:
            Словарь с порядком обхода вершин

        Сложность: O(V + E) для списка смежности, O(V²) для матрицы
        """
        visited_nodes: Set[str] = set()
        discovery_order: Dict[str, int] = {}
        order_counter = [0]

        def dfs_visit(vertex: str) -> None:
            visited_nodes.add(vertex)
            discovery_order[vertex] = order_counter[0]
            order_counter[0] += 1

            for neighbor in graph.get_adjacent_vertices(vertex):
                if neighbor not in visited_nodes:
                    dfs_visit(neighbor)

        dfs_visit(start_vertex)
        return discovery_order

    @staticmethod
    def dfs_iterative(graph: Any, start_vertex: str) -> Dict[str, int]:
        """
        Универсальный итеративный поиск в глубину.

        Args:
            graph: Граф (матрица смежности или список смежности)
            start_vertex: Стартовая вершина

        Returns:
            Словарь с порядком обхода вершин

        Сложность: O(V + E) для списка смежности, O(V²) для матрицы
        """
        visited_nodes: Set[str] = set()
        discovery_order: Dict[str, int] = {}
        stack: List[str] = []
        order_counter = 0

        stack.append(start_vertex)

        while stack:
            current_vertex = stack.pop()

            if current_vertex not in visited_nodes:
                visited_nodes.add(current_vertex)
                discovery_order[current_vertex] = order_counter
                order_counter += 1

                neighbors = graph.get_adjacent_vertices(current_vertex)
                for neighbor in reversed(neighbors):
                    if neighbor not in visited_nodes:
                        stack.append(neighbor)

        return discovery_order

    @staticmethod
    def get_shortest_path(parents_dict: Dict[str, Optional[str]],
                          target: str) -> List[str]:
        """
        Восстановление кратчайшего пути из словаря родителей.

        Args:
            parents_dict: Словарь родительских вершин
            target: Целевая вершина

        Returns:
            Список вершин кратчайшего пути
        """
        if parents_dict.get(target) is None and target not in parents_dict:
            return []

        path: List[str] = []
        current_vertex = target

        while current_vertex is not None:
            path.append(current_vertex)
            current_vertex = parents_dict[current_vertex]

        path.reverse()
        return path

    @staticmethod
    def find_connected_components(graph: Any) -> List[List[str]]:
        """
        Универсальный поиск компонент связности для неориентированного графа.

        Args:
            graph: Неориентированный граф

        Returns:
            Список компонент связности

        Сложность: O(V + E) для списка смежности, O(V²) для матрицы
        """
        visited_nodes: Set[str] = set()
        components_result: List[List[str]] = []

        def bfs_component(start_vertex: str) -> List[str]:
            """BFS для поиска одной компоненты связности."""
            component_list: List[str] = []
            component_queue: deque[str] = deque([start_vertex])
            visited_nodes.add(start_vertex)

            while component_queue:
                current_vertex = component_queue.popleft()
                component_list.append(current_vertex)

                for neighbor in graph.get_adjacent_vertices(current_vertex):
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        component_queue.append(neighbor)

            return component_list

        for vertex in graph.get_vertices():
            if vertex not in visited_nodes:
                component = bfs_component(vertex)
                components_result.append(component)

        return components_result


if __name__ == '__main__':
    print('Демонстрация алгоритмов обхода')

    list_graph = AdjacencyList()
    list_graph.add_edge('A', 'B')
    list_graph.add_edge('B', 'C')
    list_graph.add_edge('C', 'D')
    list_graph.add_edge('A', 'E')
    list_graph.add_edge('E', 'F')

    print('Вершины:', list_graph.get_vertices())

    print('\nBFS от вершины A')
    distances_output, parents_output = GraphTraversal.bfs(list_graph, 'A')
    print('Расстояния:', distances_output)
    print('Родители:', parents_output)

    path_to_d = GraphTraversal.get_shortest_path(parents_output, 'D')
    print('Кратчайший путь к D:', path_to_d)

    print('\nDFS рекурсивный от A')
    dfs_recursive_output = GraphTraversal.dfs_recursive(list_graph, 'A')
    print('Порядок обхода:', dfs_recursive_output)

    print('\nDFS итеративный от A')
    dfs_iterative_output = GraphTraversal.dfs_iterative(list_graph, 'A')
    print('Порядок обхода:', dfs_iterative_output)

    print('\nКомпоненты связности')
    components_output = GraphTraversal.find_connected_components(list_graph)
    print('Компоненты:', components_output)
