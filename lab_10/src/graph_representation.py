"""
Модуль для представления графов в памяти.
Реализует матрицу смежности и список смежности.
"""

import sys
from typing import Dict, List, Optional


class AdjacencyMatrix:
    """Класс для представления графа в виде матрицы смежности."""

    def __init__(self, directed: bool = False) -> None:
        """
        Инициализация матрицы смежности.

        Args:
            directed: Ориентированный граф (по умолчанию неориентированный)
        """
        self.matrix: List[List[int]] = []
        self.vertices: Dict[str, int] = {}
        self.vertices_list: List[str] = []
        self.directed = directed
        self.vertex_count = 0

    def add_vertex(self, vertex: str) -> None:
        """
        Добавление вершины в граф.

        Args:
            vertex: Имя вершины

        Сложность: O(V²) в худшем случае (при расширении матрицы)
        """
        if vertex not in self.vertices:
            self.vertices[vertex] = self.vertex_count
            self.vertices_list.append(vertex)
            self.vertex_count += 1

            for row in self.matrix:
                row.append(0)
            self.matrix.append([0] * self.vertex_count)

    def remove_vertex(self, vertex: str) -> None:
        """
        Удаление вершины из графа.

        Args:
            vertex: Имя вершины

        Сложность: O(V²) (перестроение матрицы)
        """
        if vertex in self.vertices:
            idx = self.vertices[vertex]

            del self.matrix[idx]
            for row in self.matrix:
                del row[idx]

            del self.vertices[vertex]
            self.vertices_list.remove(vertex)

            self.vertices = {v: i for i, v in enumerate(self.vertices_list)}
            self.vertex_count -= 1

    def add_edge(self, vertex1: str, vertex2: str, weight: int = 1) -> None:
        """
        Добавление ребра между вершинами.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина
            weight: Вес ребра (по умолчанию 1)

        Сложность: O(1) после инициализации вершин
        """
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)

        idx1 = self.vertices[vertex1]
        idx2 = self.vertices[vertex2]

        self.matrix[idx1][idx2] = weight
        if not self.directed:
            self.matrix[idx2][idx1] = weight

    def remove_edge(self, vertex1: str, vertex2: str) -> None:
        """
        Удаление ребра между вершинами.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина

        Сложность: O(1)
        """
        if vertex1 in self.vertices and vertex2 in self.vertices:
            idx1 = self.vertices[vertex1]
            idx2 = self.vertices[vertex2]

            self.matrix[idx1][idx2] = 0
            if not self.directed:
                self.matrix[idx2][idx1] = 0

    def get_adjacent_vertices(self, vertex: str) -> List[str]:
        """
        Получение списка смежных вершин.

        Args:
            vertex: Исходная вершина

        Returns:
            Список смежных вершин

        Сложность: O(V)
        """
        if vertex not in self.vertices:
            return []

        idx = self.vertices[vertex]
        adjacent = []

        for i, weight in enumerate(self.matrix[idx]):
            if weight != 0:
                adjacent.append(self.vertices_list[i])

        return adjacent

    def has_edge(self, vertex1: str, vertex2: str) -> bool:
        """
        Проверка наличия ребра между вершинами.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина

        Returns:
            True, если ребро существует, иначе False

        Сложность: O(1)
        """
        if vertex1 in self.vertices and vertex2 in self.vertices:
            idx1 = self.vertices[vertex1]
            idx2 = self.vertices[vertex2]
            return self.matrix[idx1][idx2] != 0
        return False

    def get_edge_weight(self, vertex1: str, vertex2: str) -> Optional[int]:
        """
        Получение веса ребра.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина

        Returns:
            Вес ребра или None, если ребра нет
        """
        if self.has_edge(vertex1, vertex2):
            idx1 = self.vertices[vertex1]
            idx2 = self.vertices[vertex2]
            return self.matrix[idx1][idx2]
        return None

    def get_vertices(self) -> List[str]:
        """
        Получение списка всех вершин.

        Returns:
            Список вершин
        """
        return self.vertices_list.copy()

    def get_memory_usage(self) -> int:
        """
        Оценка потребления памяти в байтах.

        Returns:
            Примерный объем памяти в байтах
        """
        matrix_size = sys.getsizeof(self.matrix) + sum(
            sys.getsizeof(row) for row in self.matrix
        )
        vertices_size = sys.getsizeof(self.vertices) + sum(
            sys.getsizeof(k) + sys.getsizeof(v)
            for k, v in self.vertices.items()
        )
        vertices_list_size = sys.getsizeof(self.vertices_list) + sum(
            sys.getsizeof(v) for v in self.vertices_list
        )
        return matrix_size + vertices_size + vertices_list_size


class AdjacencyList:
    """Класс для представления графа в виде списка смежности."""

    def __init__(self, directed: bool = False) -> None:
        """
        Инициализация списка смежности.

        Args:
            directed: Ориентированный граф (по умолчанию неориентированный)
        """
        self.graph: Dict[str, Dict[str, int]] = {}
        self.directed = directed

    def add_vertex(self, vertex: str) -> None:
        """
        Добавление вершины в граф.

        Args:
            vertex: Имя вершины

        Сложность: O(1)
        """
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def remove_vertex(self, vertex: str) -> None:
        """
        Удаление вершины из графа.

        Args:
            vertex: Имя вершины

        Сложность: O(V + E) в худшем случае
        """
        if vertex in self.graph:
            if not self.directed:
                for neighbor in self.graph[vertex]:
                    if vertex in self.graph[neighbor]:
                        del self.graph[neighbor][vertex]
            else:
                for v in self.graph:
                    if vertex in self.graph[v]:
                        del self.graph[v][vertex]

            del self.graph[vertex]

    def add_edge(self, vertex1: str, vertex2: str, weight: int = 1) -> None:
        """
        Добавление ребра между вершинами.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина
            weight: Вес ребра (по умолчанию 1)

        Сложность: O(1)
        """
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)

        self.graph[vertex1][vertex2] = weight
        if not self.directed:
            self.graph[vertex2][vertex1] = weight

    def remove_edge(self, vertex1: str, vertex2: str) -> None:
        """
        Удаление ребра между вершинами.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина

        Сложность: O(1)
        """
        if vertex1 in self.graph and vertex2 in self.graph[vertex1]:
            del self.graph[vertex1][vertex2]

            if not self.directed:
                if vertex2 in self.graph and vertex1 in self.graph[vertex2]:
                    del self.graph[vertex2][vertex1]

    def get_adjacent_vertices(self, vertex: str) -> List[str]:
        """
        Получение списка смежных вершин.

        Args:
            vertex: Исходная вершина

        Returns:
            Список смежных вершин

        Сложность: O(deg(v)) - степень вершины
        """
        if vertex in self.graph:
            return list(self.graph[vertex].keys())
        return []

    def has_edge(self, vertex1: str, vertex2: str) -> bool:
        """
        Проверка наличия ребра между вершинами.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина

        Returns:
            True, если ребро существует, иначе False

        Сложность: O(1) в среднем случае
        """
        return (vertex1 in self.graph and
                vertex2 in self.graph[vertex1])

    def get_edge_weight(self, vertex1: str, vertex2: str) -> Optional[int]:
        """
        Получение веса ребра.

        Args:
            vertex1: Первая вершина
            vertex2: Вторая вершина

        Returns:
            Вес ребра или None, если ребра нет
        """
        if self.has_edge(vertex1, vertex2):
            return self.graph[vertex1][vertex2]
        return None

    def get_vertices(self) -> List[str]:
        """
        Получение списка всех вершин.

        Returns:
            Список вершин
        """
        return list(self.graph.keys())

    def get_memory_usage(self) -> int:
        """
        Оценка потребления памяти в байтах.

        Returns:
            Примерный объем памяти в байтах
        """
        total_size = sys.getsizeof(self.graph)
        for vertex, neighbors in self.graph.items():
            total_size += sys.getsizeof(vertex)
            total_size += sys.getsizeof(neighbors)
            for neighbor, weight in neighbors.items():
                total_size += sys.getsizeof(neighbor) + sys.getsizeof(weight)
        return total_size


if __name__ == '__main__':
    print('Матрица смежности')
    matrix_graph = AdjacencyMatrix()
    matrix_graph.add_edge('A', 'B')
    matrix_graph.add_edge('B', 'C')
    matrix_graph.add_edge('A', 'C')

    print('Вершины:', matrix_graph.get_vertices())
    print('Смежные с A:', matrix_graph.get_adjacent_vertices('A'))
    print('Память:', matrix_graph.get_memory_usage(), 'байт')

    print('\nСписок смежности')
    list_graph = AdjacencyList()
    list_graph.add_edge('A', 'B')
    list_graph.add_edge('B', 'C')
    list_graph.add_edge('A', 'C')

    print('Вершины:', list_graph.get_vertices())
    print('Смежные с A:', list_graph.get_adjacent_vertices('A'))
    print('Память:', list_graph.get_memory_usage(), 'байт')
