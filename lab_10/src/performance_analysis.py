"""
Модуль для анализа производительности и визуализации результатов.
"""

import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from graph_representation import AdjacencyList, AdjacencyMatrix
from graph_traversal import GraphTraversal
from shortest_path import GraphAlgorithms


class GraphAnalyzer:
    """Класс для анализа производительности алгоритмов на графах."""

    @staticmethod
    def system_info() -> None:
        """Вывод информации о системе."""
        print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i5-13420H (2.10 GHz)
- Оперативная память: 16 GB DDR5
- ОС: Windows 11
- Python: 3.11
''')

    @staticmethod
    def generate_random_graph(
        vertex_count: int, edge_probability: float = 0.3,
        directed: bool = False, weighted: bool = False
    ) -> Tuple[AdjacencyMatrix, AdjacencyList]:
        """
        Генерация случайного графа.

        Args:
            vertex_count: Количество вершин
            edge_probability: Вероятность создания ребра
            directed: Ориентированный граф
            weighted: Взвешенный граф

        Returns:
            Кортеж (матрица смежности, список смежности)
        """
        matrix = AdjacencyMatrix(directed)
        adj_list = AdjacencyList(directed)

        vertices = [str(i) for i in range(vertex_count)]
        for v in vertices:
            matrix.add_vertex(v)
            adj_list.add_vertex(v)

        for i in range(vertex_count):
            for j in range(vertex_count):
                if i != j and random.random() < edge_probability:
                    weight = random.randint(1, 10) if weighted else 1
                    matrix.add_edge(vertices[i], vertices[j], weight)
                    adj_list.add_edge(vertices[i], vertices[j], weight)

        return matrix, adj_list

    @staticmethod
    def benchmark_algorithm(func, *args, iterations: int = 5) -> float:
        """
        Бенчмарк алгоритма с усреднением по нескольким запускам.

        Args:
            func: Функция для тестирования
            *args: Аргументы функции
            iterations: Количество запусков для усреднения

        Returns:
            Среднее время выполнения в секундах
        """
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return sum(times) / len(times)

    @staticmethod
    def measure_operations_performance() -> Dict[str, List[float]]:
        """
        Измерение производительности операций для разных представлений.

        Returns:
            Словарь с результатами измерений
        """
        graph_sizes = [100, 200, 400, 600, 800]
        performance_results = {
            'matrix_memory': [],
            'list_memory': [],
            'matrix_bfs_time': [],
            'list_bfs_time': [],
            'matrix_dfs_time': [],
            'list_dfs_time': []
        }

        for size in graph_sizes:
            matrix, adj_list = GraphAnalyzer.generate_random_graph(size, 0.1)

            performance_results['matrix_memory'].append(
                matrix.get_memory_usage() / 1024
            )
            performance_results['list_memory'].append(
                adj_list.get_memory_usage() / 1024
            )

            performance_results['matrix_bfs_time'].append(
                GraphAnalyzer.benchmark_algorithm(
                    GraphTraversal.bfs, matrix, '0'
                )
            )
            performance_results['list_bfs_time'].append(
                GraphAnalyzer.benchmark_algorithm(
                    GraphTraversal.bfs, adj_list, '0'
                )
            )

            performance_results['matrix_dfs_time'].append(
                GraphAnalyzer.benchmark_algorithm(
                    GraphTraversal.dfs_iterative, matrix, '0'
                )
            )
            performance_results['list_dfs_time'].append(
                GraphAnalyzer.benchmark_algorithm(
                    GraphTraversal.dfs_iterative, adj_list, '0'
                )
            )

        return performance_results

    @staticmethod
    def print_performance_table(
        results_data: Dict[str, List[float]], sizes_list: List[int]
    ) -> None:
        """Вывод результатов в виде таблицы."""
        print(f'{"Размер":>8} | {"Память (KB)":^21} | '
              f'{"BFS время (сек)":^21} | {"DFS время (сек)":^21}')
        print(f'{"":>8} | {"Матрица":>10} {"Список":>10} | '
              f'{"Матрица":>10} {"Список":>10} | '
              f'{"Матрица":>10} {"Список":>10}')
        print('-' * 81)

        for i, current_size in enumerate(sizes_list):
            memory_matrix = results_data['matrix_memory'][i]
            memory_list = results_data['list_memory'][i]
            bfs_matrix = results_data['matrix_bfs_time'][i]
            bfs_list = results_data['list_bfs_time'][i]
            dfs_matrix = results_data['matrix_dfs_time'][i]
            dfs_list = results_data['list_dfs_time'][i]

            print(f'{current_size:8} | '
                  f'{memory_matrix:10.2f} {memory_list:10.2f} | '
                  f'{bfs_matrix:10.6f} {bfs_list:10.6f} | '
                  f'{dfs_matrix:10.6f} {dfs_list:10.6f}')

    @staticmethod
    def plot_performance_comparison(
        results_data: Dict[str, List[float]], sizes_list: List[int]
    ) -> None:
        """
        Построение графиков сравнения производительности.

        Args:
            results_data: Результаты измерений
            sizes_list: Размеры графов
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        ax1.plot(sizes_list, results_data['matrix_memory'], 'r-',
                 label='Матрица смежности', marker='o', linewidth=2)
        ax1.plot(sizes_list, results_data['list_memory'], 'b-',
                 label='Список смежности', marker='s', linewidth=2)
        ax1.set_title('Потребление памяти', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Количество вершин', fontsize=12)
        ax1.set_ylabel('Память (KB)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(sizes_list, results_data['matrix_bfs_time'], 'r-',
                     label='Матрица', marker='o', linewidth=2)
        ax2.semilogy(sizes_list, results_data['list_bfs_time'], 'b-',
                     label='Список', marker='s', linewidth=2)
        ax2.set_title('Время выполнения BFS', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Количество вершин', fontsize=12)
        ax2.set_ylabel('Время (секунды, лог. шкала)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        ax3.semilogy(sizes_list, results_data['matrix_dfs_time'], 'r-',
                     label='Матрица', marker='o', linewidth=2)
        ax3.semilogy(sizes_list, results_data['list_dfs_time'], 'b-',
                     label='Список', marker='s', linewidth=2)
        ax3.set_title('Время выполнения DFS', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Количество вершин', fontsize=12)
        ax3.set_ylabel('Время (секунды, лог. шкала)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        ax4.plot(sizes_list, results_data['list_bfs_time'], 'g-',
                 label='BFS (список)', marker='o', linewidth=2)
        ax4.plot(sizes_list, results_data['list_dfs_time'], 'm-',
                 label='DFS (список)', marker='s', linewidth=2)
        ax4.set_title('Сравнение BFS и DFS (список смежности)',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Количество вершин', fontsize=12)
        ax4.set_ylabel('Время (секунды)', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def test_dijkstra_performance() -> None:
        """Тестирование производительности алгоритма Дейкстры."""
        dijkstra_sizes = [100, 200, 400, 600]
        dijkstra_times = []

        for current_size in dijkstra_sizes:
            _, weighted_list = GraphAnalyzer.generate_random_graph(
                current_size, 0.2, False, True
            )

            time_avg = GraphAnalyzer.benchmark_algorithm(
                GraphAlgorithms.dijkstra_adjacency_list, weighted_list, '0'
            )
            dijkstra_times.append(time_avg)

        plt.figure(figsize=(10, 6))
        plt.semilogy(dijkstra_sizes, dijkstra_times, 'purple', marker='o',
                     linewidth=3, markersize=8)
        plt.title('Производительность алгоритма Дейкстры',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Количество вершин', fontsize=14)
        plt.ylabel('Время (секунды, лог. шкала)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(dijkstra_sizes)
        plt.savefig('dijkstra_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print('\nАлгоритм Дейкстры:')
        print('Вершины | Время (сек)')
        print('-' * 25)
        for size_val, time_val in zip(dijkstra_sizes, dijkstra_times):
            print(f"{size_val:7} | {time_val:.6f}")

        print(f'\nСложность: O((V + E) log V)')

    @staticmethod
    def demonstrate_algorithms() -> None:
        """Демонстрация работы всех алгоритмов на практических задачах."""
        print('\n1. Определение связности сети')
        network = AdjacencyList()
        network.add_edge('A', 'B')
        network.add_edge('B', 'C')
        network.add_edge('D', 'E')
        network.add_edge('F', 'G')

        components = GraphTraversal.find_connected_components(network)
        print(f'Сеть содержит {len(components)} компонент(ы) связности:')
        for i, comp in enumerate(components, 1):
            print(f'  Компонента {i}: {comp}')

        print('\n2. Топологическая сортировка')
        curriculum = AdjacencyList(directed=True)
        curriculum.add_edge('A', 'B')
        curriculum.add_edge('A', 'C')
        curriculum.add_edge('B', 'D')
        curriculum.add_edge('C', 'D')
        curriculum.add_edge('C', 'E')
        curriculum.add_edge('D', 'F')

        try:
            order = GraphAlgorithms.topological_sort_adjacency_list(
                curriculum
            )
            print('Порядок выполнения задач:')
            for i, subject in enumerate(order, 1):
                print(f'  {i}. Задача {subject}')
        except ValueError as e:
            print(f'Циклическая зависимость: {e}')

        print('\n3. Кратчайший путь в сети')
        transport = AdjacencyList()
        transport.add_edge('A1', 'A2', 5)
        transport.add_edge('A2', 'A3', 10)
        transport.add_edge('A1', 'A4', 3)
        transport.add_edge('A4', 'A3', 15)
        transport.add_edge('A1', 'A3', 25)
        transport.add_edge('A4', 'A2', 2)

        distances, parents = GraphAlgorithms.dijkstra_adjacency_list(
            transport, 'A1'
        )
        path = GraphTraversal.get_shortest_path(parents, 'A3')

        print(f'Кратчайший путь от A1 до A3: {" -> ".join(path)}')
        print(f'Время в пути: {distances["A3"]}')

        print('\n4. Решение лабиринта')
        maze = [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]

        start_pos = (0, 0)
        end_pos = (4, 4)

        maze_path = GraphAlgorithms.solve_maze_shortest_path(
            maze, start_pos, end_pos
        )
        print(f'Лабиринт {len(maze)}x{len(maze[0])}')
        if maze_path:
            print(f'Путь от {start_pos} до {end_pos} найден')
            print(f'Длина пути: {len(maze_path)} шагов')
        else:
            print('Путь не найден!')


if __name__ == '__main__':
    GraphAnalyzer.system_info()

    analyzer = GraphAnalyzer()

    performance_data = analyzer.measure_operations_performance()
    graph_sizes_list = [100, 200, 400, 600, 800]

    analyzer.print_performance_table(performance_data, graph_sizes_list)

    analyzer.plot_performance_comparison(performance_data, graph_sizes_list)

    analyzer.test_dijkstra_performance()

    analyzer.demonstrate_algorithms()
