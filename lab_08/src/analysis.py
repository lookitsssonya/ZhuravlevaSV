"""
Анализ эффективности жадных алгоритмов и сравнение с другими методами.
"""

import time
import random
from typing import List, Tuple

import matplotlib.pyplot as plt

from greedy_algorithms import (
    CoinChange,
    FractionalKnapsack,
    HuffmanCoding,
    IntervalScheduling,
    PrimMST
)


def system_info() -> None:
    """Вывод информации о системе."""
    print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i5-13420H (2.10 GHz)
- Оперативная память: 16 GB DDR5
- ОС: Windows 11
- Python: 3.11
''')


class KnapsackAnalyzer:
    """Анализатор для задач о рюкзаке."""

    @staticmethod
    def brute_force_01_knapsack(
        items: List[Tuple[float, float]],
        capacity: float
    ) -> Tuple[float, List[int]]:
        """
        Решает дискретную задачу о рюкзаке методом полного перебора.

        Args:
            items: Список предметов (weight, value)
            capacity: Вместимость рюкзака

        Returns:
            Tuple[float, List[int]]:
            (максимальная стоимость, выбранные предметы)

        Сложность: O(2^n)
        """
        n = len(items)
        max_value = 0.0
        best_selection: List[int] = []

        for i in range(1 << n):
            total_weight = 0.0
            total_value = 0.0
            selection: List[int] = []

            for j in range(n):
                if i & (1 << j):
                    weight, value = items[j]
                    total_weight += weight
                    total_value += value
                    selection.append(j)

            if total_weight <= capacity and total_value > max_value:
                max_value = total_value
                best_selection = selection

        return max_value, best_selection

    @staticmethod
    def compare_knapsack_algorithms() -> None:
        """Сравнивает жадный и точный алгоритмы для задачи о рюкзаке."""
        items = [(10, 60), (20, 100), (30, 120)]
        capacity = 50

        print('Сравнение алгоритмов для задачи о рюкзаке')
        print(f'Предметы: {items}')
        print(f'Вместимость рюкзака: {capacity}')
        print()

        fractional_value, fractional_items = (
            FractionalKnapsack.solve_fractional_knapsack(items, capacity)
        )
        print('Непрерывный рюкзак (жадный алгоритм):')
        print(f'Максимальная стоимость: {fractional_value:.2f}')
        print(f'Взятые предметы: {fractional_items}')
        print()

        discrete_value, discrete_selection = (
            KnapsackAnalyzer.brute_force_01_knapsack(items, capacity)
        )
        discrete_items = [items[i] for i in discrete_selection]
        print('Дискретный рюкзак (полный перебор):')
        print(f'Максимальная стоимость: {discrete_value:.2f}')
        print(f'Взятые предметы: {discrete_items}')
        print()

        print('Сравнение результатов:')
        diff = fractional_value - discrete_value
        print(f'Разница в стоимости: {diff:.2f}')
        if fractional_value > discrete_value:
            print('Жадный подход дал неоптимальный результат')
        else:
            print('Оба алгоритма дали одинаковый результат')


class PerformanceAnalyzer:
    """Анализатор производительности алгоритмов."""

    @staticmethod
    def measure_huffman_performance() -> None:
        """Измеряет производительность алгоритма Хаффмана."""
        print('\nАнализ производительности алгоритма Хаффмана')

        sizes = [100, 500, 1000, 5000, 10000, 50000]
        times: List[float] = []

        for size in sizes:
            text = ''.join(
                random.choices('abcdefghijklmnopqrstuvwxyz ', k=size)
            )

            start_time = time.perf_counter()
            HuffmanCoding.huffman_encode(text)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            times.append(execution_time)

            print(f'Размер текста: {size} символов')
            print(f'Время выполнения: {execution_time:.6f} секунд')

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Размер входных данных (символов)')
        plt.ylabel('Время выполнения (секунды)')
        plt.title('Зависимость времени работы алгоритма Хаффмана')
        plt.grid(True, alpha=0.3)
        plt.savefig('performance_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

        print('График сохранен в файл "performance_plot.png"')


class AlgorithmCorrectness:
    """Анализ корректности жадных алгоритмов."""

    @staticmethod
    def test_interval_scheduling() -> None:
        """Тестирует корректность алгоритма выбора заявок."""
        print('\nТестирование алгоритма выбора заявок')

        intervals = [(1, 3), (2, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)]
        selected = IntervalScheduling.select_intervals(intervals)

        print(f'Все интервалы: {intervals}')
        print(f'Выбранные непересекающиеся интервалы: {selected}')
        print(f'Количество выбранных интервалов: {len(selected)}')
        print()

    @staticmethod
    def test_coin_change() -> None:
        """Тестирует алгоритм размена монет."""
        print('Тестирование алгоритма размена монет')

        standard_coins = [10000, 5000, 2000, 1000, 500, 100, 50, 10, 5, 2, 1]
        amount = 36780

        try:
            result = CoinChange.min_coins_greedy(amount, standard_coins)
            print(f'Сумма для размена: {amount/100:.2f} руб')
            print(f'Результат размена:')
            for coin in sorted(result.keys(), reverse=True):
                if result[coin] > 0:
                    print(f'  Монета {coin/100:.2f} руб: {result[coin]} шт')

            total = sum(coin * count for coin, count in result.items())
            check_result = abs(total - amount) < 1
            print(
                f'Проверка: {total/100:.2f} руб == {amount/100:.2f} руб '
                f'-> {check_result}'
            )
        except ValueError as e:
            print(f'Ошибка: {e}')
        print()

    @staticmethod
    def test_prim_algorithm() -> None:
        """Тестирует алгоритм Прима."""
        print('Тестирование алгоритма Прима')

        graph = {
            'A': [('B', 1), ('C', 3), ('D', 4)],
            'B': [('A', 1), ('C', 2), ('D', 5)],
            'C': [('A', 3), ('B', 2), ('D', 6)],
            'D': [('A', 4), ('B', 5), ('C', 6)]
        }

        mst_edges = PrimMST.prim_algorithm(graph)
        total_weight = sum(weight for _, _, weight in mst_edges)

        print('Граф:')
        for vertex, edges in graph.items():
            print(f"  {vertex}: {edges}")
        print(f'Минимальное остовное дерево: {mst_edges}')
        print(f'Общий вес MST: {total_weight}')
        print()


def main() -> None:
    """Основная функция для запуска анализа."""
    system_info()

    AlgorithmCorrectness.test_interval_scheduling()
    AlgorithmCorrectness.test_coin_change()
    AlgorithmCorrectness.test_prim_algorithm()

    KnapsackAnalyzer.compare_knapsack_algorithms()

    PerformanceAnalyzer.measure_huffman_performance()

    print('\nДемонстрация алгоритма Хаффмана')
    text = 'abcdefghijklmnop'
    codes, encoded = HuffmanCoding.huffman_encode(text)

    print(f'Исходный текст: "{text}"')
    print('Коды Хаффмана:')
    for char, code in sorted(codes.items()):
        print(f'  "{char}": {code}')
    print(f'Закодированный текст: {encoded}')

    frequencies = {char: text.count(char) for char in set(text)}
    root = HuffmanCoding.build_huffman_tree(frequencies)
    HuffmanCoding.visualize_tree(root, 'huffman_tree.png')
    print('Дерево Хаффмана сохранено в файл "huffman_tree.png"')


if __name__ == "__main__":
    main()
