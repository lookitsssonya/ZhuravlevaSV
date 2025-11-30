"""Сравнительный анализ различных подходов ДП и алгоритмов."""

import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from dynamic_programming import (
    coin_change,
    fibonacci_memo,
    fibonacci_naive,
    fibonacci_tabular,
    knapsack_01,
    knapsack_greedy_fractional,
    longest_increasing_subsequence,
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


def measure_time(func, *args, iterations: int = 1) -> float:
    """Точное измерение времени выполнения."""
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)  # Результат не используется
    end = time.perf_counter()
    return (end - start) / iterations


def compare_fibonacci_approaches() -> None:
    """Сравнение времени работы разных подходов для чисел Фибоначчи."""
    n_values: List[int] = [10, 15, 20, 25, 30, 35]
    naive_times: List[float] = []
    memo_times: List[float] = []
    tabular_times: List[float] = []

    print('Сравнение подходов для чисел Фибоначчи:')
    print('n\tНаивная\t\tМемоизация\tТабличный')
    print('-' * 55)

    for n in n_values:
        naive_time = 0.0
        if n <= 30:
            naive_time = measure_time(fibonacci_naive, n)
            naive_times.append(naive_time)
        else:
            naive_times.append(float('nan'))

        memo_time = measure_time(fibonacci_memo, n, iterations=1000)
        memo_times.append(memo_time)

        tabular_time = measure_time(fibonacci_tabular, n, iterations=1000)
        tabular_times.append(tabular_time)

        if n <= 30:
            print(
                f'{n}\t{naive_time:.6f}\t{memo_time:.6f}\t{tabular_time:.6f}'
            )
        else:
            print(f'{n}\t--\t\t{memo_time:.6f}\t{tabular_time:.6f}')

    plt.figure(figsize=(12, 8))

    valid_naive_indices = [
        i for i, t in enumerate(naive_times) if not np.isnan(t)
    ]
    if valid_naive_indices:
        plt.plot(
            [n_values[i] for i in valid_naive_indices],
            [naive_times[i] for i in valid_naive_indices],
            'ro-', label='Наивная рекурсия', linewidth=2, markersize=6
        )

    plt.plot(
        n_values, memo_times, 'go-', label='С мемоизацией',
        linewidth=2, markersize=6
    )
    plt.plot(
        n_values, tabular_times, 'bo-', label='Табличный',
        linewidth=2, markersize=6
    )

    plt.xlabel('n', fontsize=12)
    plt.ylabel('Время выполнения (секунды)', fontsize=12)
    plt.title('Сравнение времени выполнения для чисел Фибоначчи', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.savefig('fibonacci_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_knapsack_approaches() -> None:
    """Сравнение ДП и жадного алгоритма для задачи о рюкзаке."""
    weights_01: List[int] = [2, 3, 4, 5]
    values_01: List[int] = [3, 4, 5, 6]
    capacity_01: int = 5

    weights_fractional: List[int] = [2, 3, 4, 5]
    values_fractional: List[int] = [3, 4, 5, 6]
    capacity_fractional: int = 5

    print('\nСравнение алгоритмов для задачи о рюкзаке:')

    print(f'Предметы (вес, стоимость): {list(zip(weights_01, values_01))}')
    print(f'Вместимость рюкзака: {capacity_01}')

    dp_time = measure_time(
        knapsack_01, weights_01, values_01, capacity_01, iterations=1000
    )
    dp_value, dp_items = knapsack_01(weights_01, values_01, capacity_01)

    print(f'\nРюкзак 0-1 (ДП):')
    print(f'  Максимальная стоимость: {dp_value}')
    print(f'  Выбранные предметы: {dp_items}')
    items_description = [
        (weights_01[i], values_01[i]) for i in dp_items
    ]
    print(f'  Предметы (вес, стоимость): {items_description}')
    total_weight = sum(weights_01[i] for i in dp_items)
    print(f'  Суммарный вес: {total_weight}')
    print(f'  Время выполнения: {dp_time:.6f} сек')

    greedy_time = measure_time(
        knapsack_greedy_fractional, weights_fractional,
        values_fractional, capacity_fractional, iterations=1000
    )
    greedy_value = knapsack_greedy_fractional(
        weights_fractional, values_fractional, capacity_fractional
    )

    print(f'\nНепрерывный рюкзак (жадный):')
    print(f'  Максимальная стоимость: {greedy_value:.2f}')
    print(f'  Время выполнения: {greedy_time:.6f} сек')

    print(f'\nРазница в стоимости: {abs(dp_value - greedy_value):.2f}')


def analyze_dp_scalability() -> None:
    """Анализ масштабируемости алгоритмов ДП."""
    print('\nАнализ масштабируемости алгоритмов ДП:')

    sizes: List[int] = [10, 20, 30, 40, 50]
    knapsack_times: List[float] = []

    print('Рюкзак 0-1:')
    for size in sizes:
        weights: List[int] = [i % 10 + 1 for i in range(size)]
        values: List[int] = [(i * 7) % 20 + 1 for i in range(size)]
        capacity: int = sum(weights) // 2

        execution_time = measure_time(
            knapsack_01, weights, values, capacity, iterations=10
        )
        knapsack_times.append(execution_time)

        print(f'  n={size}: {execution_time:.4f} сек')


def test_practical_problems() -> None:
    """Тестирование решения практических задач."""
    print('\nРешение практических задач:')

    print('\n1. Задача размена монет')
    coins: List[int] = [1, 2, 5]
    amount: int = 11
    min_coins = coin_change(coins, amount)
    print(f'   Монеты: {coins}')
    print(f'   Сумма: {amount}')
    print(f'   Минимальное количество монет: {min_coins}')

    print('\n2. Наибольшая возрастающая подпоследовательность')
    sequence: List[int] = [10, 9, 2, 5, 3, 7, 101, 18]
    lis_length, lis_seq = longest_increasing_subsequence(sequence)
    print(f'   Последовательность: {sequence}')
    print(f'   Длина LIS: {lis_length}')
    print(f'   LIS: {lis_seq}')


if __name__ == '__main__':
    system_info()
    compare_fibonacci_approaches()
    compare_knapsack_approaches()
    analyze_dp_scalability()
    test_practical_problems()
