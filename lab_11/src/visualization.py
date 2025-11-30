"""Модуль для визуализации работы алгоритмов."""

import random
import string
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from boyer_moore import boyer_moore_search
from kmp_search import kmp_search
from rabin_karp import rabin_karp_search
from string_matching import naive_search
from z_function import z_search


def system_info() -> None:
    """Вывод информации о системе."""
    print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i5-13420H (2.10 GHz)
- Оперативная память: 16 GB DDR5
- ОС: Windows 11
- Python: 3.11
''')


def benchmark_search(algorithm, text: str, pattern: str, repetitions: int = 5
                     ) -> float:
    """Замер времени выполнения."""
    times = []

    for _ in range(repetitions):
        start_time = time.perf_counter()
        algorithm(text, pattern)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return sorted(times)[len(times) // 2]


def generate_performance_data() -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]]
]:
    """Генерирует данные производительности для графиков."""
    algorithms = {
        'Наивный': naive_search,
        'KMP': kmp_search,
        'Z-поиск': z_search,
        'Бойер-Мур': boyer_moore_search,
        'Рабин-Карп': rabin_karp_search
    }

    text_lengths = [1000, 5000, 10000, 20000, 50000]
    fixed_pattern = 'test_pattern_123'

    text_length_results = {name: [] for name in algorithms.keys()}

    print('\nТестирование на разных длинах текста')
    for length in text_lengths:
        text = ''.join(random.choices(string.ascii_lowercase, k=length))
        print(f'\nДлина текста: {length}')

        for name, algorithm in algorithms.items():
            time_taken = benchmark_search(algorithm, text, fixed_pattern, 3)
            text_length_results[name].append(time_taken)
            print(f'    {name}: {time_taken:.6f} сек')

    pattern_lengths = [10, 50, 100, 200, 500]
    fixed_text = ''.join(random.choices(string.ascii_lowercase, k=50000))

    pattern_length_results = {name: [] for name in algorithms.keys()}

    print('\nТестирование на разных длинах паттерна')
    for p_len in pattern_lengths:
        pattern = ''.join(random.choices(string.ascii_lowercase, k=p_len))

        print(f'\nДлина паттерна: {p_len}')

        for name, algorithm in algorithms.items():
            time_taken = benchmark_search(algorithm, fixed_text, pattern, 3)
            pattern_length_results[name].append(time_taken)
            print(f'    {name}: {time_taken:.6f} сек')

    return text_length_results, pattern_length_results


def plot_performance_summary(text_results: Dict[str, List[float]],
                             pattern_results: Dict[str, List[float]]) -> None:
    """Сводный график производительности."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    text_lengths = [1000, 5000, 10000, 20000, 50000]
    pattern_lengths = [10, 50, 100, 200, 500]

    colors = {
        'Наивный': 'red', 'KMP': 'green', 'Z-поиск': 'blue',
        'Бойер-Мур': 'purple', 'Рабин-Карп': 'orange'
    }

    for algorithm, times in text_results.items():
        ax1.plot(
            text_lengths, times,
            color=colors[algorithm], marker='o',
            label=algorithm, linewidth=2, markersize=6
        )

    ax1.set_title(
        'Зависимость от длины текста', fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel('Длина текста')
    ax1.set_ylabel('Время (сек)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for algorithm, times in pattern_results.items():
        ax2.plot(
            pattern_lengths, times,
            color=colors[algorithm], marker='s',
            label=algorithm, linewidth=2, markersize=6
        )

    ax2.set_title(
        'Зависимость от длины паттерна', fontsize=12, fontweight='bold'
    )
    ax2.set_xlabel('Длина паттерна')
    ax2.set_ylabel('Время (сек)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_prefix_and_z_functions() -> None:
    """Визуализация префикс-функции и Z-функции."""
    from prefix_function import compute_prefix_function
    from z_function import compute_z_function

    examples = [
        ('abcabcabc', 'Периодическая строка'),
        ('aaaabaa', 'Строка с повторениями'),
        ('abababab', 'Чередующаяся строка')
    ]

    for text, description in examples:
        pi = compute_prefix_function(text)
        z = compute_z_function(text)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x_pos = list(range(len(text)))
        bars1 = ax1.bar(
            x_pos, pi, color='lightblue', alpha=0.7, edgecolor='navy'
        )
        ax1.set_title(f'Префикс-функция: "{text}"\n{description}', fontsize=12)
        ax1.set_xlabel('Позиция i')
        ax1.set_ylabel('π[i]')
        ax1.grid(True, alpha=0.3)

        for bar, value in zip(bars1, pi):
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, height + 0.05,
                    f'{value}', ha='center', va='bottom', fontsize=8
                )

        bars2 = ax2.bar(
            x_pos, z, color='lightcoral', alpha=0.7, edgecolor='darkred'
        )
        ax2.set_title(f'Z-функция: "{text}"\n{description}', fontsize=12)
        ax2.set_xlabel('Позиция i')
        ax2.set_ylabel('z[i]')
        ax2.grid(True, alpha=0.3)

        for bar, value in zip(bars2, z):
            height = bar.get_height()
            if height > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2, height + 0.05,
                    f'{value}', ha='center', va='bottom', fontsize=8
                )

        plt.tight_layout()
        plt.savefig(f'functions_{text[:5]}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n '{text}':")
        print(f'Префикс-функция: {pi}')
        print(f'Z-функция:       {z}')

        n = len(text)
        if n > 0:
            period_pi = n - pi[n - 1]
            if n % period_pi == 0 and period_pi < n:
                print(f'Период (по префикс-функции): {period_pi}')

            for i in range(1, n):
                if i + z[i] == n and n % i == 0:
                    print(f'Период (по Z-функции): {i}')
                    break


def analyze_and_print_results(text_results: Dict[str, List[float]],
                              pattern_results: Dict[str, List[float]]) -> None:
    """Анализирует и выводит результаты тестирования."""
    print('\nЗависимость от длины текста:')

    avg_times = {}
    for algo, times in text_results.items():
        avg_times[algo] = sum(times) / len(times)

    for algo, avg_time in sorted(avg_times.items(), key=lambda x: x[1]):
        print(f'  {algo}: {avg_time:.6f} сек')

    print('\nЗависимость от длины паттерна:')

    avg_times_pattern = {}
    for algo, times in pattern_results.items():
        avg_times_pattern[algo] = sum(times) / len(times)

    for algo, avg_time in sorted(
        avg_times_pattern.items(), key=lambda x: x[1]
    ):
        print(f'  {algo}: {avg_time:.6f} сек')

    print('\nАнализ результатов:')
    fastest_text = min(avg_times, key=avg_times.get)
    fastest_pattern = min(avg_times_pattern, key=avg_times_pattern.get)

    print(f'- Самый быстрый алгоритм для поиска в тексте: {fastest_text}')
    print(f'- Самый быстрый алгоритм для разных паттернов: {fastest_pattern}')

    if fastest_text in avg_times and 'Наивный' in avg_times:
        naive_ratio = avg_times['Наивный'] / avg_times[fastest_text]
        text = f'- {fastest_text} быстрее в {naive_ratio:.1f} раз'
        print(text)


def main() -> None:
    """Основная функция для построения графиков."""
    system_info()

    print('Префикс-функции и Z-функции')
    visualize_prefix_and_z_functions()

    text_results, pattern_results = generate_performance_data()

    plot_performance_summary(text_results, pattern_results)

    analyze_and_print_results(text_results, pattern_results)


if __name__ == '__main__':
    main()
