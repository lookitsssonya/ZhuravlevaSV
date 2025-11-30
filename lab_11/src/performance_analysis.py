"""Модуль для анализа производительности."""

import random
import string
import time
from typing import Callable, Dict, List

from boyer_moore import boyer_moore_search
from kmp_search import kmp_search
from rabin_karp import rabin_karp_search
from string_matching import naive_search
from z_function import z_search


def benchmark_search(algorithm: Callable, text: str, pattern: str,
                     repetitions: int = 10) -> float:
    """Замеряет время выполнения алгоритма поиска."""
    total_time = 0.0

    for _ in range(repetitions):
        start_time = time.perf_counter()
        algorithm(text, pattern)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    return total_time / repetitions


def measure_runtime_vs_text_length() -> Dict[str, List[float]]:
    """Замеряет время выполнения при различной длине текста."""
    algorithms = {
        'Наивный': naive_search,
        'KMP': kmp_search,
        'Z-поиск': z_search,
        'Бойер-Мур': boyer_moore_search,
        'Рабин-Карп': rabin_karp_search
    }

    text_lengths = [100, 500, 1000, 2000, 5000]
    pattern = 'test_pattern'

    results = {name: [] for name in algorithms.keys()}

    print('Замер времени выполнения при различной длине текста')

    for length in text_lengths:
        text = ''.join(random.choices(string.ascii_lowercase, k=length))
        print(f'\nДлина текста: {length}, паттерн: "{pattern}"')

        for name, algorithm in algorithms.items():
            time_taken = benchmark_search(algorithm, text, pattern, 5)
            results[name].append(time_taken)
            print(f'  {name}: {time_taken:.6f} сек')

    return results


def measure_runtime_vs_pattern_length() -> Dict[str, List[float]]:
    """Замеряет время выполнения при различной длине паттерна."""
    algorithms = {
        'Наивный': naive_search,
        'KMP': kmp_search,
        'Z-поиск': z_search,
        'Бойер-Мур': boyer_moore_search,
        'Рабин-Карп': rabin_karp_search
    }

    pattern_lengths = [1, 5, 10, 20, 50]
    text = ''.join(random.choices(string.ascii_lowercase, k=10000))

    results = {name: [] for name in algorithms.keys()}

    print('\nЗамер времени выполнения при различной длине паттерна')

    for p_length in pattern_lengths:
        pattern = ''.join(random.choices(string.ascii_lowercase, k=p_length))
        print(f'\nДлина текста: {len(text)}, паттерн: длина {p_length}')

        for name, algorithm in algorithms.items():
            time_taken = benchmark_search(algorithm, text, pattern, 5)
            results[name].append(time_taken)
            print(f'  {name}: {time_taken:.6f} сек')

    return results


def analyze_worst_best_cases() -> None:
    """Анализирует поведение алгоритмов в худшем и лучшем случаях."""
    algorithms = {
        'Наивный': naive_search,
        'KMP': kmp_search,
        'Z-поиск': z_search,
        'Бойер-Мур': boyer_moore_search,
        'Рабин-Карп': rabin_karp_search
    }

    print('\nАнализ худшего и лучшего случаев')

    worst_case_text = 'a' * 1000 + 'b'
    worst_case_pattern = 'a' * 500 + 'b'

    print(f'\nХудший случай (текст: "a"*1000 + "b", паттерн: "a"*500 + "b")')
    for name, algorithm in algorithms.items():
        time_taken = benchmark_search(
            algorithm, worst_case_text, worst_case_pattern, 3
        )
        print(f'  {name}: {time_taken:.6f} сек')

    best_case_text = 'a' * 1000
    best_case_pattern = 'b' * 10

    print(f'\nЛучший случай (текст: "a"*1000, паттерн: "b"*10)')
    for name, algorithm in algorithms.items():
        time_taken = benchmark_search(
            algorithm, best_case_text, best_case_pattern, 3
        )
        print(f'  {name}: {time_taken:.6f} сек')


if __name__ == '__main__':
    text_length_results = measure_runtime_vs_text_length()
    pattern_length_results = measure_runtime_vs_pattern_length()
    analyze_worst_best_cases()
