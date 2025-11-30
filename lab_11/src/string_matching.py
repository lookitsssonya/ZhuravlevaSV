"""Модуль, содержащий различные алгоритмы поиска подстроки.

Содержит наивный алгоритм и функции для сравнения производительности.
"""

import time
from typing import List


def naive_search(text: str, pattern: str) -> List[int]:
    """Наивный алгоритм поиска подстроки.

    Временная сложность: O(n*m) в худшем случае
    Пространственная сложность: O(1)

    Args:
        text: Текст для поиска
        pattern: Искомая подстрока

    Returns:
        Список позиций начала вхождений подстроки
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)

    occurrences = []

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break

        if match:
            occurrences.append(i)

    return occurrences


def benchmark_search(algorithm, text: str, pattern: str, repetitions: int = 100
                     ) -> float:
    """Замеряет время выполнения алгоритма поиска.

    Args:
        algorithm: Функция поиска
        text: Текст для поиска
        pattern: Искомая подстрока
        repetitions: Количество повторений для усреднения

    Returns:
        Среднее время выполнения в секундах
    """
    total_time = 0.0

    for _ in range(repetitions):
        start_time = time.perf_counter()
        algorithm(text, pattern)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    return total_time / repetitions


if __name__ == '__main__':
    test_text = 'a' * 1000 + 'b' + 'a' * 1000
    test_pattern = 'a' * 500 + 'b'

    print('Сравнение алгоритмов поиска:')

    naive_time = benchmark_search(naive_search, test_text, test_pattern, 10)
    print(f'Наивный поиск: {naive_time:.6f} сек')

    from kmp_search import kmp_search
    kmp_time = benchmark_search(kmp_search, test_text, test_pattern, 10)
    print(f'KMP поиск: {kmp_time:.6f} сек')

    from z_function import z_search
    z_time = benchmark_search(z_search, test_text, test_pattern, 10)
    print(f'Z-поиск: {z_time:.6f} сек')
