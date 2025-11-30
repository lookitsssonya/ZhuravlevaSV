"""Модуль для вычисления Z-функции строки.

Временная сложность: O(n)
Пространственная сложность: O(n)
"""

from typing import List


def compute_z_function(text: str) -> List[int]:
    """Вычисляет Z-функцию для строки.

    Args:
        text: Входная строка

    Returns:
        Список значений Z-функции
    """
    n = len(text)
    if n == 0:
        return []

    z = [0] * n
    left = 0
    right = 0

    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])

        while i + z[i] < n and text[z[i]] == text[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1

    return z


def z_search(text: str, pattern: str) -> List[int]:
    """Находит все вхождения подстроки в тексте с помощью Z-функции.

    Args:
        text: Текст для поиска
        pattern: Искомая подстрока

    Returns:
        Список позиций начала вхождений подстроки
    """
    if not pattern:
        return []

    m = len(pattern)
    n = len(text)

    if m > n:
        return []

    combined = pattern + '$' + text
    z = compute_z_function(combined)

    occurrences = []

    for i in range(m + 1, len(combined)):
        if z[i] == m:
            occurrences.append(i - m - 1)

    return occurrences


def find_period_using_z(text: str) -> int:
    """Находит период строки с использованием Z-функции.

    Args:
        text: Входная строка

    Returns:
        Длина периода строки
    """
    n = len(text)
    if n == 0:
        return 0

    z = compute_z_function(text)

    for i in range(1, n):
        if i + z[i] == n and n % i == 0:
            return i

    return n


if __name__ == '__main__':
    demo_string = 'aaaabaa'
    z_result = compute_z_function(demo_string)
    print(f'Z-функция для "{demo_string}": {z_result}')

    demo_text = 'ababcabcabababd'
    demo_pattern = 'abc'
    search_result = z_search(demo_text, demo_pattern)
    print(f'Z-поиск: "{demo_pattern}" в "{demo_text}": {search_result}')
