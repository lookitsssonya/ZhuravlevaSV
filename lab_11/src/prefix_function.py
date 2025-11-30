"""Модуль для вычисления префикс-функции строки.

Временная сложность: O(n)
Пространственная сложность: O(n)
"""

from typing import List


def compute_prefix_function(text: str) -> List[int]:
    """Вычисляет префикс-функцию для строки.

    Args:
        text: Входная строка

    Returns:
        Список значений префикс-функции
    """
    n = len(text)
    pi = [0] * n

    for i in range(1, n):
        j = pi[i - 1]

        while j > 0 and text[i] != text[j]:
            j = pi[j - 1]

        if text[i] == text[j]:
            j += 1

        pi[i] = j

    return pi


def find_period_using_prefix(text: str) -> int:
    """Находит период строки с использованием префикс-функции.

    Args:
        text: Входная строка

    Returns:
        Длина периода строки (0, если строка непериодическая)
    """
    n = len(text)
    if n == 0:
        return 0

    pi = compute_prefix_function(text)
    period_length = n - pi[n - 1]

    if n % period_length == 0:
        return period_length
    else:
        return 0


if __name__ == '__main__':
    test_string = 'abcabcd'
    prefix_result = compute_prefix_function(test_string)
    print(f'Префикс-функция для "{test_string}": {prefix_result}')

    found_period = find_period_using_prefix('abcabcabc')
    print(f'Период строки "abcabcabc": {found_period}')
