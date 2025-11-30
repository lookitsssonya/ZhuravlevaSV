"""Модуль для реализации алгоритма Кнута-Морриса-Пратта (KMP) поиска подстроки.

Временная сложность: O(n + m)
Пространственная сложность: O(m)
"""

from typing import List

from prefix_function import compute_prefix_function


def kmp_search(text: str, pattern: str) -> List[int]:
    """Находит все вхождения подстроки в тексте с помощью алгоритма KMP.

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

    if m > n:
        return []

    pi = compute_prefix_function(pattern)
    occurrences = []

    j = 0

    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == m:
            occurrences.append(i - m + 1)
            j = pi[j - 1]

    return occurrences


def kmp_search_all_occurrences(text: str, pattern: str) -> List[int]:
    """Альтернативная реализация KMP для нахождения всех вхождений.

    Args:
        text: Текст для поиска
        pattern: Искомая подстрока

    Returns:
        Список позиций начала вхождений подстроки
    """
    return kmp_search(text, pattern)


if __name__ == '__main__':
    demo_text = 'ababcabcabababd'
    demo_pattern = 'ababd'

    result = kmp_search(demo_text, demo_pattern)
    print(f'KMP поиск: "{demo_pattern}" в "{demo_text}"')
    print(f'Найденные позиции: {result}')
