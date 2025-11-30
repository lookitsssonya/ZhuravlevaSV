"""Реализация алгоритма Бойера-Мура для поиска подстроки.

Временная сложность: O(n/m) в лучшем случае, O(n*m) в худшем
Пространственная сложность: O(m)
"""

from typing import List


def boyer_moore_search(text: str, pattern: str) -> List[int]:
    """Реализация алгоритма Бойера-Мура для поиска подстроки.

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

    bad_char = {}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - i - 1

    default_shift = m

    occurrences = []
    i = 0

    while i <= n - m:
        j = m - 1

        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            occurrences.append(i)
            i += 1
        else:
            shift = bad_char.get(text[i + j], default_shift)

            i += max(1, shift - (m - 1 - j))

    return occurrences


def boyer_moore_with_good_suffix(text: str, pattern: str) -> List[int]:
    """Алгоритм Бойера-Мура только с эвристикой плохого
    символа.

    Args:
        text: Текст для поиска
        pattern: Искомая подстрока

    Returns:
        Список позиций начала вхождений подстроки
    """
    return boyer_moore_search(text, pattern)


if __name__ == '__main__':
    demo_text = 'ABAAABCDABCABC'
    demo_pattern = 'ABC'

    result = boyer_moore_search(demo_text, demo_pattern)
    print(f'Бойер-Мур поиск: "{demo_pattern}" в "{demo_text}"')
    print(f'Найденные позиции: {result}')
