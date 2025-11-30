"""Реализация алгоритма Рабина-Карпа для поиска подстроки.

Временная сложность: O(n+m) в среднем случае, O(n*m) в худшем
Пространственная сложность: O(1)
"""

from typing import List


def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> List[int]:
    """Реализация алгоритма Рабина-Карпа для поиска подстроки.

    Args:
        text: Текст для поиска
        pattern: Искомая подстрока
        prime: Простое число для хеширования

    Returns:
        Список позиций начала вхождений подстроки
    """
    if not pattern:
        return []

    n = len(text)
    m = len(pattern)

    if m > n:
        return []

    occurrences = []

    pattern_hash = 0
    text_hash = 0
    h = 1

    d = 256

    for i in range(m - 1):
        h = (h * d) % prime

    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break

            if match:
                occurrences.append(i)

        if i < n - m:
            text_hash = (
                d * (text_hash - ord(text[i]) * h) + ord(text[i + m])
            ) % prime

            if text_hash < 0:
                text_hash += prime

    return occurrences


if __name__ == '__main__':
    demo_text = 'GEEKS FOR GEEKS'
    demo_pattern = 'GEEK'

    result = rabin_karp_search(demo_text, demo_pattern)
    print(f'Рабин-Карп поиск: "{demo_pattern}" в "{demo_text}"')
    print(f'Найденные позиции: {result}')
