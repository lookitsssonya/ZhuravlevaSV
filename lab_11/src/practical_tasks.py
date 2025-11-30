"""Модуль для решения практических задач
с использованием алгоритмов на строках."""

from typing import List

from kmp_search import kmp_search
from prefix_function import compute_prefix_function
from z_function import compute_z_function


def find_all_occurrences(text: str, pattern: str) -> List[int]:
    """Задача 1: Найти все вхождения паттерна в тексте.

    Использует алгоритм KMP как наиболее надежный.

    Args:
        text: Текст для поиска
        pattern: Искомый паттерн

    Returns:
        Список позиций вхождений
    """
    return kmp_search(text, pattern)


def find_string_period(text: str) -> int:
    """Задача 2: Найти период строки.

    Использует префикс-функцию для эффективного нахождения периода.

    Args:
        text: Входная строка

    Returns:
        Длина периода (0, если строка непериодическая)
    """
    n = len(text)
    if n == 0:
        return 0

    pi = compute_prefix_function(text)
    period = n - pi[n - 1]

    if n % period == 0:
        if text[:period] * (n // period) == text:
            return period

    return 0


def check_cyclic_shift(s1: str, s2: str) -> bool:
    """Задача 3: Проверить, является ли одна строка циклическим сдвигом другой.

    Args:
        s1: Первая строка
        s2: Вторая строка

    Returns:
        True, если s2 является циклическим сдвигом s1
    """
    if len(s1) != len(s2):
        return False

    doubled = s1 + s1
    return len(kmp_search(doubled, s2)) > 0


def find_longest_palindrome(text: str) -> str:
    """Задача 4: Найти самый длинный палиндром в строке.

    Использует Z-функцию для эффективного поиска.

    Args:
        text: Входная строка

    Returns:
        Самый длинный палиндром
    """
    if not text:
        return ''

    reversed_text = text[::-1]
    combined = text + '#' + reversed_text
    z = compute_z_function(combined)

    n = len(text)
    max_length = 1
    start_index = 0

    for i in range(n + 1, len(combined)):
        if z[i] > max_length:
            max_length = z[i]
            start_index = n - (i - n - 1 + max_length - 1) - 1

    return text[start_index:start_index + max_length]


def find_common_substring(strings: List[str]) -> str:
    """Задача 5: Найти самую длинную общую подстроку среди набора строк.

    Args:
        strings: Список строк

    Returns:
        Самая длинная общая подстрока
    """
    if not strings:
        return ''

    base_string = strings[0]
    n = len(base_string)
    longest = ''

    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = base_string[i:j]

            in_all = True
            for s in strings[1:]:
                if substring not in s:
                    in_all = False
                    break

            if in_all and len(substring) > len(longest):
                longest = substring

    return longest


def count_distinct_substrings(text: str) -> int:
    """Задача 6: Подсчитать количество различных подстрок в строке.

    Args:
        text: Входная строка

    Returns:
        Количество различных подстрок
    """
    n = len(text)
    distinct = set()

    for i in range(n):
        for j in range(i + 1, n + 1):
            distinct.add(text[i:j])

    return len(distinct)


def solve_practical_tasks() -> None:
    """Решает все практические задачи и выводит результаты."""
    print('Решение практических задач')

    print('\n1. Поиск всех вхождений паттерна в тексте')
    text1 = 'ababcabcabababd'
    pattern1 = 'abc'
    print(f'Найти все вхождения подстроки "{pattern1}" в строке "{text1}"')
    occurrences = find_all_occurrences(text1, pattern1)
    result_text = f'Результат: Найдено {len(occurrences)} вхождений'
    print(f'{result_text} на позициях: {occurrences}')

    print('\n2. Найти период строки')
    periodic_text = 'abcabcabc'
    print(f'Найти период строки "{periodic_text}"')
    period = find_string_period(periodic_text)
    if period > 0:
        repeats = len(periodic_text) // period
        period_info = f'Результат: Период строки {period}.'
        repetition_info = f'Строка состоит из {repeats} повторений'
        print(period_info)
        print(f'{repetition_info} "{periodic_text[:period]}"')
    else:
        print('Результат: Строка не является периодической')

    print('\n3. Проверка циклического сдвига строк')
    s1 = 'abcde'
    s2 = 'cdeab'
    print(f'Проверить, является ли "{s2}" циклическим сдвигом "{s1}"')
    is_cyclic = check_cyclic_shift(s1, s2)
    if is_cyclic:
        result = f'Результат: "{s2}" является циклическим сдвигом "{s1}"'
        print(result)
    else:
        result = f'Результат: "{s2}" не является циклическим сдвигом "{s1}"'
        print(result)

    print('\n4. Поиск самого длинного палиндрома в строке')
    palindrome_text = 'babad'
    print(f'Найти самый длинный палиндром в строке "{palindrome_text}"')
    longest_pal = find_longest_palindrome(palindrome_text)
    pal_info = f'Результат: Самый длинный палиндром: "{longest_pal}"'
    print(f'{pal_info} (длина: {len(longest_pal)})')

    print('\n5. Поиск самой длинной общей подстроки')
    strings_list = ['flower', 'flow', 'flight']
    print(f'Найти самую длинную общую подстроку в строках: {strings_list}')
    common_sub = find_common_substring(strings_list)
    if common_sub:
        common_info = 'Результат: Самая длинная общая подстрока:'
        print(f'{common_info} "{common_sub}" (длина: {len(common_sub)})')
    else:
        print('Результат: Общая подстрока не найдена')

    print('\n6. Подсчет количества различных подстрок')
    count_text = 'aba'
    print(f'Подсчитать количество различных подстрок в строке "{count_text}"')
    distinct_count = count_distinct_substrings(count_text)
    result = f'Результат: В строке "{count_text}" содержится'
    print(f'{result} {distinct_count} различных подстрок')


if __name__ == '__main__':
    solve_practical_tasks()
