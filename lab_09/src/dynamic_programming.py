"""Реализация классических алгоритмов динамического программирования."""

from typing import Dict, List, Tuple


def fibonacci_naive(n: int) -> int:
    """
    Реализация чисел Фибоначчи (наивная рекурсия).

    Временная сложность: O(2^n)
    Пространственная сложность: O(n)
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memo(n: int, memo: Dict[int, int] = None) -> int:
    """
    Числа Фибоначчи с мемоизацией (нисходящий подход).

    Временная сложность: O(n)
    Пространственная сложность: O(n)
    """
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


def fibonacci_tabular(n: int) -> int:
    """
    Табличная реализация чисел Фибоначчи (восходящий подход).

    Временная сложность: O(n)
    Пространственная сложность: O(n)
    """
    if n <= 1:
        return n

    dp: List[int] = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def knapsack_01(
    weights: List[int], values: List[int], capacity: int
) -> Tuple[int, List[int]]:
    """
    Решение задачи о рюкзаке 0-1 с восстановлением выбранных предметов.

    Временная сложность: O(n * W)
    Пространственная сложность: O(n * W)
    """
    n: int = len(weights)
    dp: List[List[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i - 1][w],
                    values[i - 1] + dp[i - 1][w - weights[i - 1]]
                )
            else:
                dp[i][w] = dp[i - 1][w]

    selected_items: List[int] = []
    w: int = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()
    return dp[n][capacity], selected_items


def lcs(seq1: str, seq2: str) -> Tuple[int, str]:
    """
    Нахождение наибольшей общей подпоследовательности с восстановлением.

    Временная сложность: O(m * n)
    Пространственная сложность: O(m * n)
    """
    m: int = len(seq1)
    n: int = len(seq2)
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_sequence: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs_sequence.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_sequence.reverse()
    return dp[m][n], ''.join(lcs_sequence)


def levenshtein_distance(str1: str, str2: str) -> int:
    """
    Вычисление расстояния Левенштейна между двумя строками.

    Временная сложность: O(m * n)
    Пространственная сложность: O(m * n)
    """
    m: int = len(str1)
    n: int = len(str2)
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )

    return dp[m][n]


def coin_change(coins: List[int], amount: int) -> int:
    """
    Минимальное количество монет (задача размена).

    Временная сложность: O(n * amount)
    Пространственная сложность: O(amount)
    """
    dp: List[float] = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return int(dp[amount]) if dp[amount] != float('inf') else -1


def longest_increasing_subsequence(nums: List[int]) -> Tuple[int, List[int]]:
    """
    Нахождение наибольшей возрастающей подпоследовательности.

    Временная сложность: O(n^2)
    Пространственная сложность: O(n)
    """
    if not nums:
        return 0, []

    n: int = len(nums)
    dp: List[int] = [1] * n
    prev: List[int] = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j

    max_length: int = max(dp)
    max_index: int = dp.index(max_length)

    sequence: List[int] = []
    current: int = max_index
    while current != -1:
        sequence.append(nums[current])
        current = prev[current]

    sequence.reverse()
    return max_length, sequence


def knapsack_greedy_fractional(
    weights: List[int], values: List[int], capacity: int
) -> float:
    """
    Жадный алгоритм для непрерывного рюкзака.

    Временная сложность: O(n log n)
    Пространственная сложность: O(n)
    """
    n: int = len(weights)
    items: List[Tuple[float, int, int]] = []

    for i in range(n):
        ratio: float = values[i] / weights[i]
        items.append((ratio, values[i], weights[i]))

    items.sort(reverse=True)

    total_value: float = 0.0
    remaining_capacity: int = capacity

    for ratio, value, weight in items:
        if remaining_capacity >= weight:
            total_value += value
            remaining_capacity -= weight
        else:
            total_value += value * (remaining_capacity / weight)
            break

    return total_value
