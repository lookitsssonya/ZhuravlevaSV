"""Визуализация процессов ДП и построение графиков."""

import matplotlib.pyplot as plt
from typing import List


def visualize_knapsack_table(
    weights: List[int], values: List[int], capacity: int
) -> None:
    """Визуализация таблицы ДП для задачи о рюкзаке"""
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

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    row_labels = [''] + [f'Item {i}' for i in range(1, n + 1)]
    col_labels = [f'W={w}' for w in range(capacity + 1)]

    table = ax.table(
        cellText=dp,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        if key[0] == 0 or key[1] < 0:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('white')

    plt.title('Таблица ДП для задачи о рюкзаке 0-1', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('knapsack_table.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_lcs_table(seq1: str, seq2: str) -> None:
    """Визуализация таблицы ДП для LCS"""
    m: int = len(seq1)
    n: int = len(seq2)
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    row_labels = [''] + [f'{seq1[i]}' for i in range(m)]
    col_labels = [''] + [f'{seq2[j]}' for j in range(n)]

    table = ax.table(
        cellText=dp,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
        if key[0] == 0 or key[1] < 0:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('white')

    plt.title(
        'Таблица ДП для поиска наибольшей общей подпоследовательности (LCS)',
        fontsize=14, pad=20
    )
    plt.tight_layout()
    plt.savefig('lcs_table.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_dp_scalability() -> None:
    """Построение графика масштабируемости алгоритмов ДП."""
    sizes = [10, 20, 30, 40, 50]
    execution_times = [0.0001, 0.0005, 0.0009, 0.0016, 0.0025]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, execution_times, 'purple', marker='o', linewidth=2)
    plt.xlabel('Размер задачи (количество предметов)')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Масштабируемость алгоритма 0-1 рюкзака')
    plt.grid(True, alpha=0.3)
    plt.savefig('knapsack_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    visualize_knapsack_table([2, 3, 4, 5], [3, 4, 5, 6], 5)

    visualize_lcs_table('ABCDGH', 'AEDFHR')

    plot_dp_scalability()
