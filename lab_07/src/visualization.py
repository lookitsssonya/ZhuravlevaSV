"""Визуализация кучи в виде дерева."""

from collections import deque
from typing import Deque, List, Tuple
from heap import Heap


def print_tree(heap: Heap) -> None:
    """
    Визуализация кучи.

    Args:
        heap: Куча для визуализации
    """
    array = heap.get_heap_array()
    if not array:
        print('(empty)')
        return

    height = 0
    n = len(array)
    while (1 << height) - 1 < n:
        height += 1

    levels: List[List[str]] = []
    queue: Deque[Tuple[int, int]] = deque()
    queue.append((0, 0))
    max_level = 0

    while queue:
        idx, level = queue.popleft()

        if level >= len(levels):
            levels.append([])
            max_level = level

        if idx >= len(array):
            levels[level].append('  ')
            if level < max_level:
                queue.append((2 * idx + 1, level + 1))
                queue.append((2 * idx + 2, level + 1))
            continue
        else:
            levels[level].append(f'{array[idx]:2d}')

        queue.append((2 * idx + 1, level + 1))
        queue.append((2 * idx + 2, level + 1))

    while levels and all(cell == '  ' for cell in levels[-1]):
        levels.pop()

    for i, level in enumerate(levels):
        spaces_before = 2 ** (len(levels) - i - 1) - 1
        spaces_between = 2 ** (len(levels) - i) - 1

        level_line = ' ' * spaces_before
        for j, cell in enumerate(level):
            level_line += cell
            if j < len(level) - 1:
                level_line += ' ' * spaces_between

        print(level_line)

        if i < len(levels) - 1:
            connection_line = ' ' * (spaces_before - 1)
            next_level = levels[i + 1] if i + 1 < len(levels) else []

            for j in range(len(level)):
                left_index = j * 2
                right_index = j * 2 + 1

                has_left = (
                    left_index < len(next_level) and
                    next_level[left_index] != '  '
                )
                has_right = (
                    right_index < len(next_level) and
                    next_level[right_index] != '  '
                )

                if has_left and has_right:
                    connection_line += '╻━┻━╻'
                elif has_left:
                    connection_line += '╻━╹  '
                elif has_right:
                    connection_line += '  ╹━╻'
                else:
                    connection_line += '     '

                if j < len(level) - 1:
                    connection_line += ' ' * (spaces_between - 3)

            print(connection_line)


def print_heap(heap: Heap) -> None:
    """Печать кучи в виде дерева."""
    print(f'Куча ({"min" if heap.is_min else "max"}):')
    print_tree(heap)


def print_heap_simple(heap: Heap) -> None:
    """Простая печать кучи в виде массива."""
    print(f'Куча ({"min" if heap.is_min else "max"}): {heap.get_heap_array()}')
