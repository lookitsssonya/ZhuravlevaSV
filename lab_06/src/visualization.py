"""Модуль для визуализации структуры дерева."""

from collections import deque
from typing import Optional

from binary_search_tree import TreeNode


def tree_to_bracket(root: Optional[TreeNode]) -> str:
    """
    Представление дерева в виде скобочной последовательности.

    Args:
        root: Корень дерева

    Returns:
        Строка с скобочным представление
    """
    if root is None:
        return '()'

    result = f'({root.value}'

    if root.left is not None or root.right is not None:
        result += tree_to_bracket(root.left)
        result += tree_to_bracket(root.right)

    result += ')'
    return result


def is_degenerate_tree(root: Optional[TreeNode]) -> bool:
    """
    Проверяет, является ли дерево вырожденным.

    Args:
        root: Корень дерева

    Returns:
        True, если дерево вырожденное
    """
    if root is None:
        return False

    def check_degenerate(node: Optional[TreeNode]) -> bool:
        if node is None:
            return True
        if node.left is not None and node.right is not None:
            return False
        return check_degenerate(node.left) and check_degenerate(node.right)

    return check_degenerate(root)


def print_tree_degenerate(root: Optional[TreeNode]) -> None:
    """
    Специальная визуализация для вырожденных деревьев.

    Args:
        root: Корень вырожденного дерева
    """
    chain = []
    current = root

    while current is not None:
        chain.append(str(current.value))
        if current.left is not None:
            current = current.left
        else:
            current = current.right

    if root.left is not None:
        direction = '←'
    else:
        direction = '→'

    tree_line = ' → '.join(chain) if direction == '→' else ' ← '.join(chain)
    print(tree_line)

    if direction == '←':
        print('Вырожденное дерево (левая ветвь)')
    else:
        print('Вырожденное дерево (правая ветвь)')


def print_tree(root: Optional[TreeNode]) -> None:
    """
    Визуализация дерева.

    Args:
        root: Корень дерева
    """
    if root is None:
        print('(empty)')
        return

    if is_degenerate_tree(root):
        print_tree_degenerate(root)
        return

    levels = []
    queue: deque[tuple[Optional[TreeNode], int]] = deque([(root, 0)])
    max_level = 0

    while queue:
        node, level = queue.popleft()

        if level >= len(levels):
            levels.append([])
            max_level = level

        if node is None:
            levels[level].append('  ')
            if level < max_level:
                queue.append((None, level + 1))
                queue.append((None, level + 1))
            continue
        else:
            levels[level].append(f'{node.value:2d}')

        queue.append((node.left, level + 1))
        queue.append((node.right, level + 1))

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


def display_tree_properties(bst) -> None:
    """
    Отображение свойств дерева.

    Args:
        bst: Бинарное дерево поиска
    """
    min_val = bst.find_min().value if bst.find_min() else 'None'
    max_val = bst.find_max().value if bst.find_max() else 'None'

    print(f'Минимальное значение: {min_val}')
    print(f'Максимальное значение: {max_val}')
    print(f'Высота дерева: {bst.height()}')
    print(f'Корректное BST: {bst.is_valid_bst()}')

    print('\nСкобочное представление:')
    print(tree_to_bracket(bst.root))

    print('\nСтруктура дерева:')
    print_tree(bst.root)
