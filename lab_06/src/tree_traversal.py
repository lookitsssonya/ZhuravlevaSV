"""Модуль реализации методов обхода дерева."""

from typing import List, Optional

from binary_search_tree import TreeNode


def inorder_recursive(node: Optional[TreeNode], result: List[int]) -> None:
    """
    Рекурсивный in-order обход (левый-корень-правый).

    Сложность: O(n)

    Args:
        node: Текущий узел
        result: Список для сохранения результатов
    """
    if node is not None:
        inorder_recursive(node.left, result)
        result.append(node.value)
        inorder_recursive(node.right, result)


def preorder_recursive(node: Optional[TreeNode], result: List[int]) -> None:
    """
    Рекурсивный pre-order обход (корень-левый-правый).

    Сложность: O(n)

    Args:
        node: Текущий узел
        result: Список для сохранения результатов
    """
    if node is not None:
        result.append(node.value)
        preorder_recursive(node.left, result)
        preorder_recursive(node.right, result)


def postorder_recursive(node: Optional[TreeNode], result: List[int]) -> None:
    """
    Рекурсивный post-order обход (левый-правый-корень).

    Сложность: O(n)

    Args:
        node: Текущий узел
        result: Список для сохранения результатов
    """
    if node is not None:
        postorder_recursive(node.left, result)
        postorder_recursive(node.right, result)
        result.append(node.value)


def inorder_iterative(root: Optional[TreeNode]) -> List[int]:
    """
    Итеративный in-order обход.

    Сложность: O(n)

    Args:
        root: Корень дерева

    Returns:
        Список значений в порядке in-order
    """
    result: List[int] = []
    stack: List[TreeNode] = []
    current: Optional[TreeNode] = root

    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.value)
        current = current.right

    return result


def get_traversal_results(root: Optional[TreeNode]) -> dict:
    """
    Получение результатов всех видов обходов.

    Args:
        root: Корень дерева

    Returns:
        Словарь с результатами всех обходов
    """
    result: dict = {
        'inorder_recursive': [],
        'preorder_recursive': [],
        'postorder_recursive': [],
        'inorder_iterative': []
    }

    if root is not None:
        inorder_recursive(root, result['inorder_recursive'])
        preorder_recursive(root, result['preorder_recursive'])
        postorder_recursive(root, result['postorder_recursive'])
        result['inorder_iterative'] = inorder_iterative(root)

    return result
