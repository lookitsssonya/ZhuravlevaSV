"""Модуль реализации бинарного дерева поиска."""

from __future__ import annotations

from collections import deque
from typing import Optional


class TreeNode:
    """Узел бинарного дерева поиска."""

    def __init__(self, value: int) -> None:
        """
        Инициализация узла.

        Args:
            value: Значение узла
        """
        self.value: int = value
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None


class BinarySearchTree:
    """Бинарное дерево поиска."""

    def __init__(self) -> None:
        """Инициализация пустого дерева."""
        self.root: Optional[TreeNode] = None

    def insert(self, value: int) -> None:
        """
        Вставка значения в дерево.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            value: Значение для вставки
        """
        new_node = TreeNode(value)

        if self.root is None:
            self.root = new_node
            return

        current = self.root
        while True:
            if value < current.value:
                if current.left is None:
                    current.left = new_node
                    return
                current = current.left
            elif value > current.value:
                if current.right is None:
                    current.right = new_node
                    return
                current = current.right
            else:
                return

    def search(self, value: int) -> bool:
        """
        Поиск значения в дереве.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            value: Значение для поиска

        Returns:
            True, если значение найдено, иначе False
        """
        current = self.root

        while current is not None:
            if value == current.value:
                return True
            elif value < current.value:
                current = current.left
            else:
                current = current.right

        return False

    def delete(self, value: int) -> None:
        """
        Удаление значения из дерева.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            value: Значение для удаления
        """
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(
        self, node: Optional[TreeNode], value: int
    ) -> Optional[TreeNode]:
        """
        Рекурсивное удаление значения.

        Args:
            node: Текущий узел
            value: Значение для удаления

        Returns:
            Обновленный узел
        """
        if node is None:
            return None

        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            min_node = self._find_min(node.right)
            node.value = min_node.value
            node.right = self._delete_recursive(node.right, min_node.value)

        return node

    @staticmethod
    def _find_min(node: TreeNode) -> TreeNode:
        """
        Поиск минимального узла в поддереве.

        Args:
            node: Корень поддерева

        Returns:
            Узел с минимальным значением
        """
        while node.left is not None:
            node = node.left
        return node

    def find_min(self, node: Optional[TreeNode] = None) -> Optional[TreeNode]:
        """
        Поиск узла с минимальным значением в поддереве.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            node: Узел для начала поиска (корень поддерева)

        Returns:
            Узел с минимальным значением или None
        """
        if node is None:
            if self.root is None:
                return None
            node = self.root

        while node.left is not None:
            node = node.left
        return node

    def find_max(self, node: Optional[TreeNode] = None) -> Optional[TreeNode]:
        """
        Поиск узла с максимальным значением в поддереве.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            node: Узел для начала поиска (корень поддерева)

        Returns:
            Узел с максимальным значением или None
        """
        if node is None:
            if self.root is None:
                return None
            node = self.root

        while node.right is not None:
            node = node.right
        return node

    def height(self, node: Optional[TreeNode] = None) -> int:
        """
        Вычисление высоты дерева/поддерева.

        Сложность: O(n) - необходимо посетить все узлы

        Args:
            node: Узел для вычисления высоты (корень поддерева)

        Returns:
            Высота дерева/поддерева
        """
        if node is None:
            if self.root is None:
                return 0
            node = self.root

        queue = deque()
        queue.append((node, 1))
        max_height = 0

        while queue:
            current_node, level = queue.popleft()
            max_height = max(max_height, level)

            if current_node.left is not None:
                queue.append((current_node.left, level + 1))
            if current_node.right is not None:
                queue.append((current_node.right, level + 1))

        return max_height

    def is_valid_bst(self) -> bool:
        """
        Проверка, является ли дерево корректным BST.

        Сложность: O(n) - необходимо посетить все узлы

        Returns:
            True, если дерево корректно, иначе False
        """
        if self.root is None:
            return True

        stack = []
        prev_value = float('-inf')
        current = self.root

        while current is not None or stack:
            while current is not None:
                stack.append(current)
                current = current.left

            current = stack.pop()

            if current.value <= prev_value:
                return False
            prev_value = current.value

            current = current.right

        return True
