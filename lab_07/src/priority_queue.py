"""Реализация приоритетной очереди на основе кучи."""

from typing import Any, Optional
from heap import Heap


class PriorityItem:
    """Элемент приоритетной очереди."""

    def __init__(self, item: Any, priority: float) -> None:
        """
        Инициализация элемента.

        Args:
            item: Объект.
            priority: Приоритет.
        """
        self.item = item
        self.priority = priority

    def __lt__(self, other: 'PriorityItem') -> bool:
        """Сравнение для min-heap (меньший приоритет - выше)."""
        return self.priority < other.priority

    def __gt__(self, other: 'PriorityItem') -> bool:
        """Сравнение для max-heap."""
        return self.priority > other.priority

    def __repr__(self) -> str:
        """Строковое представление."""
        return f'PriorityItem(item={self.item}, priority={self.priority})'


class PriorityQueue:
    """Приоритетная очередь на основе min-heap."""

    def __init__(self) -> None:
        """Инициализация приоритетной очереди."""
        self._heap = Heap(is_min=True)

    def enqueue(self, item: Any, priority: float) -> None:
        """
        Добавление элемента в очередь. Сложность: O(log n).

        Args:
            item: Объект для добавления.
            priority: Приоритет объекта.
        """
        self._heap.insert(PriorityItem(item, priority))

    def dequeue(self) -> Optional[Any]:
        """
        Извлечение элемента с наивысшим приоритетом. Сложность: O(log n).

        Returns:
            Элемент или None, если очередь пуста.
        """
        priority_item = self._heap.extract()
        return priority_item.item if priority_item else None

    def peek(self) -> Optional[Any]:
        """
        Просмотр элемента с наивысшим приоритетом. Сложность: O(1).

        Returns:
            Элемент или None, если очередь пуста.
        """
        priority_item = self._heap.peek()
        return priority_item.item if priority_item else None

    def is_empty(self) -> bool:
        """
        Проверка пустоты очереди. Сложность: O(1).

        Returns:
            True, если очередь пуста.
        """
        return self._heap.is_empty()

    def size(self) -> int:
        """
        Размер очереди. Сложность: O(1).

        Returns:
            Количество элементов в очереди.
        """
        return self._heap.size()
