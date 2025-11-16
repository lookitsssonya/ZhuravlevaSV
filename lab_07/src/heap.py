"""Реализация структуры данных Куча."""

from typing import Any, List, Optional


class Heap:
    """Универсальная куча (min-heap или max-heap)."""

    def __init__(self, is_min: bool = True) -> None:
        """
        Инициализация кучи.

        Args:
            is_min: True для min-heap, False для max-heap.
        """
        self._heap: List[Any] = []
        self.is_min = is_min

    def get_heap_array(self) -> List[Any]:
        """
        Получение массива кучи.

        Returns:
            Внутренний массив кучи.
        """
        return self._heap.copy()

    def _compare(self, a: Any, b: Any) -> bool:
        """
        Сравнение элементов в зависимости от типа кучи.

        Args:
            a: Первый элемент.
            b: Второй элемент.

        Returns:
            True, если a имеет приоритет над b.
        """
        if self.is_min:
            return a < b
        else:
            return a > b

    def _sift_up(self, index: int) -> None:
        """
        Всплытие элемента (sift-up). Сложность: O(log n).

        Args:
            index: Индекс элемента для всплытия.
        """
        if index == 0:
            return

        parent_index = (index - 1) // 2
        if self._compare(self._heap[index], self._heap[parent_index]):
            self._heap[index], self._heap[parent_index] = (
                self._heap[parent_index], self._heap[index]
            )
            self._sift_up(parent_index)

    def _sift_down(self, index: int) -> None:
        """
        Погружение элемента (sift-down). Сложность: O(log n).

        Args:
            index: Индекс элемента для погружения.
        """
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        priority_index = index

        if (left_child_index < len(self._heap) and
                self._compare(self._heap[left_child_index],
                              self._heap[priority_index])):
            priority_index = left_child_index

        if (right_child_index < len(self._heap) and
                self._compare(self._heap[right_child_index],
                              self._heap[priority_index])):
            priority_index = right_child_index

        if priority_index != index:
            self._heap[index], self._heap[priority_index] = (
                self._heap[priority_index], self._heap[index]
            )
            self._sift_down(priority_index)

    def insert(self, value: Any) -> None:
        """
        Вставка элемента в кучу. Сложность: O(log n).

        Args:
            value: Значение для вставки.
        """
        self._heap.append(value)
        self._sift_up(len(self._heap) - 1)

    def extract(self) -> Optional[Any]:
        """
        Извлечение корневого элемента. Сложность: O(log n).

        Returns:
            Корневой элемент или None, если куча пуста.
        """
        if not self._heap:
            return None

        if len(self._heap) == 1:
            return self._heap.pop()

        root = self._heap[0]
        self._heap[0] = self._heap.pop()
        self._sift_down(0)
        return root

    def peek(self) -> Optional[Any]:
        """
        Просмотр корневого элемента. Сложность: O(1).

        Returns:
            Корневой элемент или None, если куча пуста.
        """
        return self._heap[0] if self._heap else None

    def build_heap(self, array: List[Any]) -> None:
        """
        Построение кучи из массива. Сложность: O(n).

        Args:
            array: Массив для построения кучи.
        """
        self._heap = array.copy()
        for i in range(len(self._heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    def size(self) -> int:
        """
        Получение размера кучи. Сложность: O(1).

        Returns:
            Количество элементов в куче.
        """
        return len(self._heap)

    def is_empty(self) -> bool:
        """
        Проверка пустоты кучи. Сложность: O(1).

        Returns:
            True, если куча пуста.
        """
        return len(self._heap) == 0

    def __str__(self) -> str:
        """Строковое представление кучи."""
        return str(self._heap)


class MinHeap(Heap):
    """Min-Heap специализация."""

    def __init__(self) -> None:
        """Инициализация min-heap."""
        super().__init__(is_min=True)


class MaxHeap(Heap):
    """Max-Heap специализация."""

    def __init__(self) -> None:
        """Инициализация max-heap."""
        super().__init__(is_min=False)
