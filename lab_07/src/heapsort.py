"""Реализация пирамидальной сортировки (Heapsort)."""

from typing import Any, List
from heap import Heap


def heapsort(array: List[Any], ascending: bool = True) -> List[Any]:
    """
    Сортировка кучей (heapsort).

    Args:
        array: Массив для сортировки.
        ascending: True для сортировки по возрастанию.

    Returns:
        Отсортированный массив.

    Сложность: O(n log n)
    """
    heap = Heap(is_min=ascending)
    heap.build_heap(array)

    sorted_array = []
    while not heap.is_empty():
        sorted_array.append(heap.extract())

    return sorted_array


def heapsort_inplace(array: List[Any]) -> None:
    """
    Сортировка кучей in-place.

    Args:
        array: Массив для сортировки.

    Сложность: O(n log n)
    """
    def _sift_down(arr: List[Any], size: int, idx: int) -> None:
        """Вспомогательная функция для погружения."""
        largest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < size and arr[left] > arr[largest]:
            largest = left

        if right < size and arr[right] > arr[largest]:
            largest = right

        if largest != idx:
            arr[idx], arr[largest] = arr[largest], arr[idx]
            _sift_down(arr, size, largest)

    n = len(array)

    for i in range(n // 2 - 1, -1, -1):
        _sift_down(array, n, i)

    for i in range(n - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        _sift_down(array, i, 0)
