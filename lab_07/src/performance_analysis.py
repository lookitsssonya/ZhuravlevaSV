"""Экспериментальное исследование производительности."""

import random
import time
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt

from heap import MinHeap
from heapsort import heapsort


def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Измерение времени выполнения функции.

    Args:
        func: Функция для измерения.
        *args: Аргументы функции.
        **kwargs: Ключевые аргументы.

    Returns:
        Кортеж (результат, время выполнения).
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def build_heap_insertion(array: List[int]) -> MinHeap:
    """
    Построение кучи последовательной вставкой.

    Сложность: O(n log n)
    """
    heap = MinHeap()
    for item in array:
        heap.insert(item)
    return heap


def build_heap_algorithm(array: List[int]) -> MinHeap:
    """
    Построение кучи алгоритмом build_heap.

    Сложность: O(n)
    """
    heap = MinHeap()
    heap.build_heap(array)
    return heap


def quicksort(array: List[int]) -> List[int]:
    """
    Быстрая сортировка для сравнения.

    Сложность:
        - В худшем случае: O(n²)
        - В среднем и лучшем: O(n log n)
    """
    if len(array) <= 1:
        return array
    pivot = array[len(array) // 2]
    left = [x for x in array if x < pivot]
    middle = [x for x in array if x == pivot]
    right = [x for x in array if x > pivot]
    return quicksort(left) + middle + quicksort(right)


def mergesort(array: List[int]) -> List[int]:
    """
    Сортировка слиянием для сравнения.

    Сложность: O(n log n)
    """
    if len(array) <= 1:
        return array

    mid = len(array) // 2
    left = mergesort(array[:mid])
    right = mergesort(array[mid:])

    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def run_heap_building_experiment() -> None:
    """Эксперимент по построению кучи разными методами."""
    print('\nПостроение кучи')

    sizes = [100, 500, 1000, 5000, 10000]
    insertion_times = []
    build_heap_times = []

    for size in sizes:
        test_data = [random.randint(1, 10000) for _ in range(size)]

        _, time_insertion = measure_time(build_heap_insertion, test_data)
        insertion_times.append(time_insertion)

        _, time_build_heap = measure_time(build_heap_algorithm, test_data)
        build_heap_times.append(time_build_heap)

        print(f'Размер: {size:5d} | '
              f'Вставка: {time_insertion:.6f} сек | '
              f'Build_Heap: {time_build_heap:.6f} сек')

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, insertion_times, 'ro-',
             label='Последовательная вставка', linewidth=2)
    plt.plot(sizes, build_heap_times, 'bo-',
             label='Алгоритм build_heap', linewidth=2)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (секунды)')
    plt.title('Сравнение методов построения кучи')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heap_building_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def run_sorting_experiment() -> None:
    """Эксперимент по сравнению алгоритмов сортировки."""
    print('\nСравнение алгоритмов сортировки')

    sizes = [100, 500, 1000, 2000, 5000]
    heapsort_times = []
    quicksort_times = []
    mergesort_times = []

    for size in sizes:
        test_data = [random.randint(1, 10000) for _ in range(size)]

        _, time_heapsort = measure_time(heapsort, test_data.copy())
        heapsort_times.append(time_heapsort)

        _, time_quicksort = measure_time(quicksort, test_data.copy())
        quicksort_times.append(time_quicksort)

        _, time_mergesort = measure_time(mergesort, test_data.copy())
        mergesort_times.append(time_mergesort)

        print(f'Размер: {size:5d} | '
              f'Heapsort: {time_heapsort:.6f} сек | '
              f'Quicksort: {time_quicksort:.6f} сек | '
              f'Mergesort: {time_mergesort:.6f} сек')

    plt.figure(figsize=(12, 8))
    plt.plot(sizes, heapsort_times, 'ro-', label='Heapsort', linewidth=2)
    plt.plot(sizes, quicksort_times, 'go-', label='Quicksort', linewidth=2)
    plt.plot(sizes, mergesort_times, 'bo-', label='Mergesort', linewidth=2)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (секунды)')
    plt.title('Сравнение алгоритмов сортировки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sorting_algorithms_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def run_operations_experiment() -> None:
    """Эксперимент по измерению времени операций кучи."""
    print('\nВремя операций кучи')

    sizes = [100, 500, 1000, 5000, 10000]
    insert_times = []
    extract_times = []

    for size in sizes:
        heap = MinHeap()

        start_time = time.perf_counter()
        for i in range(size):
            heap.insert(random.randint(1, 10000))
        insert_time = time.perf_counter() - start_time
        insert_times.append(insert_time / size)

        start_time = time.perf_counter()
        while not heap.is_empty():
            heap.extract()
        extract_time = time.perf_counter() - start_time
        extract_times.append(extract_time / size)

        print(f'Размер: {size:5d} | '
              f'Вставка (средн.): {insert_times[-1]:.8f} сек | '
              f'Извлечение (средн.): {extract_times[-1]:.8f} сек')

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, insert_times, 'go-',
             label='Вставка (среднее)', linewidth=2)
    plt.plot(sizes, extract_times, 'ro-',
             label='Извлечение (среднее)', linewidth=2)
    plt.xlabel('Размер кучи')
    plt.ylabel('Время на операцию (секунды)')
    plt.title('Зависимость времени операций от размера кучи')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heap_operations_time.png',
                dpi=300, bbox_inches='tight')
    plt.show()
