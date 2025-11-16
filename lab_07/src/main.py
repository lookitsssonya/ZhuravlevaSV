"""Главный файл для запуска лабораторной работы по кучам."""

from heap import MaxHeap, MinHeap
from heapsort import heapsort, heapsort_inplace
from performance_analysis import (
    run_heap_building_experiment,
    run_operations_experiment,
    run_sorting_experiment
)
from priority_queue import PriorityQueue
from visualization import print_heap


def system_info() -> None:
    """Вывод информации о системе."""
    print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i5-13420H (2.10 GHz)
- Оперативная память: 16 GB DDR5
- ОС: Windows 11
- Python: 3.11
''')


def demo_heap() -> None:
    """Демонстрация работы кучи."""
    print('Демонстрация работы кучи')

    data = [10, 5, 15, 3, 7]

    print('\n1. MIN-HEAP (последовательная вставка):')
    min_heap = MinHeap()
    for item in data:
        print(f'\nВставляем {item}:')
        min_heap.insert(item)
        print_heap(min_heap)

    print('\nMIN-HEAP (извлечение):')
    while not min_heap.is_empty():
        extracted = min_heap.extract()
        print(f'\nИзвлекаем {extracted}:')
        if not min_heap.is_empty():
            print_heap(min_heap)

    print('\n2. MAX-HEAP (последовательная вставка):')
    max_heap = MaxHeap()
    for item in data:
        print(f'\nВставляем {item}:')
        max_heap.insert(item)
        print_heap(max_heap)

    print('\nMAX-HEAP (извлечение):')
    while not max_heap.is_empty():
        extracted = max_heap.extract()
        print(f'\nИзвлекаем {extracted}:')
        if not max_heap.is_empty():
            print_heap(max_heap)


def demo_heapsort() -> None:
    """Демонстрация пирамидальной сортировки."""
    print('\nДемонстрация сортировки')

    data = [9, 3, 7, 1, 8, 2, 5, 6, 4]
    print(f'Исходный массив: {data}')

    sorted_data = heapsort(data)
    print(f'Отсортированный массив (heapsort): {sorted_data}')

    data_copy = data.copy()
    heapsort_inplace(data_copy)
    print(f'Отсортированный массив (in-place): {data_copy}')


def demo_priority_queue() -> None:
    """Демонстрация приоритетной очереди."""
    print('\nДемонстрация приоритетной очереди')

    pq = PriorityQueue()

    tasks = [
        ('Задача A', 3),
        ('Задача B', 1),
        ('Задача C', 5),
        ('Задача D', 2),
        ('Задача E', 8)
    ]

    print('Добавление задач в очередь:')
    for task, priority in tasks:
        pq.enqueue(task, priority)
        print(f'  Добавлено: "{task}" с приоритетом {priority}')

    print('\nИзвлечение задач по приоритету:')
    while not pq.is_empty():
        task = pq.dequeue()
        print(f'  Выполняется: "{task}"')


def main() -> None:
    """Главная функция программы."""
    system_info()

    demo_heap()
    demo_heapsort()
    demo_priority_queue()

    run_heap_building_experiment()
    run_sorting_experiment()
    run_operations_experiment()

    print('\nГрафики сохранены в файлах:')
    print('heap_building_comparison.png')
    print('sorting_algorithms_comparison.png')
    print('heap_operations_time.png')


if __name__ == "__main__":
    main()
