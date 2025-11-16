"""Модуль для анализа производительности BST."""

import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from binary_search_tree import BinarySearchTree
from tree_traversal import (
    inorder_iterative,
    inorder_recursive,
    postorder_recursive,
    preorder_recursive,
)


def system_info() -> None:
    """Вывод информации о системе."""
    print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i5-13420H (2.10 GHz)
- Оперативная память: 16 GB DDR5
- ОС: Windows 11
- Python: 3.11
''')


def generate_balanced_tree(size: int) -> BinarySearchTree:
    """
    Генерация сбалансированного дерева.

    Args:
        size: Количество элементов

    Returns:
        Сбалансированное BST
    """
    bst = BinarySearchTree()
    values = list(range(size))
    random.shuffle(values)

    for value in values:
        bst.insert(value)

    return bst


def generate_degenerate_tree(size: int) -> BinarySearchTree:
    """
    Генерация вырожденного дерева.

    Args:
        size: Количество элементов

    Returns:
        Вырожденное BST
    """
    bst = BinarySearchTree()
    for value in range(size):
        bst.insert(value)

    return bst


def measure_search_performance(
    bst: BinarySearchTree,
    operation_count: int = 100
) -> float:
    """
    Измерение времени выполнения операций поиска.

    Args:
        bst: Дерево для тестирования
        operation_count: Количество операций поиска

    Returns:
        Среднее время поиска в секундах
    """
    max_value = bst.height() * 2
    values_to_search = [
        random.randint(0, max_value) for _ in range(operation_count)
    ]

    start_time = time.perf_counter()

    for value in values_to_search:
        bst.search(value)

    end_time = time.perf_counter()

    return (end_time - start_time) / operation_count


def measure_delete_performance(
    bst: BinarySearchTree,
    operation_count: int = 50
) -> float:
    """
    Измерение времени выполнения операций удаления.

    Args:
        bst: Дерево для тестирования
        operation_count: Количество операций удаления

    Returns:
        Среднее время удаления в секундах
    """
    max_value = bst.height() * 2
    values_to_delete = [
        random.randint(0, max_value) for _ in range(operation_count)
    ]

    total_time = 0

    for value in values_to_delete:
        test_tree = BinarySearchTree()

        def copy_tree(node):
            if node:
                test_tree.insert(node.value)
                copy_tree(node.left)
                copy_tree(node.right)

        copy_tree(bst.root)

        start_time = time.perf_counter()
        test_tree.delete(value)
        end_time = time.perf_counter()

        total_time += (end_time - start_time)

    return total_time / operation_count


def measure_traversal_performance(
    bst: BinarySearchTree,
    operation_count: int = 10
) -> Dict[str, float]:
    """
    Измерение времени выполнения различных обходов дерева.

    Args:
        bst: Дерево для тестирования
        operation_count: Количество повторений для усреднения

    Returns:
        Словарь с временами выполнения обходов
    """
    results = {}

    total_time = 0
    for _ in range(operation_count):
        result_list = []
        start_time = time.perf_counter()
        inorder_recursive(bst.root, result_list)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    results['inorder_recursive'] = total_time / operation_count

    total_time = 0
    for _ in range(operation_count):
        result_list = []
        start_time = time.perf_counter()
        preorder_recursive(bst.root, result_list)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    results['preorder_recursive'] = total_time / operation_count

    total_time = 0
    for _ in range(operation_count):
        result_list = []
        start_time = time.perf_counter()
        postorder_recursive(bst.root, result_list)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    results['postorder_recursive'] = total_time / operation_count

    total_time = 0
    for _ in range(operation_count):
        start_time = time.perf_counter()
        inorder_iterative(bst.root)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    results['inorder_iterative'] = total_time / operation_count

    return results


def analyze_performance() -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Анализ производительности для деревьев разного размера и структуры.

    Returns:
        Словарь с результатами анализа
    """
    sizes = [50, 100, 150, 200, 250]
    results = {
        'balanced': {
            'search': [],
            'delete': [],
            'inorder_recursive': [],
            'preorder_recursive': [],
            'postorder_recursive': [],
            'inorder_iterative': []
        },
        'degenerate': {
            'search': [],
            'delete': [],
            'inorder_recursive': [],
            'preorder_recursive': [],
            'postorder_recursive': [],
            'inorder_iterative': []
        }
    }

    for size in sizes:
        print(f'Анализ для размера {size}')

        try:
            balanced_tree = generate_balanced_tree(size)
            balanced_height = balanced_tree.height()
            balanced_time = measure_search_performance(balanced_tree, 100)
            delete_time = measure_delete_performance(balanced_tree, 20)
            traversal_times = measure_traversal_performance(balanced_tree, 5)

            results['balanced']['search'].append((size, balanced_time))
            results['balanced']['delete'].append((size, delete_time))
            results['balanced']['inorder_recursive'].append(
                (size, traversal_times['inorder_recursive'])
            )
            results['balanced']['preorder_recursive'].append(
                (size, traversal_times['preorder_recursive'])
            )
            results['balanced']['postorder_recursive'].append(
                (size, traversal_times['postorder_recursive'])
            )
            results['balanced']['inorder_iterative'].append(
                (size, traversal_times['inorder_iterative'])
            )

            print(f'  Сбалансированное: время={balanced_time:.6f}с, '
                  f'высота={balanced_height}')

            degenerate_tree = generate_degenerate_tree(size)
            degenerate_height = degenerate_tree.height()
            degenerate_time = measure_search_performance(degenerate_tree, 100)
            delete_time = measure_delete_performance(degenerate_tree, 20)
            traversal_times = measure_traversal_performance(degenerate_tree, 5)

            results['degenerate']['search'].append((size, degenerate_time))
            results['degenerate']['delete'].append((size, delete_time))
            results['degenerate']['inorder_recursive'].append(
                (size, traversal_times['inorder_recursive'])
            )
            results['degenerate']['preorder_recursive'].append(
                (size, traversal_times['preorder_recursive'])
            )
            results['degenerate']['postorder_recursive'].append(
                (size, traversal_times['postorder_recursive'])
            )
            results['degenerate']['inorder_iterative'].append(
                (size, traversal_times['inorder_iterative'])
            )

            print(f'  Вырожденное: время={degenerate_time:.6f}с, '
                  f'высота={degenerate_height}')

        except Exception as e:
            print(f'  Ошибка при размере {size}: {e}')
            continue

    return results


def plot_results(
    results: Dict[str, Dict[str, List[Tuple[int, float]]]]
) -> None:
    """
    Построение графиков зависимости времени операций от количества элементов.

    Args:
        results: Результаты анализа производительности
    """
    operations = [
        'search', 'delete', 'inorder_recursive', 'preorder_recursive',
        'postorder_recursive', 'inorder_iterative'
    ]
    operation_names = {
        'search': 'Поиск элемента',
        'delete': 'Удаление элемента',
        'inorder_recursive': 'In-order (рекурсивный)',
        'preorder_recursive': 'Pre-order (рекурсивный)',
        'postorder_recursive': 'Post-order (рекурсивный)',
        'inorder_iterative': 'In-order (итеративный)'
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, operation in enumerate(operations):
        ax = axes[i]

        balanced_data = results['balanced'][operation]
        if balanced_data:
            balanced_sizes = [x[0] for x in balanced_data]
            balanced_times = [x[1] * 1000000 for x in balanced_data]
            ax.plot(
                balanced_sizes, balanced_times, 'o-',
                label='Сбалансированное дерево',
                linewidth=2, markersize=6, color='blue'
            )

        degenerate_data = results['degenerate'][operation]
        if degenerate_data:
            degenerate_sizes = [x[0] for x in degenerate_data]
            degenerate_times = [x[1] * 1000000 for x in degenerate_data]
            ax.plot(
                degenerate_sizes, degenerate_times, 's-',
                label='Вырожденное дерево',
                linewidth=2, markersize=6, color='red'
            )

        ax.set_xlabel('Количество элементов')
        ax.set_ylabel('Время (микросекунды)')
        ax.set_title(operation_names[operation])
        ax.legend()
        ax.grid(True, alpha=0.3)

        if operation in ['search', 'delete']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def show_example_trees() -> None:
    """Примеры деревьев разной структуры."""
    print('Примеры деревьев разной структуры')

    print('\n1. Сбалансированное дерево:')
    balanced = BinarySearchTree()
    balanced_values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35]
    for val in balanced_values:
        balanced.insert(val)

    from visualization import display_tree_properties
    display_tree_properties(balanced)

    print('\n2. Вырожденное дерево:')
    degenerate = BinarySearchTree()
    degenerate_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for val in degenerate_values:
        degenerate.insert(val)
    display_tree_properties(degenerate)
