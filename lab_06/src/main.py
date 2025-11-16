from analysis import (
    analyze_performance,
    plot_results,
    system_info,
)
from binary_search_tree import BinarySearchTree
from tree_traversal import get_traversal_results
from visualization import display_tree_properties


def demonstrate_basic_operations():
    """Демонстрация основных операций BST."""
    bst = BinarySearchTree()

    values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
    print(f'Значения: {values}')
    for value in values:
        bst.insert(value)

    display_tree_properties(bst)

    traversals = get_traversal_results(bst.root)
    print('\nРезультаты обходов:')
    print(f"In-order: {traversals['inorder_recursive']}")
    print(f"Pre-order: {traversals['preorder_recursive']}")
    print(f"Post-order: {traversals['postorder_recursive']}")
    print(f"In-order (итеративный): {traversals['inorder_iterative']}")

    print('\nПоиск элементов:')
    test_values = [25, 55, 70, 100]
    for val in test_values:
        found = bst.search(val)
        print(f'Поиск {val}: {"найден" if found else "не найден"}')

    print('\nУдаление элемента:')
    delete_values = [25, 70]
    for val in delete_values:
        print(f'Удаляемый элемент {val}')
        bst.delete(val)
        display_tree_properties(bst)


def demonstrate_tree_types():
    """Демонстрация разных типов деревьев."""
    print('\nДемонстрация разных типов деревьев')

    print('\n1. Сбалансированное дерево:')
    balanced = BinarySearchTree()
    balanced_values = [40, 20, 60, 10, 30, 50, 70, 5, 15, 25, 35]
    for val in balanced_values:
        balanced.insert(val)
    display_tree_properties(balanced)

    print('\n2. Вырожденное дерево:')
    degenerate = BinarySearchTree()
    degenerate_values = [10, 20, 30, 40, 50, 60, 70]
    for val in degenerate_values:
        degenerate.insert(val)
    display_tree_properties(degenerate)


def main():
    """Основная функция."""
    system_info()

    demonstrate_basic_operations()

    demonstrate_tree_types()

    print('\nАнализ производительности')
    results = analyze_performance()
    plot_results(results)

    print('\nВыводы:')
    print('1. Сбалансированные деревья показывают производительность '
          'O(log n) для поиска и удаления')
    print('2. Вырожденные деревья деградируют до O(n) для поиска '
          'и удаления')
    print('3. Все обходы имеют сложность O(n) независимо от структуры '
          'дерева')
    print('4. Итеративный обход обычно быстрее рекурсивных из-за '
          'отсутствия накладных расходов на вызовы функций')


if __name__ == '__main__':
    main()
