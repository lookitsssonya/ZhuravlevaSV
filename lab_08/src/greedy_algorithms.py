"""Реализация классических жадных алгоритмов."""

import heapq
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class IntervalScheduling:
    """Класс для решения задачи о выборе заявок."""

    @staticmethod
    def select_intervals(
        intervals: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Выбирает максимальное количество непересекающихся интервалов.

        Args:
            intervals: Список интервалов в формате (start, end)

        Returns:
            List[Tuple[float, float]]: Список выбранных интервалов

        Сложность: O(n log n)
        """
        sorted_intervals = sorted(intervals, key=lambda x: x[1])

        selected: List[Tuple[float, float]] = []
        last_end: float = -float('inf')

        for interval in sorted_intervals:
            start, end = interval
            if start >= last_end:
                selected.append(interval)
                last_end = end

        return selected


class FractionalKnapsack:
    """Класс для решения непрерывной задачи о рюкзаке."""

    @staticmethod
    def solve_fractional_knapsack(
        items: List[Tuple[float, float]],
        capacity: float
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Решает непрерывную задачу о рюкзаке.

        Args:
            items: Список предметов в формате (weight, value)
            capacity: Вместимость рюкзака

        Returns:
            Tuple[float, List[Tuple[float, float]]]:
            (максимальная стоимость, список взятых предметов)

        Сложность: O(n log n)
        """
        items_with_ratio = [
            (value / weight, weight, value) for weight, value in items
        ]
        items_with_ratio.sort(reverse=True)

        total_value: float = 0.0
        taken_items: List[Tuple[float, float]] = []
        remaining_capacity: float = capacity

        for ratio, weight, value in items_with_ratio:
            if remaining_capacity >= weight:
                taken_items.append((weight, value))
                total_value += value
                remaining_capacity -= weight
            else:
                fraction = remaining_capacity / weight
                taken_items.append((remaining_capacity, value * fraction))
                total_value += value * fraction
                break

        return total_value, taken_items


class HuffmanCoding:
    """Класс для реализации алгоритма Хаффмана."""

    class Node:
        """Узел дерева Хаффмана."""

        def __init__(
            self,
            char: Optional[str],
            freq: float,
            left: Optional['HuffmanCoding.Node'] = None,
            right: Optional['HuffmanCoding.Node'] = None
        ):
            self.char: Optional[str] = char
            self.freq: float = freq
            self.left: Optional['HuffmanCoding.Node'] = left
            self.right: Optional['HuffmanCoding.Node'] = right

        def __lt__(self, other: 'HuffmanCoding.Node') -> bool:
            return self.freq < other.freq

    @staticmethod
    def build_huffman_tree(
        frequencies: Dict[str, float]
    ) -> 'HuffmanCoding.Node':
        """
        Строит дерево Хаффмана.

        Args:
            frequencies: Словарь частот символов

        Returns:
            Node: Корень дерева Хаффмана

        Сложность: O(n log n)
        """
        heap = [
            HuffmanCoding.Node(char, freq) for char,
            freq in frequencies.items()
        ]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanCoding.Node(
                None, left.freq + right.freq, left, right
            )
            heapq.heappush(heap, merged)

        return heap[0]

    @staticmethod
    def generate_codes(root: 'HuffmanCoding.Node') -> Dict[str, str]:
        """
        Генерирует коды Хаффмана для символов.

        Args:
            root: Корень дерева Хаффмана

        Returns:
            Dict[str, str]: Словарь кодов для символов

        Сложность: O(n)
        """
        codes: Dict[str, str] = {}

        def traverse(node: Optional['HuffmanCoding.Node'], code: str) -> None:
            if node is None:
                return

            if node.char is not None:
                codes[node.char] = code
                return

            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

        traverse(root, '')
        return codes

    @staticmethod
    def huffman_encode(text: str) -> Tuple[Dict[str, str], str]:
        """
        Кодирует текст с помощью алгоритма Хаффмана.

        Args:
            text: Исходный текст

        Returns:
            Tuple[Dict[str, str], str]: (словарь кодов, закодированный текст)

        Сложность: O(n log n)
        """
        if not text:
            return {}, ''

        frequencies = Counter(text)
        root = HuffmanCoding.build_huffman_tree(frequencies)
        codes = HuffmanCoding.generate_codes(root)

        encoded_text = ''.join(codes[char] for char in text)
        return codes, encoded_text

    @staticmethod
    def visualize_tree(
        root: 'HuffmanCoding.Node', filename: str = 'huffman_tree.png'
    ) -> None:
        """
        Визуализирует дерево Хаффмана.

        Args:
            root: Корень дерева Хаффмана
            filename: Имя файла для сохранения изображения
        """
        graph = nx.DiGraph()
        pos = {}

        def add_edges(
            node: Optional['HuffmanCoding.Node'], x: float, y: float, dx: float
        ) -> None:
            if node is None:
                return

            label = (
                f'{node.char}:{node.freq}'
                if node.char else f'{node.freq:.2f}'
            )
            graph.add_node(id(node), label=label)
            pos[id(node)] = (x, y)

            if node.left:
                graph.add_edge(id(node), id(node.left))
                add_edges(node.left, x - dx, y - 1, dx / 2)

            if node.right:
                graph.add_edge(id(node), id(node.right))
                add_edges(node.right, x + dx, y - 1, dx / 2)

        add_edges(root, 0, 0, 2)

        plt.figure(figsize=(12, 8))
        labels = {
            node: data['label'] for node, data in graph.nodes(data=True)
        }
        nx.draw(
            graph, pos, labels=labels, with_labels=True, node_size=2000,
            node_color='lightblue', font_size=8, arrows=False
        )
        plt.title('Дерево кодов Хаффмана')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


class CoinChange:
    """Класс для решения задачи о минимальном количестве монет."""

    @staticmethod
    def min_coins_greedy(amount: int, coins: List[int]) -> Dict[int, int]:
        """
        Решает задачу о минимальном количестве монет жадным алгоритмом.

        Args:
            amount: Сумма для размена (в минимальных единицах - копейках)
            coins: Доступные номиналы монет

        Returns:
            Dict[int, int]: Словарь {номинал: количество}

        Сложность: O(n)
        """
        coins_sorted = sorted(coins, reverse=True)
        result: Dict[int, int] = {}
        remaining = amount

        for coin in coins_sorted:
            if remaining >= coin:
                count = remaining // coin
                result[coin] = count
                remaining -= count * coin

        if remaining > 0:
            raise ValueError(
                f'Невозможно разменять сумму {amount} с данными номиналами'
            )

        return result


class PrimMST:
    """Класс для нахождения минимального остовного дерева алгоритмом Прима."""

    @staticmethod
    def prim_algorithm(
        graph: Dict[str, List[Tuple[str, float]]]
    ) -> List[Tuple[str, str, float]]:
        """
        Находит минимальное остовное дерево алгоритмом Прима.

        Args:
            graph: Граф в формате {вершина: [(сосед, вес), ...]}

        Returns:
            List[Tuple[str, str, float]]: Список рёбер MST

        Сложность: O(E log V), где E - количество ребер, V - количество вершин
        """
        if not graph:
            return []

        start_vertex = next(iter(graph.keys()))
        mst_edges: List[Tuple[str, str, float]] = []
        visited = {start_vertex}
        edges = [
            (weight, start_vertex, neighbor)
            for neighbor, weight in graph[start_vertex]
        ]
        heapq.heapify(edges)

        while edges and len(visited) < len(graph):
            weight, src, dest = heapq.heappop(edges)

            if dest not in visited:
                visited.add(dest)
                mst_edges.append((src, dest, weight))

                for neighbor, n_weight in graph[dest]:
                    if neighbor not in visited:
                        heapq.heappush(edges, (n_weight, dest, neighbor))

        return mst_edges
