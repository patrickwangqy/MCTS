from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any
from functools import total_ordering
import random
import tqdm
from abc import ABCMeta, abstractmethod
import graphviz as gz


@total_ordering
@dataclass
class Node:
    state: Any
    untried_actions: List[int]
    label: int
    parent: Optional[Node] = None
    children: List[Node] = field(default_factory=list)
    action: Optional[int] = None
    n: int = 0
    value: float = 0.0

    def __eq__(self, other: Node) -> bool:
        return self.value == other.value

    def __lt__(self, other: Node) -> bool:
        return self.value > other.value

    def __str__(self):
        return str(self.state)

    def __expr__(self):
        return str(self)

    def add_child(self, node: Node, action: int):
        node.parent = self
        node.action = action
        self.children.append(node)
        self.untried_actions.remove(action)


class MCTSEnv(metaclass=ABCMeta):
    @abstractmethod
    def initial_state(self):
        raise NotImplementedError()

    @abstractmethod
    def do_action(self, state: Any, action: int) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def calc_terminate_value(self, state: Any) -> float:
        raise NotImplementedError()

    @abstractmethod
    def avaliable_actions(self, state: Any) -> List[int]:
        raise NotImplementedError()


class MCTSTree(metaclass=ABCMeta):
    def __init__(self, env: MCTSEnv):
        self.env = env
        self.root = self.create_node(self.env.initial_state(), 1)
        self.nodes_num = 1
        self.best_state = None
        self.best_value = -float("inf")

    def search(self, max_iters, c: float = 1.0):
        for _ in tqdm.trange(max_iters):
            node = self.tree_policy(self.root, c)
            result = self.default_policy(node)
            self.backup(node, result)

    def tree_policy(self, node: Node, c: float) -> Node:
        while not self.env.is_terminal(node.state):
            if len(node.untried_actions) > 0:
                return self.expand(node)
            node = self.best_child(node, c)
        return node

    def expand(self, node: Node) -> Node:
        action = node.untried_actions[0]
        self.nodes_num += 1
        new_state = self.env.do_action(node.state, action)
        new_node = self.create_node(new_state, self.nodes_num)
        node.add_child(new_node, action)
        return new_node

    def best_child(self, node: Node, c: float) -> Node:
        values = np.array(list(map(lambda x: x.value / x.n + c * np.sqrt(2 * np.log(node.n) / x.n), node.children)), dtype=np.float32)
        max_i = np.argmax(values)
        return node.children[max_i]

    def default_policy(self, node: Node):
        while not self.env.is_terminal(node.state):
            actions = self.env.avaliable_actions(node)
            action = random.choice(actions)
            new_state = self.env.do_action(node.state, action)
            node = self.create_node(new_state)
        value = self.env.calc_terminate_value(node.state)
        if value > self.best_value:
            self.best_state = node.state
            self.best_value = value
        return value

    def backup(self, node: Node, value: float):
        while node is not None:
            node.n += 1
            node.value += value
            node = node.parent

    def create_node(self, state: Any, label=0) -> Node:
        node = Node(state, self.env.avaliable_actions(state), label)
        return node

    def show(self):
        graph = gz.Graph()
        graph.attr("graph", nodesep="0")
        graph.attr("node", shape="point")
        queue = self.root.children[:]
        while len(queue) > 0:
            node = queue.pop()
            graph.edge(str(node.parent.label), str(node.label))
            for node in node.children:
                queue.append(node)
        return graph
