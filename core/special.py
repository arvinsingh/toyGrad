import math

import numpy as np


class Special():
    def __init__(self, value, _prev: set = None, _op: str = '', requires_grad=False):
        self.data = value
        self.grad = 0.0
        self._prev = _prev if _prev is not None else set()
        self._op = _op
        self.requires_grad = requires_grad

        self._backward = lambda: None

    def __repr__(self):
        return f'Special(data={self.data}, requires_grad={self.requires_grad})'

    def __add__(self, other):
        other = other if isinstance(other, Special) else Special(other)
        out = Special(self.data + other.data, (self, other), '+', self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Special) else Special(other)
        out = Special(self.data * other.data, (self, other), '*', self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Special(self.data ** other, (self,), f'**{other}', self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = self * other ** -1
        return out

    def exp(self):
        out = Special(math.exp(self.data), (self, ), 'exp', self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        eps = 1e-8 # avoid log(0)
        out = Special(math.log(self.data + eps), (self,), 'log', self.requires_grad)

        def _backward():
            self.grad += (1 / (self.data + eps)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Special(t, (self,), 'tanh')

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Special(self.data if self.data > 0 else 0.0, (self,), 'relu', self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        sig = 1 / (1 + math.exp(-self.data))
        out = Special(sig, (self,), 'sigmoid')

        def _backward():
            if self.requires_grad:
                self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Topological sort and calculating gradient for each node in order.
        """
        topological_order = list()
        visited = set()

        def build_topological_graph(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topological_graph(child)
                topological_order.append(node)

        build_topological_graph(self)

        self.grad = 1.0
        for nodes in reversed(topological_order):
            nodes._backward()


class Param(Special):
    def __init__(self, value):
        super().__init(value)
        self.requires_grad = True

    def __repr__(self):
        return f'Param(data={self.data}, grad={self.grad})'


class SpecialTensor:
    def __init__(self, values, _prev=None, _op='', requires_grad=False):
        self.data = np.array(values, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = _prev if _prev is not None else set()
        self._op = _op
        self.requires_grad = requires_grad

        self._backward = lambda: None

    def __repr__(self):
        return f'SpecialTensor(shape={self.data.shape}, data={self.data})'

    def __add__(self, other):
        other = other if isinstance(other, SpecialTensor) else SpecialTensor(other)
        out = SpecialTensor(self.data + other.data, {self, other}, '+', self.requires_grad or other.requies_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, SpecialTensor) else SpecialTensor(other)
        out = SpecialTensor(self.data * other.data, {self, other}, '*', self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = SpecialTensor(self.data ** other, {self, other}, f'**{other}', self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other ** -1
    
    def matmul(self, other):
        other = other if isinstance(other, SpecialTensor) else SpecialTensor(other)
        out = SpecialTensor(np.dot(self.data, other.data), {self, other}, '@', self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        return out

    def transpose(self, *dims):
        if len(dims) == 0:
            # default transpose -- reverse all dimensions
            new_data = self.data.T
        elif len(dims) == 2:
            # transpose specific dimensions
            axis1, axis2 = dims
            new_data = np.swapaxes(self.data, axis1, axis2)
        else:
            raise ValueError("transpose() takes 0 or 2 arguments")

        out = SpecialTensor(new_data, {self}, 'T', self.requires_grad)

        def _backward():
            if self.requires_grad:
                if len(dims) == 0:
                    self.grad += out.grad.T
                else:
                    self.grad += np.swapaxes(out.grad, axis1, axis2)

        out._backward = _backward
        return out

    def relu(self):
        out = SpecialTensor(np.maximum(0, self.data), {self}, 'relu', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        sigmoid_data = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))  # cliping for numerical stability
        out = SpecialTensor(sigmoid_data, {self}, 'sigmoid', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += sigmoid_data * (1 - sigmoid_data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topological_order = []
        visited = set()

        def build_topological_graph(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topological_graph(child)
                topological_order.append(child)

        build_topological_graph(self)

        self.grad = np.ones_like(self.data)
        for nodes in reversed(topological_order):
            nodes._backward()


