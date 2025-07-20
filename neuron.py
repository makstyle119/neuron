import random
import math

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0
        self.inputs = []
        self.delta = 0

    def activate(self, inputs):
        self.inputs = inputs
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = 1 / (1 + math.exp(-total))
        return self.output

    def derivative(self):
        return self.output * (1 - self.output)