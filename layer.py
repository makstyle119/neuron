from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, layer_id=None):
        self.layer_id = layer_id or f"{id(self)}"
        self.neurons = [Neuron(num_inputs_per_neuron, neuron_id=f"L{self.layer_id}-N{i+1}") for i in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    def train(self, target, inputs):
        outputs = []
        for neuron in self.neurons:
            out = neuron.forward(inputs)  # Forward pass to get last output
            if neuron.fired:
                neuron.backward(target, inputs)
            outputs.append(out)
        return outputs