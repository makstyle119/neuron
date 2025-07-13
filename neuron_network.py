from layer import Layer

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layers = []
        self.learning_rate = learning_rate
        for i in range(1, len(layer_sizes)):
            layer = Layer(num_neurons=layer_sizes[i], num_inputs_per_neuron=layer_sizes[i - 1], layer_id=i)
            self.layers.append(layer)

    def forward(self, inputs):
        outputs = []
        for layer in self.layers:
            inputs = layer.forward(inputs)
            outputs.append(inputs)  # keep track of each layer's output
        return outputs[-1]  # final output only
    
    def train(self, inputs, target):
        # Step 1: Forward pass
        layer_inputs = [inputs]
        for layer in self.layers:
            output = layer.forward(layer_inputs[-1])
            layer_inputs.append(output)

        # Step 2: Backward pass (only for final layer here)
        last_layer = self.layers[-1]
        for i, neuron in enumerate(last_layer.neurons):
            neuron.backward(
                target=target[i],  # e.g., target could be [1] or [0]
                inputs=layer_inputs[-2],  # input that came to this layer
                learning_rate=self.learning_rate
            )