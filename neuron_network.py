from layer import Layer

class NeuralNetwork:
    def __init__(self, layer_configs):
        self.layers = []
        for config in layer_configs:
            layer = Layer(
                num_neurons=config["num_of_neurons"],
                num_inputs_per_neuron=config["num_inputs_per_neuron"]
            )
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def back_propagate(self, expected, learning_rate):
        # Output layer delta
        last_layer = self.layers[-1]
        for i, neuron in enumerate(last_layer.neurons):
            error = expected[i] - neuron.output
            neuron.delta = error * neuron.derivative()

        # Hidden layers delta
        for l in reversed(range(len(self.layers) - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            for i, neuron in enumerate(layer.neurons):
                error = sum(next_neuron.weights[i] * next_neuron.delta for next_neuron in next_layer.neurons)
                neuron.delta = error * neuron.derivative()

        # Update weights and biases
        for layer in self.layers:
            for neuron in layer.neurons:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] += learning_rate * neuron.delta * neuron.inputs[j]
                neuron.bias += learning_rate * neuron.delta

    def train(self, training_data, epochs, learning_rate, patience=1000, min_delta=1e-6):
        best_error = float('inf')
        wait = 0

        for epoch in range(epochs):
            total_error = 0
            for inputs, expected in training_data:
                outputs = self.forward(inputs)
                self.back_propagate(expected, learning_rate)
                total_error += sum((e - o) ** 2 for e, o in zip(expected, outputs))

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {total_error:.4f}")

            # Early stopping condition
            if best_error - total_error > min_delta:
                best_error = total_error
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f"🛑 Early stopping at epoch {epoch} (no improvement for {patience} checks)")
                break

    def predict(self, inputs):
        return self.forward(inputs)