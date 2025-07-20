from layer import Layer

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        for i, layer_obj in enumerate(layer_sizes):
            layer = Layer(
                num_neurons=layer_obj["num_of_neurons"],
                num_inputs_per_neuron=layer_obj["num_inputs_per_neuron"],
                layer_id=i+1 
            )
            self.layers.append(layer)
    
    def forward(self, inputs):
        val = inputs
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def train(self, inputs, target):
        val = inputs
        for layer in self.layers:
            layer.forward(val)
            val = layer.train(target, val)
