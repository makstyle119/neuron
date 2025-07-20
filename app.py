from neuron_network import NeuralNetwork

# 2 input neurons → 3 hidden neurons → 1 output neuron
neuron_layer = [
    { 
        "num_of_neurons": 2, 
        "num_inputs_per_neuron": 2
    }, 
    { 
        "num_of_neurons": 3, 
        "num_inputs_per_neuron": 2
    }, 
    { 
        "num_of_neurons": 1, 
        "num_inputs_per_neuron": 3
    }
]
nn = NeuralNetwork(neuron_layer, learning_rate=0.1)
inputs = [1, 0]
target = 1  # your desired output

# Train the network
for _ in range(10000):
    nn.train(inputs, target)

# Predict something
output = nn.forward(inputs)
print("Prediction:", output)

