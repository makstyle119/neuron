from neuron import Neuron
from layer import Layer
from neuron_network import NeuralNetwork

# # Example usage of the Neuron class
# neuron = Neuron(2)
# print("Weights:", neuron.weights)
# print("Bias:", neuron.bias)
# print(neuron.forward([0, 0]))  # expect ~0.3 or less
# print(neuron.forward([1, 0]))  # expect ~0.5
# print(neuron.forward([1, 1]))  # expect >0.7

# # Test it
# layer1 = Layer(2, 2, 1)  # 2 neurons, each with 2 inputs, id=1
# layer2 = Layer(2, 2, 2)  # 2 neurons, each with 2 inputs, id=2

# # print("Neuron Weights and Biases:")
# # for i, neuron in enumerate(layer.neurons):
# #     print(f"Neuron {i+1}: Weights = {neuron.weights}, Bias = {neuron.bias}")

# inputs = [1, 0]
# outputs1 = layer1.forward(inputs)
# outputs2 = layer2.forward(outputs1)

# print("\nInput:", inputs)
# print("Outputs1:", outputs1)
# print("Outputs2:", outputs2)

# # 2 input neurons → 3 hidden neurons → 1 output neuron
# nn = NeuralNetwork([2, 3, 1])

# inputs = [1, 0]
# output = nn.forward(inputs)

# print("Final Output:", output)

# # Step 1: Create 2-layer network
# layer1 = Layer(3, 2, layer_id=1)
# layer2 = Layer(1, 3, layer_id=2)  # Output layer with 1 neuron, 3 inputs (from 3 neurons in layer1)

# # Step 2: Forward pass
# inputs = [1, 0]
# outputs1 = layer1.forward(inputs)
# outputs2 = layer2.forward(outputs1)

# # Step 3: Backpropagation on output layer
# target_output = [1]  # What we want
# learning_rate = 0.1

# for i, neuron in enumerate(layer2.neurons):
#     neuron.backward(
#         target=target_output[i],
#         # output=outputs2[i],
#         inputs=outputs1,
#         learning_rate=learning_rate
#     )

# # (Optional) Print updated weights
# print("\nAfter Backpropagation:")
# for i, neuron in enumerate(layer2.neurons):
#     print(f"Neuron {i+1} Weights:", neuron.weights)
#     print(f"Neuron {i+1} Bias:", neuron.bias)

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
# nn.train(inputs, target)
for _ in range(10000):
    nn.train(inputs, target)

# Predict something
output = nn.forward(inputs)
print("Prediction:", output)

