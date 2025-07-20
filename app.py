from neuron_network import NeuralNetwork

# 2 input neurons → 3 hidden neurons → 1 output neuron
layer_config = [
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

nn = NeuralNetwork(layer_config)

# ----- XOR Training -----
if __name__ == "__main__":
    xor_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]

    nn.train(
        xor_data,
        epochs=100000,
        learning_rate=0.1,
        patience=3000,       # Wait max 3000 epochs for improvement
        min_delta=0.00001    # Accept only meaningful improvements
    )


    print("\n--- Testing XOR ---")
    for inputs, expected in xor_data:
        prediction = nn.predict(inputs)
        print(f"Input: {inputs} => Predicted: {prediction[0]:.4f}, Expected: {expected[0]}")
