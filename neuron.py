import math
import random

class Neuron:
    def __init__(self, num_inputs, neuron_id=None, threshold=0.7):
        self.weights = [random.random() for _ in range(num_inputs)]  # list of weights
        self.bias = random.random()  # single bias value
        self.neuron_id = neuron_id or f"{id(self)}"  # fallback if not provided
        # self.threshold = threshold  # activation threshold
        self.threshold = sum(self.weights) / len(self.weights) + self.bias * 0.5

    def forward(self, inputs):
        print("Neuron ID:", self.neuron_id) # print neuron ID for debugging
        print("Inputs:", inputs)  # print inputs for debugging
        print("Weights:", self.weights) # print weights for debugging
        print("Bias:", self.bias) # print bias for debugging
        
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias # weighted sum
        print("Total before activation:", total) # print total for debugging
        
        output = self.activate(total) # apply activation function
        print("Output after activation:", output) # print output for debugging

        self.last_output = output # store last output for potential use in backward pass
        
        # Firing awareness
        self.fired = output > self.threshold  # check if the neuron fired
        print("Fired Status:", self.fired) # print firing status for debugging
        print("Threshold:", self.threshold) # print threshold for debugging
        
        return output if self.fired else 0 # return output if fired, else 0

    def activate(self, z):
        # Example: sigmoid activation
        return 1 / (1 + math.exp(-z))  # or use math.exp(z) if you import math
    
    def backward(self, target, inputs, learning_rate=0.1):
        # Calculate error = target - self.last_output
        error = target - self.last_output

        # Derivative of sigmoid for self.last_output
        derivative = self.last_output * (1 - self.last_output)

        # Gradient = error * derivative
        gradient = error * derivative

        # Update each weight
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * gradient * inputs[i]

        # Update bias
        self.bias += learning_rate * gradient

        return error  # return error just for tracking