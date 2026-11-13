"""
ANN PRACTICAL ASSIGNMENT – ROLL NO. 8054
Assignment: McCulloch-Pitts Neuron for Logic Functions
Subject: Artificial Neural Networks (ANN)
Language: Python
Library: NumPy

1. AIM
To design and implement McCulloch-Pitts neurons for AND, OR and NOT
logic operations using suitable weights and threshold values, verify
all input combinations, and compare an M-P neuron with a perceptron.

2. THEORY
The McCulloch-Pitts neuron is an early mathematical model of an
artificial neuron, introduced by Warren McCulloch and Walter Pitts in
1943. It receives binary inputs (0 or 1), multiplies each input by its
weight, adds the weighted inputs, and compares the sum with a threshold.

Mathematical model:
    Net = sum(w_i * x_i)
    Y = 1, if Net >= threshold
    Y = 0, if Net < threshold

Here x_i = input, w_i = weight, Net = weighted sum,
threshold = threshold value, and Y = output.

3. DESIGN AND CALCULATIONS

AND GATE
Weights: w1 = 1, w2 = 1; threshold = 2
Equation: Net = x1 + x2
Inputs and calculations:
    (0,0): 0+0 = 0 -> Output 0
    (0,1): 0+1 = 1 -> Output 0
    (1,0): 1+0 = 1 -> Output 0
    (1,1): 1+1 = 2 -> Output 1
Explanation: Output is 1 only when both inputs are 1.


OR GATE
Weights: w1 = 1, w2 = 1; threshold = 1
Equation: Net = x1 + x2
Inputs and calculations:
    (0,0): 0+0 = 0 -> Output 0
    (0,1): 0+1 = 1 -> Output 1
    (1,0): 1+0 = 1 -> Output 1
    (1,1): 1+1 = 2 -> Output 1
Explanation: Output is 1 when at least one input is 1.

NOT GATE
Weight: w1 = -1; threshold = 0
Equation: Net = -x1
Inputs and calculations:
    (0): -(0) = 0  -> Output 1
    (1): -(1) = -1 -> Output 0
Explanation: Output is the opposite of the input.

4. COMPARISON OF LOGIC GATES
    X1  X2  AND  OR
     0   0   0   0
     0   1   0   1
     1   0   0   1
     1   1   1   1

NOT truth table:
    X1  NOT
     0   1
     1   0

5. DIFFERENCE BETWEEN M-P NEURON AND TRAINABLE PERCEPTRON
McCulloch-Pitts neuron:
- Weights are fixed manually.
- Does not learn from examples.
- Uses a threshold-based output.
- Mainly used for basic logic operations.
- No learning algorithm is required.

Trainable perceptron:
- Weights are adjusted during training.
- Learns from training examples.
- Uses an activation function.
- Used for classification tasks.
- Uses a learning rule to update weights.

Main difference: An M-P neuron has predefined weights and threshold,
whereas a perceptron learns suitable weights from training data.

6. PYTHON IMPLEMENTATION
The program below uses Python and NumPy to implement AND, OR and NOT.

7. EXECUTION RESULTS
AND: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1
OR:  (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
NOT: 0->1, 1->0

8. OBSERVATIONS
1. AND gives 1 only when both inputs are 1.
2. OR gives 1 when at least one input is 1.
3. NOT gives the complement of its input.
4. The selected weights and thresholds implement all three logic gates.
5. The M-P neuron does not automatically learn its weights.

9. RESULT
M-P neurons for AND, OR and NOT were designed and implemented using
Python and NumPy. The outputs were verified against the truth tables.

10. CONCLUSION
This practical demonstrated how weights and threshold values control
the output of an M-P neuron. AND, OR and NOT gates were implemented,
and the fixed-parameter M-P neuron was compared with a trainable
perceptron.
"""


import numpy as np

def mp_neuron(inputs, weights, threshold):
    net = np.dot(inputs, weights)
    return 1 if net >= threshold else 0

# AND Gate
and_inputs = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
])
and_weights = np.array([1, 1])
and_threshold = 2
# OR Gate
or_weights = np.array([1, 1])
or_threshold = 1
# NOT Gate
not_inputs = np.array([[0], [1]])
not_weights = np.array([-1])
not_threshold = 0

print("AND GATE")
for x in and_inputs:
    print(x, "Output:",
          mp_neuron(x, and_weights, and_threshold))
print("\nOR GATE")
for x in and_inputs:
    print(x, "Output:",
          mp_neuron(x, or_weights, or_threshold))
print("\nNOT GATE")
for x in not_inputs:
    print(x, "Output:",
          mp_neuron(x, not_weights, not_threshold))