# XOR-Gate
This neural network learns how to correctly map two binary inputs to a binary output according to the logic of a xor-gate.
<br><br>
The following table displays which inputs A and B map to which output X.

| A | B | X |
| --- | --- | --- |
| 1 | 1 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 0 | 0 | 0 |

<br>

# Why do this project? / What to learn from it?
In the [previous project](https://github.com/LinusSee/learning-ai/tree/master/first-steps/simple_perceptron) I implemented a simple perceptron using it to classify (in a cartesian coordinate system) whether a given point is above a straight line. With that project I understood the (simplest) basics of how to implement a neural network.
<br>
However this network was a crude one, having only a single neuron and a basic learning algorithm. Hence the goal of the xor-project is to learn how to implement a larger network with more sophisticated algorithms and to get familiar with the concept of gradient descent and backpropagation.
<br>
# How the network is implemented
The network is made for having two input neurons, two hidden neurons and a single output neuron. In addition to their "normal" weights, each of the hidden neurons and the output neuron has it's own weighted bias. I found that with a single weighted bias for each lane instead of each neuron, the network was not able to learn the xor-logic, more about the reason will follow later.
<br>
