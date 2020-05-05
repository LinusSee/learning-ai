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

## How to run it
I wanted to be able to run and try out the project whenever I want and as simple as possible. Therefore, and because of the networks simplicity, I implemented it in Javascript, so you can just open the index.html file which will display a UI for the network.

# Why do this project? / What to learn from it?
In the [previous project](https://github.com/LinusSee/learning-ai/tree/master/first-steps/simple_perceptron) I implemented a simple perceptron using it to classify (in a cartesian coordinate system) whether a given point is above a straight line. With that project I understood the (simplest) basics of how to implement a neural network.
<br>
However this network was a crude one, having only a single neuron and a basic learning algorithm. Hence the goal of the xor-project is to learn how to implement a larger network with more sophisticated algorithms and to get familiar with the concept of gradient descent and backpropagation.
<br>
I chose implementing a xor-gate, since it is a nonlinear problem and therefore requires more than a single neuron while not needing too large a network or too complex a problem either.
<br>
# How the network is implemented
## The basic structure
The network is made for having two input neurons, two hidden neurons and a single output neuron. In addition to their "normal" weights, each of the hidden neurons and the output neuron has it's own weighted bias. I found that with a single weighted bias for each lane instead of each neuron, the network was not able to learn the xor-logic, more about the reason will follow later. An image of the networks structure can be seen below.

![alt text](https://github.com/LinusSee/learning-ai/blob/master/first-steps/xor_gate/assets/images/network.png "PNG of the network")

<br>
<br>
The choice of two input neurons i1 and i2 is rather obvious, since basic logic gates usually have two input signals (except of course negation).
<br>
As mentioned earlier, a single neuron can only solve linear problems. Since a xor-gate is nonlinear using only a single neuron to process the two inputs was not an option. Because a xor-gate can also be implemented by (OR)AND(NAND) and all three of those gates (OR, AND + NAND) are linear problems, I decided to use two hidden layer neurons h1 and h2 an a single output neuron o2 to model this relationship.
<br>
One hidden neuron is supposed to model an OR-gate, the other a NAND-gate and the output neuron the AND-gate connecting them both. (Note: I have no idea if the network actually does it this way, but that was the thinking when designing the network)
<br>
This was also the reason why I switched from using a bias for each layer to a bias for each non-input neuron. With a single bias for both hidden neuron, I found no possible way to model both an OR-gate and a NAND-gate at the same time. Therefore the network didn't work at first, until I switched to seperate biases.
<br>
Because I chose to implement the network this way and focused on gradient descent and backpropagation, the network is pretty hardcoded and only works for the 2-2-1 structure.
<br>

## Backpropagation and gradient descent
Having understood the concept of backpropagation and gradient descent, I had quite some trouble wrapping my head around the math behind it and how to actually apply it in a neural network. Most examples regarding backpropagation I found online either only explain the concept or give a complex example that doesn't really help when still trying to grasp the basics.
<br>
At some point I found [a post](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) that actually goes through the process of backpropagation by an example.
<br>
All the math I use in this project is pretty much like explained in the blogpost, I just adjusted it for my network's structure and included the biases in the backpropagation process. Therefore I won't go into the math here, if you are interested, feel free to check out the blogpost.
<br>

## General information
The network has an adjustable learning rate and uses the sigmoid function as the activation function. It has two training methods, one for training with only a single data entry and the other for training with an entire batch. They literally do the same thing otherwise, since the train method just invokes the trainBatch method and passes only a single data entry as the batch.
<br>
There are a few methods that probably wouldn't be there in an optimal neural network, like one to perform matrix multiplication. I just chose not to use a framework for that in this example, since I wanted to keep it as simple and free of extraneous influence as possible. Also the meanSquaredError function is useless in my example and just for checking the error when debugging manually, because I hardcoded the derivatives.
