# General purpose network
The goal of this project is to implement a neural network that is not hardcoded to a certain amount of neurons per layer. It will be configurable on creation and math stuff like matrix multiplication will be done by libraries, to 
keep the code clean from all logic that doesn't belong to the network.
<br>
To test the network I will implement a UI, where a user can create a neural network, give it its training inputs and view the results. The UI will probably fairly simple and therefore only allow testing of logic gates.
<br>
When finished, the network should be able to feed forward and train no matter how many layers and neurons have been chosen.

# How to run it
Just like the previous projects I wanted it to be simple to run, so if you intend to just try out the example, simply go ahead an run the index.html in your favourite browser (no guarantees for IE ;) and it should run.

# Why do this project? / What to learn from it?
In my [previous project](https://github.com/LinusSee/learning-ai/tree/master/first-steps/xor_gate) I implemented a network learning an xor gate using gradient descent. However I literally did the math on paper for this specific network structure and then implemented it, so it was hardcode to the chosen network structure.
<br>
For this project I tried to figure out on my own how to devise a simple algorithm that fits every network structure, so it doesn't have to be implemented anew every time. I wanted to do it on my own, because I found few resources that were helpful to me and figuring it out alone helped me grasp the way it works much more than just looking it up.
<br>
<br>
While it took me a few days, I managed to figure out the algorithm and, in an attempt to provide a more clearer explanation than the ones I found, I wrote my thoughts down in this [document](https://github.com/LinusSee/learning-ai/blob/master/first-steps/general_purpose_network/assets/latex/example_network/backpropagation/backprop_through_matrix_multiplication.pdf).
<br>
Hopefully this document will be helpful to others who, as me, had problems finding an article that gives a good introduction for real beginners (and not add to the pile :D).
