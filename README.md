# Assignment 5: Convolutional Neural Networks
Adapted by Mark Sherman <shermanm@emmanuel.edu> from MIT 6.S198 under Creative Commons
Emmanuel College - IDDS 2132 Practical Machine Learning - Spring 2021

This assignment is based on work by Yaakov Helman, Natalie Lao, and Hal Abelson

1.1: Experimenting with convolutional models
============================================

Start up Model Builder: the same demo you've used for assignment 2,[ ](https://courses.csail.mit.edu/6.s198/spring-2018/model-builder/src/model-builder/)

<https://courses.csail.mit.edu/6.s198/spring-2018/model-builder/src/model-builder/>

**Set the Dataset input to MNIST, and set the Model input to Convolutional.** You'll see that this constructs a fairly elaborate model. There are two pieces, each piece consisting of three layers---a convolutional layer, a max pool layer, and a ReLU activation layer---and the entire thing feeding through a flatten layer to a fully connected layer with 10 hidden units. **Let the network train for a bit.** You should see that it quickly gets over 90% accuracy. (Move the cursor over the accuracy plot line to display the accuracy.)

**Now switch the Dataset to CIFAR 10 and train that.** The model still works, but doesn't do nearly as well: it should reach 50% accuracy after training with 40,000 examples. (Hal Abelson at MIT ran it for a million examples and it barely reached 80% accuracy.) 

**While the model is training, take a look at the hyperparameters and the input and output shapes of the layers.** 
- The first (lowest) convolutional layer has input shape `[32,32,3]`: the CIFAR images are 32⨉32 pixels and there are three RGB colors per pixel.
- The output shape is `[32,32,16]`: for each input pixel the convolutional layer generates 16 output values, as specified by the output hyperparameter for the layer.
- The first max pool layer takes that `[32,32,16]` input shape (unchanged by ReLU) and reduces it to `[16,16,16]` in accordance with the `field = 2` hyperparameter. We'll examine the meanings of these and other hyperparameters below.

1.2: How convolutional neural networks works
============================================

See Lecture recording from March 1, and the [R3 assignment in ECLearn](https://eclearn.emmanuel.edu/courses/3147959/assignments/31612152). If you haven't read and watched this material yet, now is the time to go do that before you continue. 

1.3: Experimenting with hyperparameters
=======================================

Now that you have an idea of what the layers and hyperparameters mean, **spend a few minutes modifying the network in Model Builder to see how these affect performance on CIFAR 10.**

For example, trying removing one of the Conv, ReLU, Max pool triples and see how that affects performance. Or try changing the number of outputs for the conv layers, or the field size or stride for the conv or max pool layers. Try at least three different variants that you train for 20,000 or 30,000 examples each and take notes on whether they make a difference (better or worse). Note that changing hyperparameters may require changing other hyperparameters to make the layer sizes consistent.

1.4: Visualizing convolutional neural networks
==============================================

To gain some intuition about how convolutional neural networks work, visit the Web page at [http://scs.ryerson.ca/~aharley/vis/conv/flat.html](https://www.cs.cmu.edu/~aharley/vis/conv/flat.html). It looks like this:

![display of webpage](img/harley-conv.jpg)

This is a visualization demo by Adam Harley, described in the paper "An Interactive Node-Link Visualization of Convolutional Neural Networks" (<http://www.cs.cmu.edu/~aharley/vis/harley_vis_isvc15.pdf>). It shows a network with two convolutional layers, two fully connected layers, and two max pool layers (called "downsampling" in the demo). Unlike Model Builder, the input layer is at the bottom and the output label layer at the top. The network was trained on MNIST; the labels are the digits 0 through 9.

You can use the demo's pencil and eraser tools to create inputs to the network and see how these are transformed by the filters at each layer of the network. If you click on any square at a layer, you can see the inputs and output of the neuron at that square. Notice that the convolutional layer applies tanh as an activation function to the result of the weight; the demo doesn't show this as a separate activation layer.

## WRITEUP REQUIRED - Problem 1

Spend some time playing with this demo. Draw an input, modify it, and observe how the results at each layer change as you change the drawing. Create some inputs that look vaguely like digits, but that confuse the network, i.e., where two or more of the labels register. Write up interesting observations about what you see combined with illustrative screenshots.

> Problem 1 response here

1.6: Building CNNs 
==================

Last week you explored fully connected networks with model builder. This week's homework covers the same experimentation with CNNs.

## WRITEUP REQUIRED - Problem 2

Similar to the work you did with Model Builder on Monday, investigate several choices of architectures and hyperparameters for classifying images from MNIST, Fashion MNIST, and CIFAR_10. For the following 4 questions, describe what architectures you implemented and take screenshots of the results you got for each dataset (MNIST, Fashion MNIST, and CIFAR).

Remember to add an image, put it in this folder and use the markdown code `![](imageName.ext)`

1\. Try changing the learning rate, and batch size. Make sure that these numbers are reasonable to start (i.e. won't take too long to run on your computer).

2\. Try changing the optimizer. Some optimizers have additional parameters- see what impact they appear to have, then try to find some documentation as to what they're doing to the model. Do your best to explain how those parameters are impacting your observations.

3\. Do you have a hypothesis for why CIFAR-10 is so much harder to train on than Fashion MNIST and MNIST (i.e. it's more difficult to achieve a 90%+ accuracy) while Fashion MNIST has similar training times to MNIST (even though Fashion MNIST is more complex than MNIST)?

4\. How does adding more convolutional layers relate to accuracy and training speed? Is there a point at which adding more layers plateaus or even decreases the maximum accuracy you are able to achieve with that model?

5\. Challenge: Are you able to find an architecture/combination of techniques that can get you to 60% accuracy on CIFAR-10 within 1 minute of training? 5 minutes? 10 minutes?

2: Submission 
==============

Commit images and this file to git, and push. It is considered submitted when you push changes. I will evaluate the project after teh due date, so you can push multiple times prior to then. You don't have to push each commit separately. One push at the end is fine. 

![Creative Commons License](https://lh5.googleusercontent.com/x-wojjJqkwgbIsS8V_DZQTKVM778hK75oD6bRsEyru_NZ5OmFJW_NmaEXk9nhpJvFLDmRC83VzGdW-kPim4B7aef3C9eCKhsH-2mgUbD99DoP0K-gAzrPY8FdMkNtN_KI83Q9CKz)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
