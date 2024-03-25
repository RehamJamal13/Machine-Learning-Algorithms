Neural Network Architecture 

Neural Network are mathematical models that use learning algorithms inspired by the brain to store information. Since neural networks are used in machines, they are collectively called an ‘artificial neural network.’
They are used to process complex data and non linear separable data and make predictions by learning from examples.

![Colored_neural_network svg](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/33518920-ed38-4cea-acab-7918068f6c32)
  This repository provides a simple implementation of a neural network and the backpropagation algorithm. Neural networks are a class of machine learning models inspired by the biological structure of the brain. They consist of interconnected nodes, called neurons, organized in layers. Each neuron receives input, performs some computation, and produces an output.

Neural Network Implementation
The neural network implemented in this repository consists of an input layer, one hidden layers, and an output layer. Each layer is composed of neurons, and connections between neurons are represented by weights. The network can be customized by specifying the number of neurons in each layer and the activation function used.

## Forward Pass

The forward pass involves computing the output of the neural network given an input. It consists of the following steps:

1. **Input Layer**: The input data is fed into the neural network.
2. **Hidden Layers**: The input is passed through one or more hidden layers, where each neuron computes a weighted sum of its inputs and applies the activation function (sigmoid in this case).
3. **Output Layer**: The output of the last hidden layer is passed through the output layer, which produces the final output of the neural network.

![Feed-forward-neural-network-with-sigmoid-activation-function-X-i-i-1P-input](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/3169222c-b4ed-45a3-8fcd-9715152c51cc)

Backpropagation
Backpropagation is an algorithm used to train neural networks. It involves propagating errors backward through the network, adjusting the weights of connections between neurons to minimize the difference between the actual output and the desired output.

## Implementation Overview

The provided code consists of the following components:

- **Data Generation**: Synthetic data is generated for binary classification.
- **Activation Function**: The sigmoid function is used as the activation function.
- **Loss Function**: Cross-entropy loss is utilized as the loss function.
- **Forward Pass**: Computes forward pass through the neural network.
- **Backward Pass**: Computes backward pass for updating parameters.
- **Parameter Initialization**: Random initialization of weights and biases.
- **Accuracy Calculation**: Computes accuracy of predictions.
- **Training Loop**: Iteratively trains the neural network using backpropagation.
- **Plot Decision Boundary**: Visualizes decision boundary of the trained model.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the parameters of the neural network in the direction of the negative gradient. This process continues until convergence or a predefined number of iterations.

Activation Function: Sigmoid
The sigmoid function is a commonly used activation function in neural networks. It squashes the input values between 0 and 1, making it useful for binary classification problems. The sigmoid function is defined as:

 ![1_Xu7B5y9gp0iL5ooBj7LtWw](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/3a256fbe-8f10-44aa-9475-39d635c0d4a1)


Loss Function: Cross-Entropy
The cross-entropy loss function is often used in classification problems. It measures the difference between the predicted probability distribution and the true distribution of the labels. For binary classification, the cross-entropy loss function is defined as:


​
  is the predicted probability of the positive class.



Neural networks can handle nonlinear algorithms, making them suitable for a wide range of tasks in machine learning.
Components of a Neural Network :Neurons (Nodes),Layers,Weights and Biases,Activation Functions.
Working Mechanism: Forward and Backward Propagation.
Forward Propagation = Z=WX+b  A=G(Z).
The gradients are propagated backward through the network TO CALCULATE Backward Propagation.
Loss Function: Measures the discrepancy between predicted and actual outputs during training.we use  cross entropy function in this file.
Accuracy: Measures the percentage of correctly classified instances in the test dataset.







Principal Component Analysis (PCA) is a dimensionality reduction technique that aims to find the orthogonal axes (principal components) that capture the maximum variance in the data. It achieves this by decomposing the covariance matrix of the standardized data into its eigenvectors and eigenvalues. Mathematically, PCA involves finding the eigenvectors V and eigenvalues λ of the covariance matrix C, The principal components are then formed by selecting the top k eigenvectors corresponding to the largest eigenvalues. Finally, the original data is transformed into the new coordinate system spanned by these principal components by projecting it onto them using the equation Y=XV, where Y is the transformed data matrix and V is the matrix containing the selected eigenvectors.




