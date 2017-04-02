function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

%here 'features' means number of features + 1 bias unit

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables (or not)
m = size(X, 1);

% Number of training fetures including the bias

n = size (X,2) +1;

% Number of hidden layer units excl the bias

h = hidden_layer_size;

% Number of output classifications

r = num_labels;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%Feedforward

#1. construct y-samples matrix (y_matrix) for sample outputs; each row is sample ech column is class so m x num_lables

y_matrix = eye(num_labels)(y,:); 

#2. Forward propagation

#a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a_1 = [ones(m,1) X];

#z2 equals the product of a1 and Θ1; theta in our case 25 x 401 so a_1 should be 401 x m
z_2 = a_1*Theta1';

#a2 is the result of passing z2 through g()
a_2 = sigmoid(z_2);

#Then add a column of bias units to a2 (as the first column). 
a_2 = [ones(size(a_2,1),1) a_2];

#z3 equals the product of a2 and Θ2
z_3 = a_2*Theta2';

%calculate activation values of OUTPUT LEVEL (level 3)  
a_3 = sigmoid(z_3);

#3. CF value
	
J = (-1/m).*sum(sum((y_matrix.*log(a_3) + (1.-y_matrix).*log(1-a_3))));



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Backpropagaton algorithm

% 1. Forw Prop from strt to end of NN: done (s above)

% 2. Start at the end of the NN. 
% d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (samples x outputs)
d_3 = a_3 - y_matrix;

% 3. calculate d2 for hidden layer l=2
d_2 = d_3*Theta2(:,2:end).*sigmoidGradient(z_2);

% 4. Calculate Delta matrices from input to the output of NN.
% Delta1 is the product of d2 and a1. The size is (hiddenlayer x samples) ⋅ (samples x features) --> (hiddenlayer x features)

Delta1 = d_2'*a_1;

% 5. Delta2 is the product of d3 and a2. 

Delta2 = d_3'*a_2;

%6. obtain (unregularized) gradients (big D matrixes)

Theta1_grad = (1/m)*Delta1;
Theta2_grad = (1/m)*Delta2;





%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% removing the j=0 weights as they should not be regularized
Theta1(:,1)=0;
Theta2(:,1)=0;

% regularizing the weights
Theta1_reg = (lambda/m)*Theta1;
Theta2_reg = (lambda/m)*Theta2;

% regularizing the gradients
Theta1_grad = (1/m)*Delta1 + Theta1_reg;
Theta2_grad = (1/m)*Delta2 + Theta2_reg;



#Regularization. No regularizing for the terms of the bias.
regTerm = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regTerm;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
