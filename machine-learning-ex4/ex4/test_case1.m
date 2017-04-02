%% Initialization
clear ; close all; clc

%test case script 

input_layer_size = 2;              % input layer
hidden_layer_size = 2;              % hidden layer
num_labels = 4;              % number of labels
nn_params = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = 4;

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%*****************************Start of the Algorithm**********

%Feedforward

#1. construct y-matrix for sample outputs; each row is sample ech column is class so m x num_lables

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


Theta1(:,1)=0;
Theta2(:,1)=0;

Theta1_reg = (lambda/m)*Theta1;
Theta2_reg = (lambda/m)*Theta2;


Theta1_grad = (1/m)*Delta1 + Theta1_reg;
Theta2_grad = (1/m)*Delta2 + Theta2_reg;

grad = [Theta1_grad(:) ; Theta2_grad(:)]

