function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%calc hypothesis func
h_x = X*theta;

%remove the theta_0 from theta set (zero weight is bias)
theta = theta(2:end);

%CF implemention
J = (1/(2*m)).*sum((h_x-y).^2) + (lambda/(2*m))*sum(theta.^2);

%gradient for the bias weight w/o regularization
grad_0 = (1/m)*X(:,1)'*(h_x-y);

%regularization applied to the remaining weights
grad = (1/m)*(X(:,2:end)'*(h_x-y))+(lambda/m).*theta;

%putting the gradients together
grad = [grad_0; grad];

% =========================================================================

end
