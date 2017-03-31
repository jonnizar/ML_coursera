function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



J = (1/(2*m)).*(X*theta-y)'*(X*theta-y);


%{
resSum=0;


for i=1:m

  h_x=theta'*(X(i,:))';
  resSum=resSum+power((h_x-y(i)),2);

end


J=(1/(2*m))*resSum; %compute cost
%}

% =========================================================================

end
