function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%define thetas
theta_0=theta(1,1);
theta_1=theta(2,1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

  % calc derivative over m training samples
  %def sums
  resSum_0 =0;
  resSum_1 =0;
  
  for i=1:m

    h_x=[theta_0,theta_1]*(X(i,:))';
    resSum_0=resSum_0+(h_x-y(i))*X(i,1);
    resSum_1=resSum_1+(h_x-y(i))*X(i,2);

  end

%update thetas
theta_0 = theta_0 - alpha*(1/m)*resSum_0;
theta_1 = theta_1 - alpha*(1/m)*resSum_1;

%combine thetas to vector
theta = [theta_0;theta_1];
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end


end
