function W = randInitializeWeights(L_in, L_out)
%L_in incoming connections and L_out outgoing connections to the Theta between layers
%example: theta is between two layers the incomming layer is has 2 act units and the outgoing layer
%has 5 act units . Hence L_in = 2 and L_out = 5


%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%

%One effective strategy for choosing Eps_init is to base it on the number of units in the
% network. A good choice of Eps_init is Eps_init = sqrt(6)/sqrt(L_in+L_out) where 
% L_in and L_out are number the number of units in the layers adjacent to L.

Eps_init = 0.12;

W = rand(L_out, 1 + L_in) * 2 * Eps_init - Eps_init;


% =========================================================================

end
