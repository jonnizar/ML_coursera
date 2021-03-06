function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%get number of examples

m = size(X,1);

% SORRY! VeRY UGLY CODE BUT NO TIME! 

%find correct centroid for particular example 

for i = 1:m

    %starting value for minimal length cluster k
    kMin = 1;

    %initial minimal distance value
    dMinIni = sum((X(1,:) - centroids(1,:)).^2)*100;
     
  

    %look for closest centroid 
    for k=1:K
        
        
        if k==1
        
             dMin = dMinIni;
             
        end
        
       %calc Euclidean Distance Squared
        d = sum((X(i,:) - centroids(k,:)).^2);
        
        if d < dMin
        
            dMin = d;
            kMin = k;
            
        end
    
    
    end
    


%assign example to the closest centroid 
    idx(i) = kMin;

end





% =============================================================

end

