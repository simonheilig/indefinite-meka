function [K] = rbf_kernel(X,Y,gamma)
%RBF kernel
%   returns the Gram matrix with K(x,x') = exp(- (norm(x-x')^2 / 2*sigma^2)

if ~isscalar(gamma)
    error('need a scalar parameter');
end
K = exp(- (pdist2(X,Y,'squaredeuclidean')) *gamma);
if ~issymmetric(K) && (size(X,1)==size(Y,1) &&(size(X,2)==size(Y,2)))
    K = 0.5 *(K+K');
end
end

