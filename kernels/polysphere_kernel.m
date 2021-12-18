function [K] = polysphere_kernel(X,Y,varargin)
    K = (1-((pdist2(X,Y,'squaredeuclidean')./(varargin{1}(1)^2)))).^varargin{1}(2);
end

