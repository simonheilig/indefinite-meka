function [K] = tl1_kernel(X,Y,varargin)
    K = max(varargin{1} - pdist2(X,Y,'cityblock'),0);
end

