function [vNegmax] = largestNegEigApprox(Q,L,n)
    vNegmax = eigs(@(x)Q*(L*(Q'*x)),n,1,'smallestreal');
end

