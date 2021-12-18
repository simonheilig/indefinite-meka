function K = elm_kernel(X,Y,varargin)
    dSig = 1/(2*1e10);
    KxDiag = sum(X.*X,2);
    KyDiag = sum(Y.*Y,2);
    K = 2/pi*asin((1+X*Y')./sqrt((dSig+1+KxDiag)* (dSig+1+KyDiag)' ));
% --> lt https://www.sciencedirect.com/science/article/pii/S0925231211002591?via%3Dihub
    

