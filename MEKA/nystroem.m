
function [U] = nystroem(data, m, k,kernelfunction,kernelopts)
% standard Nystrom provided by Si et al.
% to approximate the eigenvectors U
% m number of landmarks
% k is target rank
rng(100);%only for eigenvalue experiments
[n,dim] = size(data);
dex = randperm(n);
landmarks = data(dex(1:m),:);
W = kernelfunction(landmarks,landmarks,kernelopts);
E = kernelfunction(data,landmarks,kernelopts);
[Ue, Va, Ve] = svd(full(W));
vak = diag(Va(1:k,1:k));
inVa = sparse(diag(vak.^(-0.5)));
U = E * Ue(:,1:k) * inVa;
if issparse(U)
    fprintf('need full U with target rank %i\n',k);
    U = full(U);
end
