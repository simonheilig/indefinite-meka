
function Y = blkdiag_meka(B)
%helper function provided by Si et al.
c = numel(B);
p2 = zeros(c+1,1);
m2 = zeros(1, c+1);
for k=1:c    
	x = B{k};    
	[p2(k+1),m2(k+1)] = size(x); %Precompute matrix sizes    
end%Precompute cumulative matrix size
p1 = cumsum(p2);
m1 = cumsum(m2);
Y = sparse(p1(end),m1(end)); %Preallocate for full doubles only    
for k=1:c        
	Y(p1(k)+1:p1(k+1),m1(k)+1:m1(k+1)) = B{k};    
end

