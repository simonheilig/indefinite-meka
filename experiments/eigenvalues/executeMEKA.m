function [] = executeMEKA(data, kernelfunc, kernelparam, mekaopts, dataset)
%executeMEKA inspect the psd / non psd behaivour of MEKA
%   input: 
%           data as n x d matrix (n samples,d features)
%           kernelfunc function pointer
%           kernelparam array for corresponding kernel params
%           mekaopts array of structs for meka params
%           allTargetRanks array of all distinct target ranks
%
%   saves to "tmp/"+dataset+"/"+dataset+kernelIter+".mat":
%           p_prev vector of positiv eigenvalues bevor MEKA
%           n_prev vector of negativ eigenvalues bevor MEKA
%           r_prev vector of neutral eigenvalues bevor MEKA
%           p_aft matrix of positiv eigenvalues after MEKA
%           n_aft matrix of negativ eigenvalues after MEKA
%           r_aft matrix of neutral eigenvalues after MEKA
%           ApproxErr is the approximation error
%           maxNE_{prev,aft} the eigenvalue with maximum magnitude of all
%           negative eigenvalues
    fprintf("Starting with MEKA experiment for data set %s\n",dataset);
    rng(100);
    numberOfKernelOpts = length(kernelparam);
    numberOfMEKAOptions = numel(mekaopts);
    for kernelIter = 1:numberOfKernelOpts
        
        %Build original kernel matrix
        Kernel = kernelfunc(data,data,kernelparam(kernelIter));

        %Calculate statistics of original matrix eigenspectrum
        E = eig(Kernel);
        p_prev = size(E(E>1e-6),1);
        r_prev = size(E((E>=-1e-6) & (E<=1e-6)),1);
        n_prev = size(E((E<-1e-6)),1);
        maxNE_prev = min(E);

        %create memory-addresses for parfor loop
        n_aft = [];
        p_aft = [];
        r_aft = [];
        maxNE_aft = [];
        ApproxErr = [];
        parfor MekaIter = 1:numberOfMEKAOptions
            fprintf("starting with MekaIter %i, KernelIter %i\n", MekaIter,kernelIter);
            try
                %Execute MEKA approximation for current parameters
                approximation = meka(sparse(data),mekaopts(MekaIter).targetrank,kernelfunc,kernelparam(kernelIter),mekaopts(MekaIter));
                [~,Q,L,~] = approximation.execute();
                %Build matrix for analyzation purpose
                K_app = Q*L*Q';
                if ~issymmetric(K_app)
                    K_app=0.5*(K_app+K_app');
                end

                %calculate statistics of MEKA matrix eigenspectrum
                t_o = cputime;
                E_app = eig(K_app);
                maxNE_aft(MekaIter) = min(E_app);
                t_o2 = cputime;
                fprintf("find smallest eigenvalue in %f seconds (eig decomp)\nDataset: %s; KernelIter: %i; MekaIter: %i\n",(t_o2-t_o),dataset,kernelIter,MekaIter);
                n_aft(MekaIter) = size(E_app(E_app<-1e-6),1);
                p_aft(MekaIter) = size(E_app(E_app>1e-6),1);
                r_aft(MekaIter) = size(E_app((E_app <= 1e-6) & (E_app >= -1e-6)),1);

                %Calculate approximation error to original matrix and
                ApproxErr(MekaIter) = norm(Kernel-K_app,'fro')/norm(Kernel,'fro');
                
            catch ex
                fprintf("An error ocurred in Meka Iter %i and Kernel Iter %i in dataset %s\n",MekaIter, kernelIter, dataset);
                fprintf("in line %i",ex.stack.line);
            end
        end
        %because outside of this method is a parfor used
        saveIteration("tmp/"+dataset+"/"+dataset+kernelIter+".mat", p_prev, n_prev, r_prev, p_aft,n_aft, r_aft, maxNE_prev, maxNE_aft, ApproxErr);
    end
end

function []=saveIteration(file, p_prev, n_prev, r_prev, p_aft, n_aft, r_aft, maxNE_prev, maxNE_aft, ApproxErr)
    save(file, 'p_prev', 'n_prev', 'r_prev', 'p_aft', 'n_aft', 'r_aft', 'maxNE_prev', 'maxNE_aft', 'ApproxErr');
end


