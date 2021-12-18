function [best_param,best_C,losses] =tuneParams(kerneltype,kernelfun,params,boxconstraints,X,Y)
    %tuneParams computes the best kernel parameters and SVM-C parameter
    %with a 5-Fold cross-validation
    
    global myK
    losses = {};
    best = inf;
    best_param = 0;
    best_C = 0;
    for param_iter = 1:length(params)
        param = params{param_iter};
        C_iter = 1;
        K = kernelfun(X,X,param);
        if kerneltype == "tl1" % in case of the indefinite kernel, correct the matrix to be psd
            [v,e] = eig(K);
            shift = min(diag(e));
            if shift < -1e-6
               fprintf("non psd kernel %f\n",shift);
               K = v*e*v' + 2*abs(shift)*eye(size(X,1));
            end
        end
        
        myK = K;
        for C = boxconstraints
            fprintf("%s : starting with param iter %i and C iter %i\n",kerneltype,param_iter,C_iter);
            indices = [1:size(X,1)]';
            t = templateSVM('Standardize',false,'Solver','L1QP','KernelFunction','fKernel','IterationLimit',1e5,'BoxConstraint',C);
            [SVMModel_o] = fitcecoc(indices,Y,'Learners',t);
            CVMdl_o = crossval(SVMModel_o,'KFold',5);
            genError_o = kfoldLoss(CVMdl_o,'Mode','individual');
            losses{param_iter,C_iter} = mean(genError_o);
            if losses{param_iter,C_iter} < best
                best = losses{param_iter,C_iter};
                best_param = param;
                best_C = C;
            end
            C_iter = C_iter +1;
        end
    end
end