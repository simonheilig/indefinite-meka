% this script evaluates the generalization ability of an approximated kernel
% SVM problem
% the hyperparameter are tuned with a 5Fold-CV and the meka parameters are
% selected such that a minimal approximation error is obtained with minimal
% memory consumption

datasets=["spambase","synth2","cpusmall","gesture","synth1","pendigit"];

execute_svm(datasets,"rbf",@rbf_kernel,{0.01,0.1,1,10,15},[1e-3,1,10,100,1000],[16,32,64,128,256],[3,5,10]);

execute_svm(datasets,"elm",@elm_kernel,{1},[1e-3,1,10,100,1000],[16,32,64,128,256],[3,5,10]);

execute_svm(datasets,"poly",@polysphere_kernel,{[2 2],[2 4],[2 6],[2,8],[2,10],[3 2],[3 4],[3 6],[3,8],[3,10]},[1e-3,1,10,100,1000],[16,32,64,128,256],[3,5,10]);

execute_svm(datasets,"tl1",@TL1_kernel,{},[1e-3,1,10,100,1000],[16,32,64,128,256],[3,5,10]);

function [] = execute_svm(datasets,kerneltype,kernelfun,kernelparams,Cs,ranks,clusters)
    global myK
    tic
    for dataset=datasets
        if dataset == "spambase"
            D = load("spam.mat");
            Y = D.spambase(:,58);
            X = D.spambase(:,1:57);
        elseif dataset == "optdigit"
            D = load("optDigitTra.mat");
            X = D.optdigitstra(:,1:62);
            Y = D.optdigitstra(:,63);
        elseif dataset=="synth2"
            D = load("synth2.mat");
            Y = D.syntheticblobs(:,3);
            X = D.syntheticblobs(:,1:2);
        elseif dataset=="synth1"
            D = load("synth1.mat");
            Y = D.syntheticclassification(:,16);
            X = D.syntheticclassification(:,1:15);
        elseif dataset == "cpusmall"
            D = load("cpusmall.mat");
            X = D.X;
            Y = D.Y;
        elseif dataset == "pendigit"
            D = load("pendigits.mat");
            X = D.X;
            Y = D.Y;
        elseif dataset == "gesture"
            D = load("gestures.mat");
            Y = D.gestures(:,33);
            X = D.gestures(:,1:32);
        end


        if kerneltype == "rbf"
            X = bsxfun(@rdivide,bsxfun(@minus,X,min(X)),max(X)-min(X));
        elseif kerneltype == "elm"
            X = bsxfun(@rdivide,bsxfun(@minus,X,mean(X)),std(X));
            X = myl2norm(X);
        elseif kerneltype == "poly"
            X = myl2norm(X);
        elseif kerneltype == "tl1"
            X = bsxfun(@rdivide,bsxfun(@minus,X,min(X)),max(X)-min(X));
            d = size(X,2);
            kernelparams = {0.7*d};
        end
       
        CVO = cvpartition(Y,'KFold',10);
        
        accuracies_o = zeros(CVO.NumTestSets,1);
        accuracies_pp = zeros(CVO.NumTestSets,1);
        accuracies_c = zeros(CVO.NumTestSets,1);
        train_accuracies_o = zeros(CVO.NumTestSets,1);
        train_accuracies_pp = zeros(CVO.NumTestSets,1);
        train_accuracies_c = zeros(CVO.NumTestSets,1);
        shifts = zeros(CVO.NumTestSets,1);
        Qs = {};
        Ls = {};
        paramss = {};
        Css =  zeros(CVO.NumTestSets,1);
        rank_mekas =  zeros(CVO.NumTestSets,1);
        nocs =  zeros(CVO.NumTestSets,1);
        relapprox_pps = zeros(CVO.NumTestSets,1);
        relapprox_cs = zeros(CVO.NumTestSets,1);
        meka_approx_tunings = {};
        
        for cv_iter = 1:CVO.NumTestSets
            fprintf("begin cross validation %i %s\n",cv_iter,dataset);
            train_idx = CVO.training(cv_iter);
            test_idx = CVO.test(cv_iter);
            Y_test = Y(test_idx);
            Y_train = Y(train_idx);
            X_train = X(train_idx,:);

            %5fold-cv for hyperparameter tuning
            [params,C,~] = tuneParams(kerneltype,kernelfun,kernelparams,Cs,X_train,Y_train);
            %subsequently, tune meka parameter k,c
            [Q,L,shift,rank_meka,noc,meka_approx_tuning] = tuneMEKA(X,kernelfun,params,ranks,clusters);
            
            %build up all kernel matrices for training and testing a SVM
            %classifier
            K = kernelfun(X,X,params);

            Kpp = full(Q*L*Q');
            Kpp = 0.5*(Kpp+Kpp');
            
            Kc = Kpp + 2*shift*eye(size(Kpp,1));
            
            K_test = K(test_idx,test_idx);
            K_traintest = K(train_idx,test_idx);
            K = K(train_idx,train_idx);

            Kpp_test = Kpp(test_idx,test_idx);
            Kpp_traintest = Kpp(train_idx,test_idx);
            Kpp = Kpp(train_idx,train_idx);
            
            Kc_test = Kc(test_idx,test_idx);
            Kc_traintest = Kc(train_idx,test_idx);
            Kc = Kc(train_idx,train_idx);
           
            relapprox_pp = norm(K-Kpp,'fro')/norm(K,'fro');
            relapprox_c = norm(K-Kc,'fro')/norm(K,'fro');

            
            %train accuracy original
            myK = K;
            indices = [1:length(train_idx(train_idx==1))]';
            indices_test = ([1:length(test_idx(test_idx==1))]+length(train_idx(train_idx==1)))';

            t = templateSVM('Standardize',false,'Solver','L1QP','KernelFunction','fKernel','IterationLimit',1e5,'BoxConstraint',C);
            SVMModel_o = fitcecoc(indices,Y_train,'Learners',t);
            train_label_o = predict(compact(SVMModel_o),indices);
            train_accuracy_o = sum((train_label_o == Y_train))/length(Y_train)*100;

            %test accuracy original
            Ktesting = [K,K_traintest;K_traintest',K_test];
            myK = Ktesting;
            test_label_o = predict(compact(SVMModel_o),indices_test);
            test_accuracy_o = sum((test_label_o == Y_test))/length(Y_test)*100;

            %train accuracy MEKA approximation
            myK = Kpp;
            SVMModel_pp = fitcecoc(indices,Y_train,'Learners',t);
            train_label_pp = predict(compact(SVMModel_pp),indices);
            train_accuracy_pp = sum((train_label_pp == Y_train))/length(Y_train)*100;

            %test accuracy MEKA approximation
            Kpp_testing = [Kpp,Kpp_traintest;Kpp_traintest',Kpp_test];
            myK = Kpp_testing;
            test_label_pp = predict(compact(SVMModel_pp),indices_test);
            test_accuracy_pp = sum((test_label_pp == Y_test))/length(Y_test)*100;

            %train accuracy shifted MEKA
            myK = Kc;
            SVMModel_c = fitcecoc(indices,Y_train,'Learners',t);
            train_label_c = predict(compact(SVMModel_c),indices);
            train_accuracy_c = sum((train_label_c == Y_train))/length(Y_train)*100;

            %test accuracy shifted MEKA
            Kc_testing = [Kc,Kc_traintest;Kc_traintest',Kc_test];
            myK = Kc_testing;
            test_label_c = predict(compact(SVMModel_c),indices_test);
            test_accuracy_c = sum((test_label_c == Y_test))/length(Y_test)*100;
            
            %logging all results
            accuracies_o(cv_iter) = test_accuracy_o;
            accuracies_pp(cv_iter) = test_accuracy_pp;
            accuracies_c(cv_iter) = test_accuracy_c;
            train_accuracies_o(cv_iter)  = train_accuracy_o;
            train_accuracies_pp(cv_iter)  = train_accuracy_pp;
            train_accuracies_c(cv_iter)  = train_accuracy_c;
            shifts(cv_iter) = shift;
            Qs{cv_iter} = Q;
            Ls{cv_iter} = L;
            paramss{cv_iter} = params;
            Css(cv_iter) = C;
            rank_mekas(cv_iter) =  rank_meka;
            nocs(cv_iter) =  noc;
            relapprox_pps(cv_iter) = relapprox_pp;
            relapprox_cs(cv_iter) = relapprox_c;
            meka_approx_tunings{cv_iter} = meka_approx_tuning;
        end
        
        fprintf("%s %s summary:\n",kerneltype,dataset);
        fprintf("original: %.2f (\\pm %.2f)\n",mean(accuracies_o),std(accuracies_o));
        fprintf("MEKA: %.2f (\\pm %.2f)\n",mean(accuracies_pp),std(accuracies_pp));
        fprintf("MEKA+shift: %.2f (\\pm %.2f)\n",mean(accuracies_c),std(accuracies_c));
        save("results/"+dataset+"10CV_"+kerneltype+".mat","accuracies_o","accuracies_pp","accuracies_c","train_accuracies_o","train_accuracies_pp","train_accuracies_c","shifts","CVO","relapprox_pps","relapprox_cs","paramss","Css","rank_mekas","nocs","meka_approx_tunings");
    end
    toc
end