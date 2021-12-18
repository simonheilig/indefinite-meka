%this is the setup script for evaluating negative eigenvalues in MEKA
%approximation

rng(100);
datasets = ["spambase","synth1","gesture","synth2","cpusmall","pendigit"]; 

% set kernel
kerneltype = "elm";
kernelfun = @elm_kernel;

% prepare MEKA parameters
mekaOpts(1).noc = 5;
mekaOpts(1).eta = 0.1;
mekaOpts(1).targetrank = 20;

mekaOpts(2).noc = 5;
mekaOpts(2).eta = 0.1;
mekaOpts(2).targetrank = 40;

mekaOpts(3).noc = 5;
mekaOpts(3).eta = 0.1;
mekaOpts(3).targetrank = 80;

mekaOpts(4).noc = 5;
mekaOpts(4).eta = 0.1;
mekaOpts(4).targetrank = 160;

mekaOpts(5).noc = 5;
mekaOpts(5).eta = 0.1;
mekaOpts(5).targetrank = 320;

mekaOpts(6).noc = 10;
mekaOpts(6).eta = 0.1;
mekaOpts(6).targetrank = 20;

mekaOpts(7).noc = 10;
mekaOpts(7).eta = 0.1;
mekaOpts(7).targetrank = 40;

mekaOpts(8).noc = 10;
mekaOpts(8).eta = 0.1;
mekaOpts(8).targetrank = 80;

mekaOpts(9).noc = 10;
mekaOpts(9).eta = 0.1;
mekaOpts(9).targetrank = 160;

mekaOpts(10).noc = 10;
mekaOpts(10).eta = 0.1;
mekaOpts(10).targetrank = 320;

mekaOpts(11).noc = 15;
mekaOpts(11).eta = 0.1;
mekaOpts(11).targetrank = 20;

mekaOpts(12).noc = 15;
mekaOpts(12).eta = 0.1;
mekaOpts(12).targetrank = 40;

mekaOpts(13).noc = 15;
mekaOpts(13).eta = 0.1;
mekaOpts(13).targetrank = 80;

mekaOpts(14).noc = 15;
mekaOpts(14).eta = 0.1;
mekaOpts(14).targetrank = 160;

mekaOpts(15).noc = 15;
mekaOpts(15).eta = 0.1;
mekaOpts(15).targetrank = 320;

%perform on each dataset
for datasetIter = [1:length(datasets)]
    dataset = datasets(datasetIter);
    if dataset == "spambase"
        D = load("spam.mat");
        Y = D.spambase(:,58);
        X = D.spambase(:,1:57);
    elseif dataset=="synth1"
        D = load("synth1.mat");
        Y = D.syntheticblobs(:,3);
        X = D.syntheticblobs(:,1:2);
    elseif dataset == "gesture"
        D = load("gestures.mat");
        Y = D.gestures(:,33);
        X = D.gestures(:,1:32);
    elseif dataset == "cpusmall"
        D = load("cpusmall.mat");
        X = D.X;
        Y = D.Y;
    elseif dataset=="synth2"
        D = load("synth2.mat");
        Y = D.syntheticclassification(:,16);
        X = D.syntheticclassification(:,1:15);
    elseif dataset == "pendigit"
        D = load("pendigits.mat");
        X = D.X;
        Y = D.Y;
    end

    % kernel specific preprocessing
    if kerneltype == "rbf"
        X = bsxfun(@rdivide,bsxfun(@minus,X,min(X)),max(X)-min(X));
        kernelparams = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10];
    elseif kerneltype == "elm"
        X = bsxfun(@rdivide,bsxfun(@minus,X,mean(X)),std(X));
        X = myl2norm(X);
        kernelparams = [1];
    end
    
    try
        tic
        executeMEKA(X,kernelfun,kernelparams,mekaOpts,dataset);
        toc
        
        createFinalFile(dataset, kernelparams);
    catch exception
        fprintf("Error occurred %s in dataset %s",exception.identifier, dataset);
    end
end