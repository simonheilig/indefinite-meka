function [] = createFinalFile(datasetName, parameter)
    all_p_aft = zeros(length(parameter),15);
    all_n_aft = zeros(length(parameter),15);
    all_r_aft = zeros(length(parameter),15);
    all_ApproxErr = zeros(length(parameter),15);
    all_maxNE_aft = zeros(length(parameter),15);
   
    for i = 1:length(parameter)
        load("tmp/"+datasetName+"/"+datasetName+i+".mat", 'p_prev', 'n_prev', 'r_prev', 'p_aft', 'n_aft', 'r_aft', 'maxNE_prev', 'maxNE_aft', 'ApproxErr');
        all_p_prev(i) = p_prev;
        all_n_prev(i) = n_prev;
        all_r_prev(i) = r_prev;
        all_p_aft(i,:) = p_aft;
        all_n_aft(i,:) = n_aft;
        all_r_aft(i,:) = r_aft;

        all_ApproxErr(i,:) = ApproxErr;
        all_maxNE_aft(i,:) = maxNE_aft; 
        all_maxNE_prev(i) = maxNE_prev;

        clear('p_prev', 'n_prev', 'r_prev', 'p_aft', 'n_aft', 'r_aft', 'maxNE_prev', 'maxNE_aft', 'ApproxErr');
    end
    
    save("tmp/"+datasetName+"/"+datasetName+"_analyzed.mat",'all_p_prev', 'all_n_prev', 'all_r_prev', 'all_p_aft',  'all_n_aft', 'all_r_aft', 'all_ApproxErr', 'all_maxNE_aft', 'all_maxNE_prev');
end