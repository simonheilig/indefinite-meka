function [best_Q,best_L,best_shift,best_rank,best_cluster,relapproxs] = tuneMEKA(X,kernelfun,params,ranks,clusters)
    %tuneMEKA computes the best MEKA approximation in terms of
    %approximation error and memory consumption among all rank and cluster
    %combinations
    
    best_relapprox = inf;
    best_rank = 0;
    best_cluster = 0;
    relapproxs = ones(length(ranks),length(clusters)).*inf;
    rank_iter = 1;

    for rank = ranks
        cluster_iter = 1;
        for cluster = clusters
            fprintf("MEKA optimization %f, %f\n",rank,cluster);
            mekaopts.eta = 0;
            mekaopts.noc = cluster;

            approximation = meka(sparse(X),rank,kernelfun,params,mekaopts);
            [~,Q,L,shift] = approximation.execute();
            if isnan(shift)
                continue
            end
            
            K = kernelfun(X,X,params);
            Kpp = full(Q*L*Q');
            Kpp = .5*(Kpp + Kpp');
            
            relapprox = (norm(K-Kpp,'fro')/norm(K,'fro'));
            if (relapprox*(rank*cluster)) < best_relapprox %best performance with minimal memory consumption
               best_relapprox = relapprox*(rank*cluster);
               best_rank = rank;
               best_cluster = cluster;
               best_Q = Q;
               best_L = L;
               best_shift = shift;
            end
            relapproxs(rank_iter,cluster_iter) = relapprox;
            cluster_iter = cluster_iter + 1;
           
        end
        rank_iter = rank_iter + 1;
       
    end

end