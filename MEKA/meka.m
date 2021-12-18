% the main Memory Efficient Kernel Approximation (MEKA) solver
% solver = meka(A,targetRank,kernelfunction,kernelopts,opts);
% [Q,L,shift] = solver.execute();
classdef meka
    properties
        n
        A
        targetRank
        kernelfunction
        kernelopts
        numberOfClusters
        percentageDeleteBlocks
        pointsRelatedToClusters
        clusterSizes
    end
    methods
    function obj = meka(A,targetRank,kernelfunction,kernelopts,opts)
        if isempty(opts.noc)
            opts.noc = 10;
        end
        if isempty(opts.eta)
            opts.eta = 0.1;
        end
        obj.n = size(A,1);
        obj.A = A;
        obj.targetRank = targetRank;
        obj.kernelfunction = kernelfunction;
        obj.kernelopts = kernelopts;
        obj.numberOfClusters = opts.noc;
        obj.percentageDeleteBlocks = opts.eta;
    end
    
    function [obj,Q,L,shift] = execute(obj)
        rng(100);%only for eigenvalue experiments
        [centerSimilarities, nearestCenterIndex] = obj.executeKmeans();

        threshold = obj.computeThreshold(centerSimilarities);
        
        [obj,orderedIndices] = obj.orderData(nearestCenterIndex);

        rankPerClusters = obj.computeRankPerClusters();

        [L,Q,rankPerClusters] = obj.approximateDiagonalBlocks(rankPerClusters);

        [L] = obj.approximateOffDiagonalBlocks(L, Q, rankPerClusters, centerSimilarities, threshold);

        [Q] = meka.revertClustering(Q, orderedIndices);
        L = sparse(cell2mat(L));
        
        shift = largestNegEigApprox(Q,L,obj.n);
    end

    function [centerSimilarities,nearestCenterIndex] = executeKmeans(obj)
        rng(100);%only for eigenvalue experiments
        if obj.n > 20000
            kmeansSubsampleIndices = randsample(1:obj.n,20000);
        else
            kmeansSubsampleIndices = 1:obj.n;
        end
        maxIterations = 1000;
        [~,centers] = mykmeans(obj.A(kmeansSubsampleIndices,:)', obj.numberOfClusters, maxIterations);
        allDistancesToCenters = pdist2(obj.A,centers','squaredeuclidean');
        [~,nearestCenterIndex] = min(allDistancesToCenters,[],2);

        centerSimilarities = obj.kernelfunction(centers',centers',obj.kernelopts);
    end

    function [threshold] = computeThreshold(obj,centerSimilarities)
        [sortedCenterSimilarities,~] = sort(centerSimilarities(:),'ascend');
        if obj.percentageDeleteBlocks == 0
            threshold = 0;
        else
            indexOfThreshold = ceil((obj.numberOfClusters*obj.numberOfClusters-obj.numberOfClusters)*obj.percentageDeleteBlocks);
            threshold = sortedCenterSimilarities(indexOfThreshold);
        end
    end

    function [obj,orderedIndices] = orderData(obj,nearestCenterIndex)
        obj.pointsRelatedToClusters = cell(obj.numberOfClusters,1);
        orderedIndices = [];
        for i = 1:obj.numberOfClusters
            obj.pointsRelatedToClusters{i} = find(nearestCenterIndex==i);
            orderedIndices = [orderedIndices; obj.pointsRelatedToClusters{i}];
            obj.clusterSizes(i) = numel(obj.pointsRelatedToClusters{i});
        end
    end

    function [rankPerClusters] = computeRankPerClusters(obj)
        rankPerClusters = zeros(obj.numberOfClusters,1);
        for i = 1:obj.numberOfClusters
            rankPerClusters(i) = ceil(obj.targetRank*obj.clusterSizes(i)/sum(obj.clusterSizes));%in case of classification experiment: obj.targetRank;%
        end

        residual = sum(rankPerClusters)-obj.targetRank;
        if residual ~= 0
            rankPerClusters(end) = rankPerClusters(end)-residual;
        end
    end

    function [L, Q, rankPerClusters] = approximateDiagonalBlocks(obj, rankPerClusters)
        for i = 1:obj.numberOfClusters
            pointsIndexCurrentCluster = obj.pointsRelatedToClusters{i};
            sizeOfCurrentCluster = obj.clusterSizes(i);
            currentClusterRank = rankPerClusters(i);
            currentClusterRank = min(currentClusterRank,sizeOfCurrentCluster);
            L{i,i} = eye(currentClusterRank);

            numberOfLandmarks = 2*currentClusterRank;
            numberOfLandmarks = min(numberOfLandmarks, sizeOfCurrentCluster);

            [Q{i}]=nystroem(obj.A(pointsIndexCurrentCluster,:), numberOfLandmarks, currentClusterRank, obj.kernelfunction, obj.kernelopts);
            rankPerClusters(i) = currentClusterRank;
        end
    end

    function [L] = approximateOffDiagonalBlocks(obj, L, Q, rankPerClusters, centerSimilarities, threshold)
        rng(100);%only for eigenvalue experiments
        for i = 1:obj.numberOfClusters
            indicesClusterA = obj.pointsRelatedToClusters{i}; 
            rankClusterA = rankPerClusters(i);
            for j = (i+1):obj.numberOfClusters
                indicesClusterB = obj.pointsRelatedToClusters{j};
                rankClusterB = rankPerClusters(j);

                if centerSimilarities(i,j) < threshold 
                    L{i,j} = zeros(rankClusterA,rankClusterB);
                    L{j,i} = zeros(rankClusterB,rankClusterA);
                else
                    sizeOfClusterA = obj.clusterSizes(i);
                    sizeOfClusterB = obj.clusterSizes(j);
                    sampledRows = randsample(1:sizeOfClusterA,min(3*rankClusterA,sizeOfClusterA)); 
                    sampledColumns = randsample(1:sizeOfClusterB,min(3*rankClusterB,sizeOfClusterB));
                    subsampledBasisA = Q{i}(sampledRows,:);
                    subsampledBasisB = Q{j}(sampledColumns,:);
                    originalBlockSubsampled = obj.kernelfunction(obj.A(indicesClusterA(sampledRows),:),obj.A(indicesClusterB(sampledColumns),:),obj.kernelopts);
                    L{i,j} = pinv(subsampledBasisA'*subsampledBasisA,1e-6)*subsampledBasisA'*originalBlockSubsampled*subsampledBasisB*pinv(subsampledBasisB'*subsampledBasisB,1e-6);
                    L{j,i} = L{i,j}';
                end
            end
        end
    end
    end
    methods(Static)
        function [Q] = revertClustering(Q, orderedIndices)
            [~,revertClusteringIndices] = sort(orderedIndices);
            Q = blkdiag_meka(Q);
            Q = Q(revertClusteringIndices,:);
        end
    end
end