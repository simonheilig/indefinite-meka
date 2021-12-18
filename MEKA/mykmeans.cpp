// This is the MEX wrapper for kmeans provided by Si et al. 
// 
// Invocation form within Matlab:
// [idx centers] = mykmeans(data, k, maxiter, [weight]) 
// 
// input arguments: 
// 		data: the d by n sparse matrix. 
// 		k	: number of clusters. 
// 		maxiter: number of iterations. 
// 		weight (optional): n by 1 matrix (nonnegative). 
//
// output arguments: 
// 		idx: cluster membership
// 		centers: centers

#include "mex.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#define MAXDIS 1000000000000000.0
using namespace std;

void exit_with_help()
{
	mexPrintf(
 "[idx centers] = mykmeans(data, k, maxiter, [weight])\n"
 "input arguments: \n"
 "      data   : the d by n sparse matrix. \n"
 "      k      : number of clusters. \n"
 "      maxiter: number of iterations. \n"
 "      weight (optional): n by 1 matrix (nonnegative). \n"
 "output arguments: \n"
 "      idx: cluster membership\n"
 "      centers: centers\n"
	);
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //srand(100);
	double tol = 1e-10;
    if (nrhs < 3 || nrhs >4 ) {
		exit_with_help();
		fake_answer(plhs);
    } 
	else
	{
		mwIndex *ir, *jc;
		int d = (int)mxGetM(prhs[0]);
		int n = (int)mxGetN(prhs[0]);
		double *values = mxGetPr(prhs[0]);
		long nnz = (long)mxGetNzmax(prhs[0]);
		ir = mxGetIr(prhs[0]);
		jc = mxGetJc(prhs[0]);

		int k = (int)mxGetScalar(prhs[1]);
		int maxiter = (int)mxGetScalar(prhs[2]);
		
		int isweight = 0;
		double totalweight = n;
		double *weight;
		if ( nrhs == 4)
		{
			isweight = 1;
			weight = mxGetPr(prhs[0]);
			// Weight should have n elements, not checked. 
			//

			totalweight = 0;
			for ( int i=0 ; i<n ; i++ )
				totalweight += weight[i];
		}
		plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
		plhs[1] = mxCreateDoubleMatrix(d, k, mxREAL);
		vector<int> idx(n);
		double *idx_out = mxGetPr(plhs[0]);
		double *centers = mxGetPr(plhs[1]);
		double loss;

		vector<double> xnorm(n, 0);
		for ( int i=0 ; i<n ; i++ )
		{
			for ( int ptr = jc[i] ; ptr<jc[i+1] ; ptr++ )
				xnorm[i] += values[ptr]*values[ptr];
		}

		for ( int i=0 ; i<n ; i++ )
			idx[i] = rand()%k;


		for ( int iter = 0 ; iter<maxiter ; iter++ )
		{
		//	printf("iter %d\n", iter);
			double change = 0;
			// Compute centers
			for ( int i=0 ; i<k*d ; i++ )
				centers[i] = 0;
			vector<double> count(k, 0);
			for ( int i=0 ; i<n ; i++ )
			{
				double *nowcenter = &(centers[idx[i]*d]);
				if ( isweight == 0 )
				{
					for ( int ptr = jc[i] ; ptr<jc[i+1] ; ptr++ )
						nowcenter[ir[ptr]] += values[ptr];
					count[idx[i]] += 1;
				} else {
					for ( int ptr = jc[i] ; ptr<jc[i+1] ; ptr++ )
						nowcenter[ir[ptr]] += weight[i]*values[ptr];
					count[idx[i]] += weight[i];
				}
			}
			for ( int i=0 ; i<k ; i++ )
			{
				double *nowcenter = &(centers[i*d]);
				for ( int j=0 ; j<d ; j++)
					nowcenter[j]/=count[i];
			}

			// Compute idx
			vector<double> center_norm(k,0);
			for ( int i=0 ; i<k ; i++ )
			{
				double *nowcenter = &(centers[i*d]);
				for (int j=0 ; j<d ; j++ )
					center_norm[i] += nowcenter[j]*nowcenter[j];
			} 
			loss = 0;
			for ( int i=0 ; i<n ; i++ )
			{
				double dis = MAXDIS;
				int minidx = -1;
				for ( int j=0 ; j<k ; j++ )
				{
					double *nowcenter = &(centers[j*d]);
					double nowdis = 0;
					for ( int ptr = jc[i] ; ptr<jc[i+1] ; ptr++ )
						nowdis += values[ptr]*nowcenter[ir[ptr]];
					nowdis = xnorm[i] - 2*nowdis + center_norm[j];
					if ( nowdis < dis)
					{
						dis = nowdis;
						minidx = j;
					}
				}
				if ( idx[i] != minidx )
				{
					idx[i] = minidx;
					if ( isweight == 0)
						change += 1;
					else
						change += weight[i];
				}
				if (isweight == 0)
					loss += dis;
				else
					loss += weight[i]*dis;
			}

			vector<int> center_count(k,0);
			for ( int i=0 ; i<n ; i++ )
				center_count[idx[i]] += 1;

			for ( int i=0 ; i<k ; i++ )
				if ( center_count[i] == 0)
				{
					while(1)
					{
						int ii = rand()%n;
						if ( center_count[idx[ii]] > 1)
						{
							center_count[idx[ii]]--;
							center_count[i]++;
							idx[ii] = i;
							break;
						}
					}
				}

			//printf("Loss: %lf\n", loss);

			if (change < totalweight*tol)
				break;
		}

		// Finalize, compute centers
		for ( int i=0 ; i<k*d ; i++ )
			centers[i] = 0;
		vector<double> count(k, 0);
		for ( int i=0 ; i<n ; i++ )
		{
			double *nowcenter = &(centers[idx[i]*d]);
			if ( isweight == 0 )
			{
				for ( int ptr = jc[i] ; ptr<jc[i+1] ; ptr++ )
					nowcenter[ir[ptr]] += values[ptr];
				count[idx[i]] += 1;
			} else {
				for ( int ptr = jc[i] ; ptr<jc[i+1] ; ptr++ )
					nowcenter[ir[ptr]] += weight[i]*values[ptr];
				count[idx[i]] += weight[i];
			}
		}
		for ( int i=0 ; i<k ; i++ )
		{
			double *nowcenter = &(centers[i*d]);
			for ( int j=0 ; j<d ; j++)
				nowcenter[j]/=count[i];
		}
		for ( int i=0 ; i<n ; i++ )
			idx_out[i] = idx[i]+1;

		if ( nlhs >=3)
		{
			plhs[2] = mxCreateDoubleScalar(loss);
		}
	}
}
