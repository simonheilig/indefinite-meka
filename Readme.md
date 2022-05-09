# Memory Efficient Kernel Approximation for Non-Stationary and Indefinite Kernels (Heilig, Münch, Schleif)

## 1. Implementation Hints
This package extends the implementation of Si et al. (2014) and provides the experimental setups of the experiments presented in the main paper. 

Compile and Run the program:

- To compile the program, please type "mex -largeArrayDims mykmeans.cpp" in the matlab environment to obtain the fast kmeans. 

- Eigenvalue experiment can be found in "experiments/eigenvalues/setupEigenvalues.m"

- Classification experiment can be found in "experiments/classification/setupClassification.m"

- The main solver is located in "MEKA/meka.m", which is a extended and refactored version of Si et al. (2014)

## 2. Out of Sample Extension
If the model has to be applied to new data points, one would like to modify them in a consistent way with respect to the training scenario. 
In our proposal we provide a strategy for a shift correction and a normalization, which plays a key role and needs to be taken into account, if a new test point **x**' is considered. 

The training model finally consists of the matrices **Q**<sup>i</sup>, the link matrix **L** a shift parameter λ<sub>shift</sub> and an index set **_I_** (block-wise denoted as **_I_**<sub>_i_</sub>), referring to known reference points. Further, the self-similarities
k(**x**,**x**) of the training points need to be stored if a normalization is required for a non-stationary kernel function.

The challenge of an out-of-sample extension can be solved in different ways,
here we suggest two strategies:

(1) In the direct approach, one needs to calculate the kernel evaluations of k(**x**',**x**), with respect to some **x**, e.g. the support vectors, needed in the prediction model. This is done directly on the original, unapproximated kernel function. If the kernel function has to be normalized an additional step, as shown in Remark 1, is required. 
Note that by evaluating self similarities k(**x**,**x**) the parameter λ<sub>shift</sub> needs to be added. This approach is particular useful if the MEKA approximation is very accurate and the evaluation of the kernel function is cheap.

(2) The indirect approach maps the new point in the approximated kernel representation as follows.
From the MEKA algorithm part 2 we have stored the landmark matrices of each cluster **Q**<sup>i</sup>. Additionally, we need to store the cluster-wise matrices (from the SVD) used to generate **Q**<sup>i</sup>, which is rather cheap. Now we calculate the similarities of k(**x**',**x**<sub>_l_</sub>) for each block _i_, using the original kernel function, where _l_ ∈ **_I_**<sub>_i_</sub> are the landmark indices of block _i_. These small landmark vectors are used to generate an extended **Q**<sup>i</sup>. From the enlarged **Q**<sup>i</sup> a block-matrix **Q** is constructed and can be used in the same way as in the MEKA approach. Additional modifications regarding the shift correction and normalization can be applied as shown before.

## 3. Hyperparameters
Typical hyperparameters which are obtained in the classification experiments are listed in following table. Note that the overall rank approximation is _c_ times _k_, since the same rank per cluster strategy was used. The parameters are tuned with a grid-search and 5-fold cross-validation. The classification results were collected via a 10-fold cross-validation and the hyperparameter tuning was executed in each fold. Hence, the following Table provides just an excerpt of the utilized parameters. 

| Data Set | ID | _k_ | _c_ | γ | ρ | _p;a_ | _C_<sub>rbf</sub> | _C_<sub>elm</sub> | _C_<sub>poly</sub> | _C_<sub>tl1</sub> | 
| :---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |
| spambase | 1 | 128    | 3  | 0.1 | 39.9 | 10;2 | 1000|1000| 1000|1|
|artificial 1|2 | 64    |  3 |  1 | 1.4 | 8;2 |   100 |1 | 1000|10|
|cpusmall|3 | 16 | 3 | 10 | 8.4 | 10;2 |10|10|1000|1|
|gesture|4 |  16   |  3 |  15  | 22.4 | 10;2 |1000 |1000|100|1|
|artificial 2|5 |  16  |  3   |  0.1 | 10.5 | 2;3 |  100 | 1|1|1|
|pendigit| 6 | 16 | 3 | 1 | 11.2 | 8;3 | 10 | 100|100|100|

## 4. Kernel Properties and Preprocessing
The next table presents a brief overview of the different kernel functions used in this paper. We do not claim completeness of the listed properties and literature references but show information to support the relevance of an extended MEKA approach. The kernel properties of the Gaussian rbf and polynomial kernel are collected from [1] and in case of the other kernels the information are obtained from the first publishing paper. 

| Kernel | Properties | Publications | Preprocessing |
| :---: | --- | --- | ---|
|Gaussian rbf | γ ≥ 0</br> k<sub>rbf</sub>(x, y) ≥ 0</br>k<sub>rbf</sub>(x, x) = 1</br> infinite dimensional feature space</br> unitary-invariant</br> shift-invariant|introduced in [2] for the SVM; among other things it is used for kernel based methods on EEG signals [3], opinion mining and sentiment analysis [4], ground penetrating radar analysis [5] | [0,1]-Normalization|
|Polynomial|p > 0, q ≥ 0</br>"d + p choose p"-feature space</br> unitary-invariant</br> non-stationary | one of the first ocurrences was [6]; among other things it is used for termite detection [7], person re-identification [8], speaker verification [9] | L2-Normalization |
|Extreme Learning|parameter-insensitive</br> rbf alternative</br> differentiable</br> non-stationary | introduced in [10]; among other things it is used for clustering by fuzzy neural gas [11], alumina concentration estimation [12], predicting wear loss [13] | (σ, µ)-Normalization</br> L2-Normalization |
|Truncated Manhattan | 0 ≤ ρ ≤ d</br> two-level deep piecewise linear</br> compactly supported</br> indefinite</br>ρ = 0.7d stable performance</br> shift-invariant} | introduced in [14]; among other things it is used in LS-SVM and PCA [15], piecewise linear kernel support vector clustering [16] | [0,1]-Normalization |

##### References
<font size="2">
[1] Shawe-Taylor, J. and Cristianini N. (2004). Kernel methods for pattern analysis. Cambridge University Press.</br>
[2] Schölkopf, B., Sung, K.-K., Burges, C. J., Girosi, F., Niyogi, P., Poggio, T., and Vapnik, V. (1997). Comparing support vector machines with gaussian kernels to radial basis function classifiers. IEEE Transactions on Signal Processing, 45(11):2758–2765.</br>
[3] Bajoulvand, A., Zargari Marandi, R., Daliri, M. R., and Sabzpoushan, S. H. (2017). Analysis of folk music preference of people from different ethnic groups using kernel-based methods on eeg signals. Applied Mathematics and Computation, 307:62–70.</br>
[4] Gopi, A. P., Jyothi, R. N. S., Narayana, V. L., and Sandeep, K. S. (2020). Classification of tweets data based on polarity using improved rbf kernel of svm. International Journal of Information Technology, pages 1–16.</br>
[5] Tbarki, K., Said, S. B., Ksantini, R., and Lachiri, Z. (2016). Rbf kernel based svm classification for landmine detection and discrimination. In 2016 International Image Processing, Applications and Systems (IPAS), pages 1–6. IEEE.</br>
[6] Poggio, T. (1975). On optimal nonlinear associative recall. Biological Cybernetics, 19(4):201–209.</br>
[7] Achirul Nanda, M., Boro Seminar, K., Nandika, D., and Maddu, A. (2018). A comparison study of kernel functions in the support vector machine and its application for termite detection. Information, 9(1):5.</br>
[8] Chen, D., Yuan, Z., Hua, G., Zheng, N., and Wang, J. (2015). Similarity learning on an explicit polynomial kernel feature map for person re-identification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1565–1573.</br>
[9] Yaman, S. and Pelecanos, J. (2013). Using polynomial kernel support vector machines for speaker verification. IEEE Signal Processing Letters, 20(9):901–904.</br>
[10] Frénay, B. and Verleysen, M. (2011). Parameter-insensitive kernel in extreme learning for non-linear support vector regression. Neurocomputing, 74(16):2526–2531.</br>
[11] Geweniger, T., Fischer, L., Kaden, M., Lange, M., and Villmann, T. (2013). Clustering by fuzzy neural gas and evaluation of fuzzy clusters. Computational Intelligence and Neuroscience, 2013.</br>
[12] Zhang, S., Zhang, T., Yin, Y., and Xiao, W. (2017). Alumina concentration detection based on the kernel extreme learning machine. Sensors, 17(9):2002.</br>
[13] Ulas, M., Altay, O., Gurgenc, T., and Ozel, C. (2020). A new approach for prediction of the wear loss of pta surface coatings using artificial neural network and basic, kernel-based, and weighted extreme learning machine. Friction, 8(6):1.</br>
[14] Huang, X., Suykens, J. A. K., Wang, S., Hornegger, J., and Maier, A. (2017b). Classification With Truncated ℓ1 Distance Kernel. IEEE Transactions on Neural Networks and Learning Systems, 29(5):2025–2030.</br>
[15] Huang, X., Maier, A., Hornegger, J., and Suykens, J. A. (2017a). Indefinite kernels in least squares support vector machines and principal component analysis. Applied and Computational Harmonic Analysis, 43(1):162–172.</br>
[16] Shang, C., Huang, X., and You, F. (2017). Data-driven robust optimization based on kernel learning. Computers & Chemical Engineering, 106:464–479.</br>
</font>

## 4. Copyright
### Experiments and Xxtended MEKA Copyright
Copyright (c) 2021 Simon Heilig, Maximilian Münch, Frank-Michael Schleif
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Original MEKA Copyright
Copyright (c) 2014 Si Si,  Cho-Jui Hsieh, and Inderjit S. Dhillon
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


