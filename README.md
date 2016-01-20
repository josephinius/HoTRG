# HoTRG

This code is a C++ implementation of the HOTRG algorithm based on the article: 

Coarse-graining renormalization by higher-order singular value decomposition
Phys. Rev. B 86, 045139 (2012), http://arxiv.org/abs/1201.1144v4 

For linear algebra (Singular Value Decomposition (SVD) in particular), 
Eigen library (3.2.7) is called. 

This code is parallelized by OpenMP. 

Compilation under Mac: 

g++ -m64 -O3 -fopenmp -I/.../Eigen main.cpp -o main.x
