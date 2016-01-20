/*******************************************************************************************************/
/*                                                                                                     */
/* main.cpp                                                                                            */
/* HoTRG                                                                                               */
/* version 1.0																						   */
/*                                                                                                     */
/* This code is HOTRG algorithm for Ising and Potts model on regular square lattice                    */
/* Copyright (C) 2016  Jozef Genzor <jozef.genzor@gmail.com>                                           */
/*                                                                                                     */
/*                                                                                                     */
/* This file is part of HoTRG.                                                                         */
/*                                                                                                     */
/* HoTRG is free software: you can redistribute it and/or modify                                       */
/* it under the terms of the GNU General Public License as published by                                */
/* the Free Software Foundation, either version 3 of the License, or                                   */
/* (at your option) any later version.                                                                 */
/*                                                                                                     */
/* HoTRG is distributed in the hope that it will be useful,                                            */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of                                      */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                       */
/* GNU General Public License for more details.                                                        */
/*                                                                                                     */
/* You should have received a copy of the GNU General Public License                                   */
/* along with HoTRG.  If not, see <http://www.gnu.org/licenses/>.                                      */
/*                                                                                                     */
/*******************************************************************************************************/

/*
 * This code is an implementation of the HOTRG algorithm based on the article: 
 *
 * Coarse-graining renormalization by higher-order singular value decomposition
 * Phys. Rev. B 86, 045139 (2012), http://arxiv.org/abs/1201.1144v4 
 *
 *
 * For linear algebra (Singular Value Decomposition (SVD) in particular), 
 * Eigen library (3.2.7) is called. 
 * 
 * This code is parallelized by OpenMP. 
 * 
 *
 * Compilation under Mac:
 *
 * g++ -m64 -O3 -fopenmp -I/.../Eigen main.cpp -o main.x
 *
 */

/*******************************************************************************************************/


#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <omp.h>
#include <vector>

#include "SVD"
#include "Eigenvalues"
#include "tensor.hpp"

#define NDEBUG
#include <assert.h>

using namespace Eigen;
using namespace std;

#define  LIMIT_F     1.E-15     //precision for free energy
#define  LIMIT_M     1.E-15     //precision for magnetization
#define  LIMIT_PE    1.E-10     //precision for the entropy 

Tensor::Tensor(): 
itsD_i(2), 
itsD_j(2), 
itsD_k(2), 
itsD_l(2)
{
	itsM = new double[itsD_l*itsD_k*itsD_j*itsD_i];
}

Tensor::Tensor(int d_i, int d_j, int d_k, int d_l): 
itsD_i(d_i), 
itsD_j(d_j), 
itsD_k(d_k), 
itsD_l(d_l)
{
	itsM = new double[GetSize()];
}

Tensor::Tensor(double** const W, int d): 
itsD_i(d), 
itsD_j(d), 
itsD_k(d), 
itsD_l(d)
{
	itsM = new double[itsD_l*itsD_k*itsD_j*itsD_i];
	
	for (int i=0; i<itsD_i; i++) 
	{
		for (int j=0; j<itsD_j; j++) 
		{
			for (int k=0; k<itsD_k; k++) 
			{
				for (int l=0; l<itsD_l; l++) 
				{
					itsM[itsD_k*itsD_j*itsD_i*i + itsD_j*itsD_i*j + itsD_i*k + l] = 0;
					for (int a=0; a<d; a++)
					{
						itsM[itsD_k*itsD_j*itsD_i*i + itsD_j*itsD_i*j + itsD_i*k + l] += W[a][i]*W[a][j]*W[a][k]*W[a][l];
					}
				}
			}
		}
	}
}

//impurity tensor constructor
Tensor::Tensor(double** const W, int IP, int d): 
itsD_i(d), 
itsD_j(d), 
itsD_k(d), 
itsD_l(d)
{
	itsM = new double[itsD_l*itsD_k*itsD_j*itsD_i];
	
	if ( IP==1 ) //Ising initialization
	{		
		for (int i=0; i<itsD_i; i++) 
		{
			for (int j=0; j<itsD_j; j++) 
			{
				for (int k=0; k<itsD_k; k++) 
				{
					for (int l=0; l<itsD_l; l++) 
					{
						itsM[itsD_k*itsD_j*itsD_i*i + itsD_j*itsD_i*j + itsD_i*k + l] = 0;
						for (int a=0; a<d; a++)
						{
							itsM[itsD_k*itsD_j*itsD_i*i + itsD_j*itsD_i*j + itsD_i*k + l] += (2*a - 1)*W[a][i]*W[a][j]*W[a][k]*W[a][l];
						}
					}
				}
			}
		}
	}
	
	if ( IP==2 ) //Potts initialization
	{
		for (int i=0; i<itsD_i; i++) 
		{
			for (int j=0; j<itsD_j; j++) 
			{
				for (int k=0; k<itsD_k; k++) 
				{
					for (int l=0; l<itsD_l; l++) 
					{
						int a=0;
						itsM[itsD_k*itsD_j*itsD_i*i + itsD_j*itsD_i*j + itsD_i*k + l] = W[a][i]*W[a][j]*W[a][k]*W[a][l];
					}
				}
			}
		}
	}
}

Tensor::Tensor(const Tensor & rhs)
{	
	itsD_i = rhs.GetD_i();
	itsD_j = rhs.GetD_j();
	itsD_k = rhs.GetD_k();
	itsD_l = rhs.GetD_l();
	itsM = new double[rhs.GetSize()];
	
	(*this).Copy(rhs);
}

Tensor::~Tensor()
{	
	delete [] itsM;
	itsM = NULL;
	
	itsD_i = 0;
	itsD_j = 0;
	itsD_k = 0;
	itsD_l = 0;
}

Tensor & Tensor::operator= (const Tensor & rhs)
{	
	if ( this == &rhs ) 
	{
		return *this;
	}
	
	delete [] itsM;
	itsD_i = rhs.GetD_i();
	itsD_j = rhs.GetD_j();
	itsD_k = rhs.GetD_k();
	itsD_l = rhs.GetD_l();
	itsM = new double[rhs.GetSize()];
	
	(*this).Copy(rhs);
	
	return *this;
}

const double & Tensor::operator() (int i, int j, int k, int l) const 
{	
	assert ( (i>=0) && (i<itsD_i) );
	assert ( (j>=0) && (j<itsD_j) );
	assert ( (k>=0) && (k<itsD_k) );
	assert ( (l>=0) && (l<itsD_l) );
	
	return itsM[itsD_l*itsD_k*itsD_j*i + itsD_l*itsD_k*j + itsD_l*k + l];
}

double & Tensor::operator()(int i, int j, int k, int l) 
{
	assert ( (i>=0) && (i<itsD_i) );
	assert ( (j>=0) && (j<itsD_j) );
	assert ( (k>=0) && (k<itsD_k) );
	assert ( (l>=0) && (l<itsD_l) );
	
	return itsM[itsD_l*itsD_k*itsD_j*i + itsD_l*itsD_k*j + itsD_l*k + l];
}


Tensor Tensor::operator* (const Tensor & rhs) const //this is coarse-graining in Y
{
	assert ( itsD_l == rhs.GetD_k() ); //boundary test for the summation index
		 
	int t_i = itsD_i*rhs.GetD_i();
	int t_j = itsD_j*rhs.GetD_j();
	int t_k = itsD_k;
	int t_l = rhs.GetD_l();
	
	Tensor temp(t_i, t_j, t_k, t_l); 
		
	int itsD_ip = rhs.GetD_i();
	int itsD_jp = rhs.GetD_j();
	int itsD_lp = rhs.GetD_l();
	
	double sum;
	
#pragma omp parallel shared(temp) private(sum)
	{
#pragma omp for schedule(static)
		for (int i=0; i<itsD_i; i++) 
		{
			for (int j=0; j<itsD_j; j++) 
			{
				for (int k=0; k<itsD_k; k++) 
				{
					for (int i_p=0; i_p<itsD_ip; i_p++) 
					{
						for (int j_p=0; j_p<itsD_jp; j_p++) 
						{
							for (int l_p=0; l_p<itsD_lp; l_p++) 
							{
								sum = 0;
								for (int s=0; s<itsD_l; s++) 
								{
									sum += (*this)(i, j, k, s)*rhs(i_p, j_p, s, l_p);
								}
								temp(itsD_ip*i + i_p, itsD_jp*j + j_p, k, l_p) = sum;
							}
						}
					}
				}
			}
		}
	}
	
	return temp;
}

Tensor Tensor::IndexRotation_neg() const
{
	int t_i = itsD_l;
	int t_j = itsD_k;
	int t_k = itsD_i;
	int t_l = itsD_j;
	
	Tensor temp(t_i, t_j, t_k, t_l);
		
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					temp(i, j, k, l) = (*this)(k, l, j, i);
				}
			}
		}
	}
	
	return temp; 
}

Tensor Tensor::IndexRotation_pos() const
{
	int t_i = itsD_k;
	int t_j = itsD_l;
	int t_k = itsD_j;
	int t_l = itsD_i;
	
	Tensor temp(t_i, t_j, t_k, t_l);
		
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					temp(i, j, k, l) = (*this)(l, k, i, j);
				}
			}
		}
	}
	
	return temp; 
}

void Tensor::IndexRotation_neg_replace()
{
	int t_i = itsD_l;
	int t_j = itsD_k;
	int t_k = itsD_i;
	int t_l = itsD_j;
	
	Tensor temp(t_i, t_j, t_k, t_l);
		
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					temp(i, j, k, l) = (*this)(k, l, j, i);
				}
			}
		}
	}
	
	(*this) = temp;
}

//this rotation function is used in this code 
void Tensor::IndexRotation_pos_replace()
{
	int t_i = itsD_k;
	int t_j = itsD_l;
	int t_k = itsD_j;
	int t_l = itsD_i;
	
	Tensor temp(t_i, t_j, t_k, t_l);
		
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					temp(i, j, k, l) = (*this)(l, k, i, j);
				}
			}
		}
	}
	
	(*this) = temp;
}

Tensor Tensor::CoarseGrain_Y() const //not used here; coarse-graining in Y is done in * operator
{
	if ( (itsD_i != itsD_j) || (itsD_k != itsD_l) ) 
	{
		cout << "incorrect CoarseGrain_Y-ing" << endl;
	}
	
	cout << "in coarse grain y" << endl;
	
	int t_i = itsD_i*itsD_i;
	int t_j = t_i;
	int t_k = itsD_k;
	int t_l = t_k;
	
	Tensor temp(t_i, t_j, t_k, t_l);
	
	double sum;
	
	for (int i=0; i<itsD_i; i++) 
	{
		for (int j=0; j<itsD_j; j++) 
		{
			for (int k=0; k<itsD_k; k++) 
			{
				for (int i_p=0; i_p<itsD_i; i_p++) 
				{
					for (int j_p=0; j_p<itsD_j; j_p++) 
					{
						for (int l_p=0; l_p<itsD_l; l_p++) 
						{
							sum = 0;
							for (int s=0; s<itsD_l; s++) 
							{
								sum += (*this)(i, j, k, s)*(*this)(i_p, j_p, s, l_p);
							}
							temp(itsD_i*i + i_p, itsD_j*j + j_p, k, l_p) = sum;
						}
					}
				}
			}
		}
	}
	
	return temp;
}

Tensor Tensor::CoarseGrain_X() const //not used here
{
	if ( (itsD_i != itsD_j) || (itsD_k != itsD_l) ) 
	{
		cout << "incorrect CoarseGrain_X-ing" << endl;
	}
	
	cout << "in coarse grain x" << endl;
	
	int t_i = itsD_i;
	int t_j = itsD_j;
	int t_k = itsD_k*itsD_k;
	int t_l = itsD_l*itsD_l;
	
	Tensor temp(t_i, t_j, t_k, t_l);
	
	for (int i=0; i<itsD_i; i++) 
	{
		for (int k=0; k<itsD_k; k++) 
		{
			for (int l=0; l<itsD_l; l++) 
			{
				for (int j_p=0; j_p<itsD_j; j_p++) 
				{
					for (int k_p=0; k_p<itsD_k; k_p++) 
					{
						for (int l_p=0; l_p<itsD_l; l_p++) 
						{
							temp(i, j_p, itsD_k*k + k_p, itsD_l*l + l_p) = 0;
							for (int s=0; s<itsD_j; s++) 
							{
								temp(i, j_p, itsD_k*k + k_p, itsD_l*l + l_p) += (*this)(i,s,k,l)*(*this)(s,j_p,k_p,l_p);
							}
						}
					}
				}
			}
		}
	}
	
	return temp;
}

void Tensor::Copy(const Tensor & from) //private copy
{
	//tensors should be of the same dimensions 
	assert ( itsD_i == from.itsD_i );
	assert ( itsD_j == from.itsD_j );
	assert ( itsD_k == from.itsD_k );
	assert ( itsD_l == from.itsD_l );
	
	for (int i=0; i<itsD_i; i++) 
	{
		for (int j=0; j<itsD_j; j++) 
		{
			for (int k=0; k<itsD_k; k++) 
			{
				for (int l=0; l<itsD_l; l++) 
				{
					(*this)(i,j,k,l) = from(i,j,k,l);
				}
			}
		}
	}
}

/*******************************************************************************************************/

void printTensor(const Tensor & rhs);
void checkInput(const int & IP, const int & FM, const int & HO, const int & Par);
void outDescription(const int & IP, const int & d, const int & FM, const int & HO, 
					const int & Par, const double & h, const int & D, const int & N, const int & threads);
void fileDescription(FILE* fw, const int & IP, const int & d, const int & FM, 
					 const int & HO, const int & Par, const int & D, const double & h);

JacobiSVD<MatrixXd> svdFunctionLeft(const Tensor & T);
JacobiSVD<MatrixXd> svdFunctionRight(const Tensor & T);
JacobiSVD<MatrixXd> svdFunctionUpper(const Tensor & T); //not used 
JacobiSVD<MatrixXd> svdFunctionDown(const Tensor & T); // not used

MatrixXd truncation(int *dim, const int D, const JacobiSVD<MatrixXd> & UY1, const JacobiSVD<MatrixXd> & UY2); //for HOSVD optimization on
MatrixXd truncation(int *dim, const int D, const JacobiSVD<MatrixXd> & UY1); //for HOSVD optimization being off

Tensor updateY(const int dim, const MatrixXd & U, const Tensor & M); 
Tensor updateX(const int dim, const MatrixXd & U, const Tensor & M); //not used 

void normalization(Tensor & T, Tensor & IMT, vector<double> & lnZ_aux);
void normalization(Tensor & T, vector<double> & lnZ_aux);

double freeEnergyComput(const double temperature, const Tensor &T, const vector<double> & lnZ_aux, const int iter);
double magnetizationComput(const Tensor & T, const Tensor & IMT);
double pseudoEntropyComput(const int dim, const JacobiSVD<MatrixXd> & U1, vector<double> & sing_val);

void wIsingInit(double** const W, const double & temperature, const double & h);
void wPottsInit(double** const W, const double & temperature, const double & h, const int & d);
void wDecoratedIsingInit(double** const W, const double & temperature, const double & h);
void wDecoratedPottsInit(double** const W, const double & temperature, const double & h, const int & d);

/*
 * The other two quantities, X1 (calculated by x1Function) and X2 (calculated by x2Function), 
 * are the measures of the difference between two fixed-points for the RG flow of the tensors 
 * (as such they are supposed to show where a transition occurs). 
 * Their definitions could be found here: 
 *
 * http://arxiv.org/pdf/1405.1179v1.pdf (Eq. 10), 
 *
 * and particularly in
 *
 * http://arxiv.org/pdf/0903.1069v2.pdf (Eq. 17).
 *
 */ 

double x1Function(const Tensor & T); //calculates X1 (see the comment above)
double x2Function(const Tensor & T); //calculates X2 (see the comment above)

int deltaFunc(const int & i, const int & j);

/*******************************************************************************************************/

int main () 
{
	double start, end;
	start = omp_get_wtime(); 
	
	int IP_r;
	int d_r;
	double h_r; 
	double initial_r; 
	double final_r; 
	double delta_r; 
	int D_r;
	int N_r;
	int FM_r;
	int HO_r;
	int Par_r;
	int threads_r;
	
	FILE* par;
	par = fopen("INIT.txt", "r");
	
	if ( par == NULL ) 
	{ 
		printf("Can't open the input file \"INIT.txt\"\n");
		return 1;
	}
	
	fscanf(par, "%d%*[^\n]", &IP_r); // for Ising (1), for Potts (2)
	fscanf(par, "%d%*[^\n]", &d_r);
	fscanf(par, "%lf%*[^\n]", &h_r);
	fscanf(par, "%lf%*[^\n]", &initial_r); 
	fscanf(par, "%lf%*[^\n]", &final_r);  
	fscanf(par, "%lf%*[^\n]", &delta_r);  
	fscanf(par, "%d%*[^\n]", &D_r);		  
	fscanf(par, "%d%*[^\n]", &N_r);		  
	fscanf(par, "%d%*[^\n]", &FM_r);
	fscanf(par, "%d%*[^\n]", &HO_r);
	fscanf(par, "%d%*[^\n]", &Par_r);
	fscanf(par, "%d%*[^\n]", &threads_r);
	
	fclose(par);
		
	const int IP = IP_r;
	
	if ( IP==1 ) // for Ising 
	{
		d_r = 2;
	}
	const int d = d_r;
	const double h = h_r;
	const double initial = initial_r;
	const double final = final_r;
	const double delta = delta_r;
	const int D = D_r;
	const int N = N_r; //max number of iterations
	const int FM = FM_r;
	const int HO = HO_r;
	const int Par = Par_r;
	const int threads = threads_r;

	checkInput(IP, FM, HO, Par);
	
	omp_set_dynamic(0); //turn off the dynamic mode
	omp_set_num_threads(threads);
	
	outDescription(IP, d, FM, HO, Par, h, D, N, threads);
	
	FILE* fw;
	if ( (fw = fopen("DATA.txt", "w")) == NULL )        //this creates file DATA.txt
	{
		printf("Subor \"DATA.txt\" sa nepodarilo otvorit\n");
		return 1;
	}
	fclose(fw);
	
	fw = fopen("DATA.txt","a");	
	fileDescription(fw, IP, d, FM, HO, Par, D, h);
	fclose(fw);
	
	double temperature;
	
	for (temperature = initial; temperature <= final; temperature += delta) 
	{
		int i;
		double** W = 0;
		W = new double*[d]; //memory allocated for elements of rows
		for(i=0; i<d; ++i)  //memory allocated for elements of each column
		{
			W[i] = new double[d];
		}
		
		if ( Par == 1 ) 
		{
			if ( IP == 1 ) //Ising model
			{
				wDecoratedIsingInit(W, temperature, h);
			}
			else //Potts model
			{
				wDecoratedPottsInit(W, temperature, h, d); 
			}
		}
		else 
		{
			if ( IP == 1 ) //Ising model
			{
				wIsingInit(W, temperature, h); 
			}
			else //Potts model
			{
				wPottsInit(W, temperature, h, d); 
			}
		}

		vector<double> lnZ_aux; //this vector will store all normalizations
		
		lnZ_aux.reserve(N);
				
		Tensor* pTen = new Tensor(W,d);
		Tensor* pIMT = NULL;
		
		if ( FM==1 ) 
		{			
			pIMT = new Tensor(W, IP, d); //calling impurity tensor constructor 
		}
		
		for (i=0; i<d; i++)
		{
			delete [] W[i];
		}
		delete [] W;
		
		int iter = 0;
		double free_energy = 0;
		double free_energy_new = 1;
		
		double mag = 0;
		double mag_new = 1;
		
		double pseudo_entropy = 0;
		double pseudo_entropy_new = 1;
		vector<double> sing_val;
		
		double x1 = 0;
		double x2 = 0;
		
		printf("# Iter\t\tTemp\t\tFree energy\t\tMagnetization\t\tPseudo entr\t\tX1\t\t\tX2\t\t\tSing values\n");
		
		//while (((iter < N) && ((fabs(free_energy - free_energy_new) > LIMIT_F) || (fabs(mag - mag_new) > LIMIT_M))) || (iter <= 3))
		while (((iter < N) && 
				((fabs(free_energy - free_energy_new) > LIMIT_F) || 
				 (fabs(mag - mag_new) > LIMIT_M) || 
				 (fabs(pseudo_entropy - pseudo_entropy_new) > LIMIT_PE) )) || 
			   (iter <= 1))
		{
			Tensor* pM = new Tensor((*pTen)*(*pTen));
			
			Tensor* pMP = NULL;
			if ( FM==1 ) 
			{
				pMP = new Tensor((*pIMT)*(*pTen));
				
				delete pIMT;
				pIMT = NULL;
			}
		
			delete pTen;
			pTen = NULL;

			JacobiSVD<MatrixXd>* pU1;
			JacobiSVD<MatrixXd>* pU2;
			
			if ( HO == 0 ) //HOSVD optimization off
			{
				pU1 = new JacobiSVD<MatrixXd>(svdFunctionLeft(*pM));
			}
			else //HOSVD optimization on 
			{
#pragma omp parallel
				{
#pragma omp sections
					{
#pragma omp section
						pU1 = new JacobiSVD<MatrixXd>(svdFunctionLeft(*pM));
#pragma omp section
						pU2 = new JacobiSVD<MatrixXd>(svdFunctionRight(*pM));
					}
				}
			}

			int dim = (*pM).GetD_i(); //which has to be same as (*pM).GetD_j()
			
			sing_val.clear();
			sing_val.reserve(dim);
			
			pseudo_entropy = pseudo_entropy_new;
			pseudo_entropy_new = pseudoEntropyComput(dim, *pU1, sing_val);
			
			MatrixXd* pU;
			if ( HO == 0 ) 
			{
				pU = new MatrixXd(truncation(&dim, D, *pU1));
			}
			else 
			{
				pU = new MatrixXd(truncation(&dim, D, *pU1, *pU2));
				
				delete pU2;
				pU2 = NULL;
			}
						
			delete pU1;
			pU1 = NULL;

			pTen = new Tensor(updateY(dim, *pU, *pM));
			
			delete pM;
			pM = NULL;
		
			if ( FM==1 ) 
			{
				pIMT = new Tensor(updateY(dim, *pU, *pMP));
								
				delete pMP;
				pMP = NULL;
			}
			
			delete pU;
			pU = NULL;
			
			(*pTen).IndexRotation_pos_replace();
			
			if ( FM==1 ) 
			{
				(*pIMT).IndexRotation_pos_replace();
			}
			
			if ( FM==1 ) 
			{
				normalization(*pTen, *pIMT, lnZ_aux);
			}
			else 
			{
				normalization(*pTen, lnZ_aux);
			}
		
			mag = mag_new;
			if ( FM==1 ) 
			{
				mag_new = magnetizationComput(*pTen, *pIMT);
				if ( IP == 2 ) //for Potts model
				{
					mag_new = (d*mag_new - 1)/(d-1);
				}
			}
						
			free_energy = free_energy_new;
			free_energy_new = freeEnergyComput(temperature, *pTen, lnZ_aux, iter);
			
			x1 = x1Function(*pTen);
			x2 = x2Function(*pTen);
			
			printf("%d\t\t%1.6E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\n", 
				   iter, temperature, free_energy_new, mag_new, pseudo_entropy_new, x1, x2, 
				   sing_val[0], sing_val[1], sing_val[2], sing_val[3]);
		
			iter++;
		}
		
		delete pTen;
		pTen = NULL;
		
		if ( FM==1 ) 
		{
			delete pIMT;
			pIMT = NULL;
		}
				
		fw = fopen("DATA.txt","a");	
		fprintf(fw,"%1.6E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t%1.16E\t\t%d", 
				temperature, free_energy_new, mag_new, pseudo_entropy_new, x1, x2, 
				sing_val[0], sing_val[1], sing_val[2], sing_val[3], iter);
		fprintf(fw,"\n");
		fclose(fw);
		
	}
	
	end = omp_get_wtime();    
	printf ("# Run time is %f seconds\n", end - start);
	
	return 0;
}

/*******************************************************************************************************/

void printTensor(const Tensor & rhs)
{	
	int d_i = rhs.GetD_i();
	int d_j = rhs.GetD_j();
	int d_k = rhs.GetD_k();
	int d_l = rhs.GetD_l();
	
	for (int i=0; i<d_i; i++) 
	{
		for (int j=0; j<d_j; j++) 
		{
			for (int k=0; k<d_k; k++) 
			{
				for (int l=0; l<d_l; l++) 
				{
					cout << "(" << i <<"," << j << "," << k << "," << l << ") = " << rhs(i,j,k,l) << endl;
				}
			}
		}
	}
}

void checkInput(const int & IP, const int & FM, const int & HO, const int & Par)
{
	if ( IP != 1 && IP != 2 ) 
	{
		cout << "choose correct value for Ising (1) or Potts (2) in INIT.txt" << endl;
		abort();
	}
	if ( FM != 0 && FM != 1 ) 
	{
		cout << "choose correct value for f (0) or m (1) in INIT.txt" << endl;
		abort();
	}
	if ( HO != 0 && HO != 1 ) 
	{
		cout << "choose correct value for \"HOSVD optimization\": OFF (0) or ON (1) in INIT.txt" << endl;
		abort();
	}
	if ( Par != 0 && Par != 1 ) 
	{
		cout << "choose correct value for parametrization: (0) or (1) in INIT.txt" << endl;
		abort();
	}
}

void outDescription(const int & IP, const int & d, const int & FM, const int & HO, 
					const int & Par, const double & h, const int & D, const int & N, const int & threads)
{
	if ( IP==1 ) 
	{
		printf("# Ising model\n");
	}
	else 
	{
		printf("# %d-states Potts model\n", d);
	}
	
	if ( FM==1 ) 
	{
		printf("# calculation of impurity tensor\n");
	}
	else 
	{
		printf("# calculaton of free energy only\n");
	}
	
	if ( HO==1 ) 
	{
		printf("# \"HOSVD optimization\" is on\n");
	}
	else 
	{
		printf("# \"HOSVD optimization\" is off\n");
	}
	
	if ( Par==1 ) 
	{
		printf("# Fisher's parametrization is employed\n");
	}
	else 
	{
		if ( IP==1 ) 
		{
			printf("# Xiang's parametrization is employed\n");
		}
		else 
		{
			printf("# numerical factorization is employed\n");
		}
	}
	
	printf("# h = %1E\t\tD = %d\t\tN = %d\t\tthreads = %d\n", h, D, N, threads);
}

void fileDescription(FILE* fw, const int & IP, const int & d, const int & FM, 
					 const int & HO, const int & Par, const int & D, const double & h)
{
	if ( IP==1 ) 
	{
		fprintf(fw, "# Ising model\n");
	}
	else 
	{
		fprintf(fw, "# %d-states Potts model\n", d);
	}
	if ( FM==1 ) 
	{
		fprintf(fw, "# calculation of impurity tensor\n");
	}
	else 
	{
		fprintf(fw, "# calculation of free energy only\n");
	}
	if ( HO==1 ) 
	{
		fprintf(fw, "# \"HOSVD optimization\" is on\n");
	}
	else 
	{
		fprintf(fw, "# \"HOSVD optimization\" is off\n");
	}
	if ( Par==1 ) 
	{
		fprintf(fw, "# Fisher's parametrization\n");
	}
	else 
	{
		if ( IP==1 ) 
		{
			fprintf(fw, "# Xiang's parametrization is employed\n");
		}
		else 
		{
			fprintf(fw, "# numerical factorization is employed\n");
		}
	}
	fprintf(fw, "# D=%d, h=%1E\n", D, h);
	fprintf(fw, "# Temp\t\tFree energy\t\tMagnetization\t\tPseudo entr\t\tX1\t\t\tX2\t\t\tSing values\t\t\t\t\t\t\t\t\t\t\t\tIter\n");
	//fprintf(fw, "# Temp\t\tFree energy\t\t\tIter\n");
}

JacobiSVD<MatrixXd> svdFunctionLeft(const Tensor & T)
{
	//first unfolding 
		
	MatrixXd A_1(T.GetD_i(),T.GetD_j()*T.GetD_k()*T.GetD_l());
	
	for (int i=0; i<T.GetD_i(); i++) 
	{
		for (int j=0; j<T.GetD_j(); j++) 
		{
			for (int k=0; k<T.GetD_k(); k++) 
			{
				for (int l=0; l<T.GetD_l(); l++) 
				{
					A_1(i, T.GetD_l()*T.GetD_k()*j + T.GetD_l()*k + l) = T(i,j,k,l);
				}
			}
		}
	}
	
	//singular value decomposition
	
	//cout << "Here is the matrix A_1:" << endl << A_1 << endl;
	JacobiSVD<MatrixXd> svd_1(A_1, ComputeThinU); //this will not compute V!
	//JacobiSVD<MatrixXd> svd(A_1, ComputeThinU | ComputeThinV);
	//cout << "Its singular values are:" << endl << svd_1.singularValues() << endl;
	//cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd_1.matrixU() << endl;
	//cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd_1.matrixV() << endl;
	
	return svd_1;
}

JacobiSVD<MatrixXd> svdFunctionRight(const Tensor & T)
{
	//second unfolding 
		
	MatrixXd A_2(T.GetD_j(), T.GetD_k()*T.GetD_l()*T.GetD_i());
	
	for (int i=0; i<T.GetD_i(); i++) 
	{
		for (int j=0; j<T.GetD_j(); j++) 
		{
			for (int k=0; k<T.GetD_k(); k++) 
			{
				for (int l=0; l<T.GetD_l(); l++) 
				{
					A_2(j, T.GetD_i()*T.GetD_l()*k + T.GetD_i()*l + i) = T(i,j,k,l);
				}
			}
		}
	}
	
	//singular value decomposition
	
	//cout << "Here is the matrix A_2:" << endl << A_2 << endl;
	JacobiSVD<MatrixXd> svd_2(A_2, ComputeThinU); //this will not compute V!
	//JacobiSVD<MatrixXd> svd_2(A_2, ComputeThinU | ComputeThinV);
	//cout << "Its singular values are:" << endl << svd_2.singularValues() << endl;
	//cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd_2.matrixU() << endl;
	//cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd_2.matrixV() << endl;
	
	return svd_2;	
}

JacobiSVD<MatrixXd> svdFunctionUpper(const Tensor & T)
{
	//third unfolding
		
	MatrixXd A_3(T.GetD_k(), T.GetD_l()*T.GetD_i()*T.GetD_j());
	
	for (int i=0; i<T.GetD_i(); i++) 
	{
		for (int j=0; j<T.GetD_j(); j++) 
		{
			for (int k=0; k<T.GetD_k(); k++) 
			{
				for (int l=0; l<T.GetD_l(); l++) 
				{
					A_3(k, T.GetD_j()*T.GetD_i()*l + T.GetD_j()*i + j) = T(i,j,k,l);
				}
			}
		}
	}
	
	//singular value decomposition
	
	//cout << "Here is the matrix A_3:" << endl << A_3 << endl;
	JacobiSVD<MatrixXd> svd_3(A_3, ComputeThinU); //this will not compute V!
	//JacobiSVD<MatrixXd> svd_3(A_3, ComputeThinU | ComputeThinV);
	//cout << "Its singular values are:" << endl << svd_3.singularValues() << endl;
	//cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd_3.matrixU() << endl;
	//cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd_3.matrixV() << endl;
	
	return svd_3;
}

JacobiSVD<MatrixXd> svdFunctionDown(const Tensor & T)
{
	//fourth unfolding
		
	MatrixXd A_4(T.GetD_l(), T.GetD_i()*T.GetD_j()*T.GetD_k());
	
	for (int i=0; i<T.GetD_i(); i++) 
	{
		for (int j=0; j<T.GetD_j(); j++) 
		{
			for (int k=0; k<T.GetD_k(); k++) 
			{
				for (int l=0; l<T.GetD_l(); l++) 
				{
					A_4(l, T.GetD_k()*T.GetD_j()*i + T.GetD_k()*j + k) = T(i,j,k,l);
				}
			}
		}
	}
	
	//singular value decomposition
	
	//cout << "Here is the matrix A_4:" << endl << A_4 << endl;
	JacobiSVD<MatrixXd> svd_4(A_4, ComputeThinU); //this will not compute V!
	//JacobiSVD<MatrixXd> svd_4(A_4, ComputeThinU | ComputeThinV);
	//cout << "Its singular values are:" << endl << svd_4.singularValues() << endl;
	//cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd_4.matrixU() << endl;
	//cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd_4.matrixV() << endl;
	
	return svd_4;
}

MatrixXd truncation(int *dim, const int D, const JacobiSVD<MatrixXd> & U1, const JacobiSVD<MatrixXd> & U2)
{
	if ( (*dim) > D ) 
	{
		//calculate epsilon 1
		double epsilon_1 = 0;
		for (int i=D; i<(*dim); i++) 
		{
			epsilon_1 += U1.singularValues()(i);
		}
		
		//cout << "epsilon_1 = " << setprecision(18) << epsilon_1 << endl;
		//cout << "1-mode singular values are: " << endl << U1.singularValues() << endl;
		
		//calculate epsilon 2
		double epsilon_2 = 0;
		for (int i=D; i<(*dim); i++) 
		{
			epsilon_2 += U2.singularValues()(i);
		}
		
		//cout << "epsilon_2 = " << setprecision(18) << epsilon_2 << endl;
		//cout << "2-mode singular values are: " << endl << U2.singularValues() << endl;
		
		MatrixXd U((*dim),D); 
		
		if ( epsilon_2 < epsilon_1 ) 
		{
			for (int i=0; i<(*dim); i++) 
			{
				for (int j=0; j<D; j++) 
				{						
					U(i,j) = U2.matrixU()(i,j);
				}
			}
		}
		else 
		{
			for (int i=0; i<(*dim); i++) 
			{
				for (int j=0; j<D; j++) 
				{						
					U(i,j) = U1.matrixU()(i,j);
				}
			}
		}
		
		(*dim) = D;		
		
		return U;
	}
	else 
	{
		MatrixXd U((*dim),(*dim));
		
		for (int a=0; a<(*dim); a++) 
		{
			for (int b = 0; b<(*dim); b++) 
			{						
				U(a,b) = U1.matrixU()(a,b);
			}
		}
		
		return U;
	}
}

MatrixXd truncation(int *dim, const int D, const JacobiSVD<MatrixXd> & U1)
{
	if ( (*dim) > D ) 
	{
		MatrixXd U((*dim),D); 
		
		for (int i=0; i<(*dim); i++) 
		{
			for (int j=0; j<D; j++) 
			{						
				U(i,j) = U1.matrixU()(i,j);
			}
		}
		
		(*dim) = D;
		
		return U;
	}
	else 
	{
		MatrixXd U((*dim),(*dim));
		
		for (int a=0; a<(*dim); a++) 
		{
			for (int b = 0; b<(*dim); b++) 
			{						
				U(a,b) = U1.matrixU()(a,b);
			}
		}
		
		return U;
	}
}


Tensor updateY(const int dim, const MatrixXd & U, const Tensor & M)
{
	int t_i = dim;
	int t_j = dim;
	int t_k = M.GetD_k();
	int t_l = M.GetD_l();

	assert ( M.GetD_i() == M.GetD_j() );
	assert ( dim == U.cols() && M.GetD_i() == U.rows() );
	
	double sum;
	
	double* UMY = new double[t_i*M.GetD_j()*t_k*t_l];
		
#pragma omp parallel private(sum)
	{		
#pragma omp for schedule(static)
		for (int x=0; x<t_i; x++) 
		{
			for (int j=0; j<M.GetD_j(); j++) 
			{
				for (int y=0; y<t_k; y++) 
				{
					for (int y_p=0; y_p<t_l; y_p++) 
					{
						sum = 0;
						for (int i=0; i<M.GetD_i(); i++) 
						{
							sum += U(i,x)*M(i,j,y,y_p);
						}
						UMY[t_l*t_k*M.GetD_j()*x + t_l*t_k*j + t_l*y + y_p] = sum;
					}
				}
			}
		}
	}
		
	Tensor temp(t_i, t_j, t_k, t_l); 
	
#pragma omp parallel private(sum)
	{		
#pragma omp for schedule(static)
		for (int x=0; x<t_i; x++) 
		{
			for (int x_p=0; x_p<t_j; x_p++) 
			{
				for (int y=0; y<t_k; y++) 
				{
					for (int y_p=0; y_p<t_l; y_p++) 
					{
						sum = 0;
						for (int j=0; j<M.GetD_j(); j++) 
						{
							sum += UMY[t_l*t_k*M.GetD_j()*x + t_l*t_k*j + t_l*y + y_p]*U(j,x_p);
						}
						temp(x,x_p,y,y_p) = sum;
					}
				}
			}
		}
	}
		
	delete [] UMY;
		
	return temp;
}
	
Tensor updateX(const int dim, const MatrixXd & U, const Tensor & M)
{
	int t_i = M.GetD_i();
	int t_j = M.GetD_j();
	int t_k = dim;
	int t_l = dim;
	
	assert ( M.GetD_k() == M.GetD_l() );
	assert ( dim == U.cols() && M.GetD_k() == U.rows() );
	
	double sum;
	
	double* UMX = new double[M.GetD_l()*t_k*t_j*t_i];
	
#pragma omp parallel private(sum)
	{
#pragma omp for schedule(static)
		for (int x=0; x<t_i; x++) 
		{
			for (int x_p=0; x_p<t_j; x_p++) 
			{
				for (int y=0; y<t_k; y++) 
				{
					for (int l=0; l<M.GetD_l(); l++) 
					{
						sum = 0;
						for (int k=0; k<M.GetD_k(); k++) 
						{
							sum += U(k,y)*M(x,x_p,k,l);
						}
						UMX[M.GetD_l()*t_k*t_j*x + M.GetD_l()*t_k*x_p + M.GetD_l()*y + l] = sum;
					}
				}
			}
		}
	}
	
	Tensor temp(t_i, t_j, t_k, t_l); 
	
#pragma omp parallel private(sum)
	{
#pragma omp for schedule(static)
		for (int x=0; x<t_i; x++) 
		{
			for (int x_p=0; x_p<t_j; x_p++) 
			{
				for (int y=0; y<t_k; y++) 
				{
					for (int y_p=0; y_p<t_l; y_p++) 
					{
						sum = 0;
						for (int l=0; l<M.GetD_l(); l++) 
						{
							sum += U(l,y_p)*UMX[M.GetD_l()*t_k*t_j*x + M.GetD_l()*t_k*x_p + M.GetD_l()*y + l];
						}
						temp(x,x_p,y,y_p) = sum;
					}
				}
			}
		}
	}
	
	delete [] UMX;
			
	return temp;
}

double freeEnergyComput(const double temperature, const Tensor &T, const vector<double> & lnZ_aux, const int iter)
{
	double free_energy;
	double size = pow(2.0, (iter + 1));
	//cout << "system size is: " << size << endl;
	double sum = 0;
	
	for (int i=0; i<=(iter); i++) 
	{
		sum += pow(2.0, (iter - i))*log(lnZ_aux[i]);
	}
	
	int t_i = T.GetD_i();
	int t_k = T.GetD_k();
	
	double correction = 0;
		
	for (int k=0; k<t_k; k++) 
	{
		for (int i=0; i<t_i; i++) 
		{
			correction += T(i,i,k,k);
		}
	}
	
	free_energy = - temperature*(sum + log(correction))/size;
	
	return free_energy;
}

void normalization(Tensor & T, vector<double> & lnZ_aux)
{
	int t_i = T.GetD_i();
	int t_j = T.GetD_j();
	int t_k = T.GetD_k();
	int t_l = T.GetD_l();
	
	double norm = fabs(T(0,0,0,0));
	
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					if ( fabs(T(i,j,k,l)>norm) ) 
					{
						norm = fabs(T(i,j,k,l));
					}
				}
			}
		}
	}
	
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					T(i,j,k,l) /= norm;
				}
			}
		}
	}
	
	lnZ_aux.push_back(norm);
}

void normalization(Tensor & T, Tensor & IMT, vector<double> & lnZ_aux)
{
	int t_i = T.GetD_i(); //IMT.GetD_i() is equal to t_i ...
	int t_j = T.GetD_j();
	int t_k = T.GetD_k();
	int t_l = T.GetD_l();
		
	double norm = fabs(T(0,0,0,0));
	
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					if ( fabs(T(i,j,k,l)>norm) ) 
					{
						norm = fabs(T(i,j,k,l));
					}
				}
			}
		}
	}
				
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_j; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_l; l++) 
				{
					T(i,j,k,l) /= norm;
					IMT(i,j,k,l) /= norm;
				}
			}
		}
	}
	
	lnZ_aux.push_back(norm);
}

double magnetizationComput(const Tensor & T, const Tensor & IMT)
{
	double mag;
	
	int t_i = T.GetD_i(); // = IMT.GetD_i()
	int t_k = T.GetD_k(); // = IMT.GetD_k()
	
	double imt_tr=0;
	double ten_tr=0;
	
	for (int i=0; i<t_i; i++) 
	{
		for (int k=0; k<t_k; k++) 
		{
			ten_tr += T(i,i,k,k);
			imt_tr += IMT(i,i,k,k);
		}
	}
	
	mag = imt_tr/ten_tr;
			
	return mag;
}

double x1Function(const Tensor & T)
{
	int t_i = T.GetD_i(); 
	int t_k = T.GetD_k(); 
	
	double trA = 0;
	
	for (int i=0; i<t_i; i++) 
	{
		for (int k=0; k<t_k; k++) 
		{
			trA += T(i,i,k,k);
		}
	}
	
	MatrixXd ST(t_i, t_i);
	
	double sum;
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_i; j++) 
		{
			sum=0;
			for (int k=0; k<t_k; k++) 
			{
				sum += T(i,j,k,k);
			}
			ST(i,j) = sum;
		}
	}
	
	double trA2 = 0;
	
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_i; j++) 
		{
			trA2 += ST(i,j)*ST(j,i);
		}
	}
	
	double X1 = trA*trA/trA2; 
	
	return X1;
}

double x2Function(const Tensor & T)
{
	int t_i = T.GetD_i(); 
	int t_k = T.GetD_k(); 
	
	double trA = 0;
	
	for (int i=0; i<t_i; i++) 
	{
		for (int k=0; k<t_k; k++) 
		{
			trA += T(i,i,k,k);
		}
	}
	
	double trA2 = 0;
	
	for (int i=0; i<t_i; i++) 
	{
		for (int j=0; j<t_i; j++) 
		{
			for (int k=0; k<t_k; k++) 
			{
				for (int l=0; l<t_k; l++) 
				{
					trA2 += T(i,j,k,l)*T(j,i,l,k);
				}
			}
		}
	}
	
	double X2 = trA*trA/trA2; 
	
	return X2;
}

void wDecoratedIsingInit(double** const W, const double & temperature, const double & h)
{
	double b = log(sqrt(exp(2/temperature) + sqrt(exp(4/temperature) - 1))); //temperature rescalling
	
	for (int s=0; s<2; s++) 
	{
		for (int i=0; i<2; i++) 
		{
			//field term requires unrescaled temperature
			W[s][i] = exp(b*(1-2*((double) s))*(1-2*((double) i)))*exp(h*(1-2*((double) s))/(4*temperature))/sqrt(2*sqrt(cosh(2*b)));
		}
	}
}

void wIsingInit(double** const W, const double & temperature, const double & h)
{
	double b = 1/temperature;
	
	W[0][0] =   sqrt(cosh(b))*exp(b*h/4); 
	W[0][1] =   sqrt(sinh(b))*exp(b*h/4);
	W[1][0] =   sqrt(cosh(b))*exp(-b*h/4);
	W[1][1] = - sqrt(sinh(b))*exp(-b*h/4);
}

void wDecoratedPottsInit(double** const W, const double & temperature, const double & h, const int & d)
{
	double b = log(exp(1/temperature) + sqrt((exp(1/temperature) + d - 1)*(exp(1/temperature) - 1))); //temperature rescalling 
	
	for (int s=0; s<d; s++) 
	{
		for (int i=0; i<d; i++) 
		{
			//field term requires unrescaled temperature
			W[s][i] = exp(b*deltaFunc(s,i))*exp(h*deltaFunc(s,0)/(4*temperature))/sqrt(d - 2 + 2*exp(b));
		}
	}
}

void wPottsInit(double** const W, const double & temperature, const double & h, const int & d)
{
	double b = 1/temperature;	
	MatrixXd B(d,d); 
	
	B(0,0) = exp(b+b*h/2);
	for (int j=1; j<d; j++) 
	{
		B(0,j) = exp(b*h/4);
	}
	for (int i=1; i<d; i++) 
	{
		B(i,0) = exp(b*h/4);
	}
	for (int i=1; i<d; i++) 
	{
		B(i,i) = exp(b);
	}
	for (int i=1; i<(d-1); i++) 
	{
		for (int j=(i+1); j<d; j++) 
		{
			B(i,j) = 1;
		}
	}
	for (int j=1; j<d; j++) 
	{
		for (int i=(j+1); i<d; i++) 
		{
			B(i,j) = 1;
		}
	}
	
	SelfAdjointEigenSolver<MatrixXd> ES(B);
	
	//cout << "The eigenvalues of B are:" << endl << ES.eigenvalues() << endl;
	//cout << "The matrix of eigenvectors, V, is:" << endl << ES.eigenvectors() << endl;
	//double lambda = ES.eigenvalues()[0];
	//cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
	//VectorXd v = ES.eigenvectors().col(0);
	//cout << "v = " << endl << v << endl;
	//cout << "If v is the corresponding eigenvector, then lambda * v = " << endl << lambda * v << endl;
	//cout << "... and B * v = " << endl << B * v << endl << endl;
	
	MatrixXd V = ES.eigenvectors();
	
	MatrixXd D = ES.eigenvalues().asDiagonal();
	//cout << "V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;
	
	for (int i=0; i<d; i++) 
	{
		D(i,i) = sqrt(ES.eigenvalues()[i]);
	}
	
	MatrixXd W_m(V*D);
	
	//cout << "B is: " << endl << B << endl;
	//cout << "W_m*W_m^(T)= " << endl << W_m*W_m.transpose() << endl;
	 
	for (int i=0; i<d; i++) 
	{
		for (int j=0; j<d; j++) 
		{
			W[i][j] = W_m(i,j);
		}
	}
}

double pseudoEntropyComput(const int dim, const JacobiSVD<MatrixXd> & U1, vector<double> & sing_val)
{
	for (int i=0; i<dim; i++) 
	{
		sing_val.push_back(U1.singularValues()(i));
	}
	
	double norm = 0;
	
	for (int i=0; i<dim; i++) 
	{
		if ( (sing_val[i]*sing_val[i])<1.E-32 )
		{
			break;
		}
		norm += sing_val[i]*sing_val[i];
	}
	
	norm = sqrt(norm);
	
	for (int i=0; i<dim; i++) 
	{
		sing_val[i] /= norm;
	}
	
	double pseudo_entropy = 0;
	
	for (int i=0; i<dim; i++) 
	{
		if ( (sing_val[i]*sing_val[i])<1.E-32 )
		{
			break;
		}
		//pseudo_entropy += - sing_val[i]*sing_val[i]*log2(sing_val[i]*sing_val[i]);
		pseudo_entropy += - sing_val[i]*sing_val[i]*log(sing_val[i]*sing_val[i]); //natural logarithm 
	}
	
	return pseudo_entropy;
}

int deltaFunc(const int & i, const int & j) 
{
	return (i == j);
}
