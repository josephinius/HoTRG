/*
 * tensor.hpp
 * HoTRG
 * version 1.0
 *
 * Copyright (C) 2016  Jozef Genzor <jozef.genzor@gmail.com>                                        
 *
 */

/*******************************************************************************************************/
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


#include <iostream>
#include <cmath>
#include <stdlib.h>

class Tensor
{
public: 
	Tensor();
	Tensor(int d_i, int d_j, int d_k, int d_l);
	Tensor(double** const W, int d);
	Tensor(double** const W, int IP, int d); //impurity tensor constructor
 	Tensor(const Tensor & rhs);
	~Tensor();
	
	Tensor & operator= (const Tensor & rhs);
	const double & operator() (int i, int j, int k, int l) const;
	double & operator()(int i, int j, int k, int l);
	Tensor operator* (const Tensor & rhs) const;
	
	int GetD_i() const { return itsD_i; }
	int GetD_j() const { return itsD_j; }
	int GetD_k() const { return itsD_k; }
	int GetD_l() const { return itsD_l; }
	
	int GetSize() const { return itsD_l*itsD_k*itsD_j*itsD_i; }
	double * GetArray() const { return itsM; }
	
	Tensor IndexRotation_neg() const;
	Tensor IndexRotation_pos() const;
	void IndexRotation_neg_replace();
	void IndexRotation_pos_replace();
	
	Tensor CoarseGrain_Y() const;
	Tensor CoarseGrain_X() const;
	
private: 
	void Copy(const Tensor &); //private copy function (for copy constructor and assignment operator)
    double * itsM;  //tensor data (array)
	int itsD_i;     //dimensions
	int itsD_j;
	int itsD_k;
	int itsD_l;
};
