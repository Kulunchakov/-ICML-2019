
/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mex.h>
#include <mexutils.h>
#include <svm.h>

// w=mexSvmMiso(y,X,tablambda,param);

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
      const int nlhs) {
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should not be sparse");

   if (!mexCheckType<T>(prhs[1])) 
      mexErrMsgTxt("type of argument 2 is not consistent");

   if (!mxIsStruct(prhs[3])) 
      mexErrMsgTxt("argument 4 should be a struct");

   T* pry = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsy=mxGetDimensions(prhs[0]);
   INTM my=static_cast<INTM>(dimsy[0]);
   INTM ny=static_cast<INTM>(dimsy[1]);
   Vector<T> y(pry,my*ny);

   T* prX = reinterpret_cast<T*>(mxGetPr(prhs[1]));
   const mwSize* dimsX=mxGetDimensions(prhs[1]);
   INTM p=static_cast<INTM>(dimsX[0]);
   INTM n=static_cast<INTM>(dimsX[1]);
   Matrix<T> X(prX,p,n);

   T* prw0 = reinterpret_cast<T*>(mxGetPr(prhs[2]));
   const mwSize* dimsw0=mxGetDimensions(prhs[2]);
   INTM pw0=static_cast<INTM>(dimsw0[0]);
   INTM nw0=static_cast<INTM>(dimsw0[1]);
   Vector<T> w0(prw0,pw0*nw0);


//   const int nclasses=y.maxval()+1;
//   plhs[0]=createMatrix<T>(p,nclasses);
//   T* prw=reinterpret_cast<T*>(mxGetPr(plhs[0]));
//   Matrix<T> W(prw,p,nclasses);
   plhs[0]=createMatrix<T>(p,1);
   T* prw=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Vector<T> w(prw,p);
   w.copy(w0);

   srandom(0);
   const int epochs = getScalarStructDef<int>(prhs[3],"epochs",100);
   plhs[1]=createMatrix<T>(epochs,1);
   T* prlogs=reinterpret_cast<T*>(mxGetPr(plhs[1]));
   Vector<T> logs(prlogs,epochs);

   int threads = getScalarStructDef<int>(prhs[3],"threads",-1);
   const T lambda = getScalarStruct<T>(prhs[3],"lambda");
   const T l1_scale = getScalarStructDef<T>(prhs[3],"l1_scale",0);
   const int setting = getScalarStruct<int>(prhs[3],"setting");
   const int seed = getScalarStruct<int>(prhs[3],"seed");
   const bool averaging = getScalarStructDef<bool>(prhs[3],"averaging",false);
   const bool decreasing = getScalarStructDef<bool>(prhs[3],"decreasing",false);
   const bool use_adam = getScalarStructDef<bool>(prhs[3],"use_adam",false);
   const bool use_ac_sa = getScalarStructDef<bool>(prhs[3],"use_ac_sa",false);
   const bool use_sgd = getScalarStructDef<bool>(prhs[3],"use_sgd",false);
   const bool search_min = getScalarStructDef<bool>(prhs[3],"search_min",false);
   const bool store_log = getScalarStructDef<bool>(prhs[3],"store_log",true);
   const T L = getScalarStruct<T>(prhs[3],"L");
   const T dropout = getScalarStructDef<T>(prhs[3],"dropout",0);
   const int loss = getScalarStructDef<int>(prhs[3],"loss",1);
   const int eval_freq= getScalarStructDef<int>(prhs[3],"eval_freq",1);
   const int mb = getScalarStructDef<int>(prhs[3],"mb",1);

   srandom(seed);
   const bool accelerated = getScalarStructDef<T>(prhs[3],"accelerated",false);
   if (threads == -1) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   threads=init_omp(threads);
   if (use_adam) {
      adam(y,X,w,L,lambda,epochs,dropout,eval_freq,logs,l1_scale,loss,setting,store_log,seed);
   }
   if (use_ac_sa) {
      ac_sa(y,X,w,L,lambda,epochs,dropout,eval_freq,logs,l1_scale,loss,setting,store_log,seed);
   }
   if (use_sgd) {
      sgd(y,X,w,L,lambda,epochs,averaging,decreasing,dropout,eval_freq,logs,l1_scale,loss,setting,store_log,seed,mb);
   }
   if (search_min) {
      acc_random_svrg(y,X,w,L,lambda,epochs,decreasing,dropout,eval_freq,logs,l1_scale,loss,setting,store_log,seed);
   }
   /*if (accelerated) {
      if (use_sgd) {
         acc_sgd(y,X,w,L,lambda,epochs,decreasing,dropout,eval_freq,logs,l1_scale,loss);
      } else {
         acc_random_svrg(y,X,w,L,lambda,epochs,decreasing,dropout,eval_freq,logs,l1_scale,loss);
      }
   } else {
      if (use_sgd) {
         sgd(y,X,w,L,lambda,epochs,averaging,decreasing,dropout,eval_freq,logs,l1_scale,loss);
      } else {
         random_svrg(y,X,w,L,lambda,epochs,averaging,decreasing,dropout,eval_freq,logs,l1_scale,loss);
      }
   } */
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 4)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 2) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs);
      }
   }




