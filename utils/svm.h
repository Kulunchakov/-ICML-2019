#ifndef SVM_H
#define SVM_H  
// clc; build; test_saga_generic_simple
#include "linalg.h"

template <typename T>
void calculate_non_zeros(const Vector<T>& tmp) {
   int count = 0;
   const int n = tmp.n();
   for (int ii=0; ii<n; ++ii)
      if (tmp[ii]!=T(0.0)) count++;
   cout << "Percentage of non-zeros:" << 100 * T(count) / T(n) << "%" << endl;
}

// calculates l1 norm of a vector
template <typename T>
T nrm1(const Vector<T>& tmp){
   const int n = tmp.n();
   T nrm1 = T(0);
   for (int ii=0; ii<n; ++ii)
      nrm1 += ABS(tmp[ii]);
   return nrm1;
}


template <typename T>
void lasso_prox(Vector<T>& tmp, const T scale) {
   const int n = tmp.n();
   for (int ii=0; ii<n; ++ii)
      tmp[ii] = T(ABS(tmp[ii]) >= scale) * (tmp[ii] - T(SIGN(tmp[ii])) * scale);
}


template <typename T>
void dropout_vec(Vector<T>& tmp, const T dropout) {
   const int n = tmp.n();
   if (dropout)
      for (int ii=0; ii<n; ++ii)
         if (random() <= RAND_MAX*dropout) tmp[ii]=0;
}

template <typename T>
T compute_loss(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const int loss, const T l1_scale = 0) {
   if (loss==0) {
      return compute_loss_logistic(y,X,w,lambda,dropout,freq,l1_scale);
   } else {
      return compute_loss_sqhinge(y,X,w,lambda,dropout,freq,l1_scale);
   }
}

/// ok no dropout, fine.
template <typename T>
T compute_loss_logistic(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const T l1_scale) {
   const int n = y.n();
   T loss=0;
   Vector<T> tmp;
   if (dropout) {
      for (int ll=0; ll<freq; ++ll) {
         for (int kk=0; kk<n; ++kk) {
            X.copyCol(kk,tmp);
            dropout_vec(tmp,dropout);
            loss += logexp(-y[kk]*tmp.dot(w));
         }
      }
      loss *= T(1.0)/(freq*n);
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         loss += logexp(-y[kk]*tmp[kk]);
      }
      loss *= T(1.0)/(n);
   } 
   loss += T(0.5)*lambda*w.nrm2sq();
   if (l1_scale) loss += l1_scale*nrm1(w);
   return loss;
}


template <typename T>
T compute_loss_sqhinge(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const T l1_scale) {
   const int n = y.n();
   T loss=0;
   Vector<T> tmp;
   if (dropout) {
      for (int ll=0; ll<freq; ++ll) {
         for (int kk=0; kk<n; ++kk) {
            X.copyCol(kk,tmp);
            dropout_vec(tmp,dropout);
            const T los=MAX(0,1-y[kk]*tmp.dot(w));
            loss += los*los;
         }
      }
      loss *= T(0.5)/(freq*n);
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         const T los=MAX(0,1-y[kk]*tmp[kk]);
         loss += los*los;
      }
      loss *= T(0.5)/(n);
   } 
   loss += T(0.5)*lambda*w.nrm2sq();
   if (l1_scale) loss += l1_scale*nrm1(w);
   return loss;
}


template <typename T>
void compute_grad(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout, const int loss) {
   if (loss==0) {
      return compute_grad_logistic(y,x,w,grad,lambda,dropout);
   } else {
      return compute_grad_sqhinge(y,x,w,grad,lambda,dropout);
   }
}


template <typename T>
void compute_grad_logistic(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   grad.copy(x);
   dropout_vec(grad,dropout);
   const T s = T(1.0)/(T(1.0)+exp_alt<T>(y*grad.dot(w)));
   grad.scal(-y*s);
   grad.add(w,lambda);
}


template <typename T>
void compute_grad_sqhinge(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, 
                           const T dropout) {
   grad.copy(x);
   dropout_vec(grad,dropout);
   const T s = MAX(0,1-y*grad.dot(w));
   grad.scal(-y*s);
   grad.add(w,lambda);
}


template <typename T>
void compute_fullgrad(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, 
                     const T dropout, const int loss) {
   if (loss==0) {
      return compute_fullgrad_logistic(y,X,w,grad,lambda,dropout);
   } else {
      return compute_fullgrad_sqhinge(y,X,w,grad,lambda,dropout);
   }
}

template <typename T>
void compute_fullgrad_logistic(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, 
                              const T lambda, const T dropout) {
   const int n = y.n();
   Vector<T> tmp;
   if (dropout) {
      grad.setZeros();
      grad.resize(w.n());
      for (int kk=0; kk<n; ++kk) {
         X.copyCol(kk,tmp);
         dropout_vec(tmp,dropout);
         const T s = T(1.0)/(T(1.0)+exp_alt<T>(y[kk]*tmp.dot(w)));
         grad.add(tmp,-y[kk]*s);
      }
      grad.scal(T(1.0/n));
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         const T s = T(1.0)/(T(1.0)+exp_alt<T>(y[kk]*tmp[kk]));
         tmp[kk]=-y[kk]*s;
      }
      X.mult(tmp,grad,T(1.0)/n);
   }
   grad.add(w,lambda);
}


template <typename T>
void compute_fullgrad_sqhinge(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, 
                              const T lambda, const T dropout) {
   const int n = y.n();
   Vector<T> tmp;
   if (dropout) {
      grad.setZeros();
      grad.resize(w.n());
      for (int kk=0; kk<n; ++kk) {
         X.copyCol(kk,tmp);
         dropout_vec(tmp,dropout);
         const T s=MAX(0,1-y[kk]*tmp.dot(w));
         grad.add(tmp,-y[kk]*s);
      }
      grad.scal(T(1.0/n));
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         tmp[kk]=-y[kk]*MAX(0,1-y[kk]*tmp[kk]);
      }
      X.mult(tmp,grad,T(1.0)/n);
   }
   grad.add(w,lambda);
}


void store_string_to_file(const string& text, const string filename, bool append = true) {
   fstream f;
   if (append) f.open(filename.c_str(), ios_base::out | std::fstream::app);
   else f.open(filename.c_str(), ios_base::out); 
   f << text << "\n";
   f.close();
}


template <typename T>
void store_vector_to_file(const Vector<T>& array, const string filename, bool append = true) {
   fstream f;
   if (append) f.open(filename.c_str(), ios_base::out | std::fstream::app);
   else f.open(filename.c_str(), ios_base::out); 
   for(size_t ii = 0; ii < array.n()-1; ++ ii) 
      f << array[ii] << ' ';
   f << array[array.n()-1] << "\n";
   f.close();
}

// example: exp_data1_s1_d0_l10_n300_0_seed0_loss0
// 'my_exps/exp_data%d_s%d_d%d_l%d_n%d_%d_seed%d_loss%d.mat',dataset,setting,dropout,lambda_factor,nepochs,init_epochs,seed,loss
string create_title_filename(const int setting, const int dropout, const int loss, const int seed) {
   string filename = "my_exps/exp_data2";
   filename += "_s"+to_string(int(setting))+"_d"+to_string(dropout);
   filename += "_l10_n300_0_seed"+to_string(seed);
   filename += "_loss"+to_string(loss)+".txt";
   return filename;
}


template <typename T>
void saga_miso(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, 
               const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const T l1_scale = 0, 
               const T beta = 0, const int param_loss = 1, const int setting = 0, const bool store_log = false, const int seed = 0) {
   const string method_name = beta ? "miso" : "saga";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();
   const T eta= T(1.0)/(12*L);

   // print intro to log_file (todo?)

   logs.resize(epochs);
   Vector<T> wav;
   wav.copy(w);

   Matrix<T> table_grads;
   table_grads.resize(n,p);
   Vector<T> grad_anchor, grad_entry, grad, grad2, col, miso_term, update;
   miso_term.copy(w); 
   miso_term.scal(-beta);

   // initialize gradients table 
   for (int ii = 0; ii<n; ++ii) {
      X.refCol(ii,col);
      compute_grad(y[ii],col,w,grad_entry,lambda,dropout,param_loss);
      if (beta) grad_entry.add(miso_term);
      table_grads.setRow(ii,grad_entry);
   }
   table_grads.meanRow(grad_anchor);

   cout << method_name << endl;   
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = averaging ? compute_loss(y,X,wav,lambda,dropout,eval_freq,param_loss,l1_scale) : compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         const T etak = decreasing ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
         cout << "Iteration " << ii << " - eta: " << etak << " - obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
         // calculate_non_zeros(w);
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);
      table_grads.extractRow(ind, grad2);
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);

      const T etak = decreasing ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);

      // grad_anchor update
      const int jnd = random() % n;
      X.refCol(jnd,col);

      table_grads.extractRow(jnd, grad_entry);
      compute_grad(y[jnd],col,w,update,lambda,dropout,param_loss);
      table_grads.setRow(jnd, update);
      update.add(grad_entry,-T(1.0));
      update.scal(T(1.0) / n);
      if (beta) {
         miso_term.copy(w); 
         miso_term.scal(-beta);
         update.add(miso_term);
      }
      grad_anchor.add(update);

      if (averaging) {
         const T tau = MIN(lambda*etak,T(1.0)/(5*n));
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging)
      w.copy(wav);
   // calculate_non_zeros(w);
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}

template <typename T>
void random_svrg(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, 
                  const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const T l1_scale = 0, 
                  const int param_loss = 1, const int setting = 0, const bool store_log = false, const int seed = 0) {
   const string method_name = "random_svrg";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();
   const T eta= T(1.0)/(12*L);
   logs.resize(epochs);
   Vector<T> wav;
   wav.copy(w);
   Vector<T> anchor;
   anchor.copy(w);
   Vector<T> grad_anchor, grad, grad2, col;
   compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);

   cout << "--SVRG--" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = averaging ? compute_loss(y,X,wav,lambda,dropout,eval_freq,param_loss,l1_scale) : compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         const T etak = decreasing ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
         cout << "Iteration " << ii << " - eta: " << etak << " - obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
         // calculate_non_zeros(w);
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda,dropout,param_loss);
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);

      const T etak = decreasing ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);

      if (random() % n == 0) {
         anchor.copy(w);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);
      }
      if (averaging) {
         const T tau = MIN(lambda*etak,T(1.0)/(5*n));
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging)
      w.copy(wav);
   // calculate_non_zeros(w);
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}


template <typename T>
void acc_random_svrg(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, 
                     const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const T l1_scale = 0, 
                     const int param_loss = 1, const int setting = 0, const bool store_log = false, const int seed = 0) {
   const string method_name = "acc_random_svrg";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();
   const T eta= MIN(T(1.0)/(3*L), T(1.0)/(5*lambda*n));
   logs.resize(epochs);
   Vector<T> anchor, grad_anchor, grad, grad2, col, y_k, v_k;
   anchor.copy(w);
   y_k.copy(w);
   v_k.copy(w);
   compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);

   cout << "Accelerated SVRG" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         const T etak = decreasing ? MIN(eta,12*n/(5*lambda*(T(ii+1)*T(ii+1))))  : eta;
         cout << "Iteration " << ii << " - eta: " << etak << " - obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
         // calculate_non_zeros(w);
      }
      const T etak = decreasing ? MIN(eta,12*n/(5*lambda*(T(ii+1)*T(ii+1))))  : eta;
      const T deltak=sqrt(T(5.0)*etak*lambda/(3*n));
      const T thetak=(3*n*deltak-5*lambda*etak)/(3-5*lambda*etak);
      y_k.copy(v_k);
      y_k.scal(thetak);
      y_k.add(anchor,T(1.0-thetak));

      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,y_k,grad,lambda,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda,dropout,param_loss);
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);
      w.copy(y_k);
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);

      v_k.scal(1-deltak);
      v_k.add(y_k,deltak);
      v_k.add(w,(deltak/(lambda*etak)));
      v_k.add(y_k,-(deltak/(lambda*etak)));

      if (random() % n == 0) {
         anchor.copy(w);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);
      }
   }
   // calculate_non_zeros(w);
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}


template <typename T>
void adam(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, 
   const int epochs, const T dropout, 
   const int eval_freq, Vector<T>& logs, const T l1_scale = 0, const int param_loss = 1, 
   const int setting = 0, const bool store_log = false, const int seed = 0) {

   const string method_name = "adam";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();

   const T alpha = T(0.001);
   const T beta_1 = T(0.9);
   const T beta_2 = T(0.999);
   const T eps = T(0.001*0.001*0.01);
   // powers of beta
   T beta_1_t = beta_1;
   T beta_2_t = beta_2;
   // moments
   Vector<T> moment_1, moment_2, temp;
   Vector<T> moment_1_corr, moment_2_corr; // corrected moments
   moment_1.setZeros();
   moment_1.resize(p);
   moment_2.setZeros();
   moment_2.resize(p);

   Vector<T> eps_vec(p);
   eps_vec.set(eps);

   logs.resize(epochs);
   Vector<T> grad, grad_sq, col;

   cout << "adam" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         cout << "Iteration " << ii << " -- obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);

      moment_1.scal(beta_1);
      moment_1.add(grad, T(1.0) - beta_1);

      grad.mult_elementWise(grad, grad_sq);
      moment_2.scal(beta_2);
      moment_2.add(grad_sq, T(1.0) - beta_2);

      moment_1_corr.copy(moment_1); 
      moment_1_corr.scal(T(1.0) / (T(1.0) - beta_1_t)); 
      moment_2_corr.copy(moment_2); 
      moment_2_corr.scal(T(1.0) / (T(1.0) - beta_2_t)); 

      beta_1_t = beta_1_t * beta_1;
      beta_2_t = beta_2_t * beta_2;

      moment_2_corr.Sqrt();
      moment_2_corr.add(eps_vec);
      moment_2_corr.inv();

      moment_1_corr.mult_elementWise(moment_2_corr, temp);
      w.add(temp,-alpha);

      // if (l1_scale) lasso_prox(w, etak * l1_scale);
   }
   cout << "store_log: " << store_log << "\n";
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}


template <typename T>
void ac_sa(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, 
   const int epochs, const T dropout, const int eval_freq, Vector<T>& logs, 
   const T l1_scale = 0, const int param_loss = 1, 
   const int setting = 0, const bool store_log = false, const int seed = 0) {

   const string method_name = "ac_sa";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();
   logs.resize(epochs);
   Vector<T> grad, col;
   Vector<T> wmd, wag, temp;
   wag.copy(w);

   cout << "ac_sa" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         const T etak = T(2.0) / (T(ii) + T(2.0));
         const T gak = L * T(4.0) / ((T(ii) + 1) * (T(ii) + T(2.0)));
         cout << "Iteration " << ii << " -- eta " << etak << " -- obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
      }
      // Equation (2.5)
      const int t = T(ii + 1);
      const T etak = T(2.0) / (t + T(1.0)); // alpha_k
      const T gak = L * T(4.0) / (t * (t + T(1.0)));
      
      temp.copy(wag);
      temp.scal((T(1.0)-etak)*(lambda + gak) / (gak + (T(1.0)-etak*etak)*lambda));
      wmd.copy(temp);
      temp.copy(w);
      temp.scal(etak*((T(1.0)-etak)*lambda + gak) / (gak + (T(1.0)-etak*etak)*lambda));
      wmd.add(temp);

      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);

      // Equation (2.6)
      const T c = (T(1.0)-etak)*lambda + gak; // right coefficient before V(x_{t-1},x)
      temp.copy(wmd);
      temp.scal(lambda);
      temp.add(w, c);
      temp.add(grad,-etak);
      temp.scal(T(1.0) / (lambda + c));
      w.copy(temp);
      // Equation (2.7)
      temp.scal(etak);
      temp.add(wag, T(1.0) - etak);
      wag.copy(temp);

      if (l1_scale) lasso_prox(w, etak * l1_scale);
   }
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}


template <typename T>
void sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, 
   const int epochs, const bool averaging, const bool decreasing, const T dropout, 
   const int eval_freq, Vector<T>& logs, const T l1_scale = 0, const int param_loss = 1, 
   const int setting = 0, const bool store_log = false, const int seed = 0, const int mb = 1) {

   const string method_name = "sgd";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();
   const T eta= T(1.0)/(L);
   logs.resize(epochs);
   Vector<T> wav;
   wav.copy(w);
   Vector<T> grad, grad2, col;

   cout << "SGD" << endl;
   const int num_iter= (n*epochs)/mb;
   const int freq_epoch= n/mb;
   int last_log=0;
   for (int ii = 0; ii<num_iter; ++ii) {
      if (ii*mb/(eval_freq*n) >= last_log) {
         const T loss = averaging ? compute_loss(y,X,wav,lambda,dropout,eval_freq,param_loss,l1_scale) : compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         const T etak = decreasing ? MIN(eta,T(2.0)/(lambda*(ii+2))) : eta;
         cout << "Iteration " << ii << " -- eta " << etak  << " -- obj " <<  loss << endl;
 //        logs[ii/(eval_freq*n)]=loss;
         logs[last_log++]=loss;
      }

      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);
      if (mb > 1) {
         for (int ii=2; ii<mb; ++ii) {
            const int ind = random() % n;
            X.refCol(ind,col);
            compute_grad(y[ind],col,w,grad2,lambda,dropout,param_loss);
            grad.add(grad2);
         }
         grad.scal(T(1.0)/mb);
      }
      
      const T etak = decreasing ? MIN(eta,T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);
      if (averaging) {
         const T tau = lambda*etak;
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging)
      w.copy(wav);
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}

template <typename T>
void acc_sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, 
   const T lambda, const int epochs, const bool decreasing, const T dropout, 
   const int eval_freq, Vector<T>& logs, const T l1_scale = 0, const int param_loss = 1, 
   const int setting = 0, const bool store_log = false, 
   const int seed = 0, const int mb = 1) {

   const string method_name = "acc_sgd";
   const string filename = create_title_filename(setting, int(dropout * 100), int(param_loss), seed);
   const int n = y.n();
   const int p = X.m();
   const T eta= (T(1.0)/(L));
   logs.resize(epochs);
   Vector<T> yk, wold;
   yk.copy(w);
   Vector<T> grad, col, grad2;

   cout << "Acc SGD" << endl;
   const int num_iter= (n*epochs)/mb;
   const int freq_epoch= n/mb;
   int last_log=0;
   for (int ii = 0; ii<num_iter; ++ii) {
      if (ii*mb/(eval_freq*n) >= last_log) {
//      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss,l1_scale);
         const T etak = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;
         cout << "Iteration " << ii << " -- eta " << etak  << " -- obj " <<  loss << endl;
 //        logs[ii/(eval_freq*n)]=loss;
         logs[last_log++]=loss;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,yk,grad,lambda,dropout,param_loss);
      if (mb > 1) {
         for (int ii=2; ii<mb; ++ii) {
            const int ind = random() % n;
            X.refCol(ind,col);
            compute_grad(y[ind],col,yk,grad2,lambda,dropout,param_loss);
            grad.add(grad2);
         }
         grad.scal(T(1.0)/mb);
      }

      const T etak = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;

      wold.copy(w);
      w.copy(yk);
      w.add(grad,-etak);
      if (l1_scale) lasso_prox(w, etak * l1_scale);
      const T etakp1 = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+3)*T(ii+3))) : eta;
      const T deltak=sqrt(lambda*etak);
      const T deltakp1=sqrt(lambda*etakp1);
      const T betak=deltak*(1-deltak)*etakp1/(etak*deltakp1+ etakp1*deltak*deltak);
      yk.copy(w);
      wold.add(w,-T(1.0));
      yk.add(wold,-betak);
   }
   if (store_log){
      store_vector_to_file(logs, filename);
      store_vector_to_file(w, filename);
   }
}



#endif
