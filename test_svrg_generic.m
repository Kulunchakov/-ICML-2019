function [] = main(data,dropout,lambda_factor,seed,nepochs,init_epochs,setting,loss)
% test_svrg_generic 2 0 10 0 300 0 14 0

if isdeployed || (strcmp(class(data),'char'))
   data=str2num(data);
   dropout=str2num(dropout);
   lambda_factor=str2num(lambda_factor);
   seed=str2num(seed);
   nepochs=str2num(nepochs);
   init_epochs=str2num(init_epochs);
   setting=str2num(setting);
   loss=str2num(loss);
end

name=sprintf('exps/exp_data%d_s%d_d%d_l%d_n%d_%d_seed%d_loss%d.mat',data,setting,dropout,lambda_factor,nepochs,init_epochs,seed,loss);
name

if data==1
%%%%% Dataset 1 - CKN %%%%%%
   load('/scratch2/clear/mairal/ckn_matrix.mat');
   X=psiTr;
   n=size(X,2);
   y=-ones(n,1);
   y(find(Ytr==0))=1;
elseif data==2
   %%%%% Dataset 2 - gene100 %%%%%%
   load('/scratch/clear/abietti/data/vant.mat');
   X=X';
   mex_normalize(X);
   y=Y(:,2);
elseif data==3
   load('/scratch/clear/mairal/large_datas/alpha.full_norm.mat');
   y=y(1:250000);
   X=X(:,1:250000);
   mex_normalize(X);
end

n=size(X,2);
param.lambda=1/(lambda_factor*n);  %% This is the regularization parameter
% set up seed for all experiments
param.seed=seed;
param.loss=loss;
nepochs=nepochs;
init_epochs=init_epochs;
X=double(X);
y=double(y);
param.threads=1;
param.dropout=dropout;
if (dropout==0)
   param.eval_freq=1;
else
   param.eval_freq=5;
end
w0=zeros(size(X,1),1);
if loss==0
   L=0.25;
else 
   L=1;
end
param.L = L;
param.setting = setting;
param.store_log = true;

if setting==14
   %%%% Exp 14 - Adam
   param.use_adam=true;
   param.epochs=nepochs;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting==15
   %%%% Exp 15 - Algorithm from Ghadimi and Lan
   param.use_ac_sa=true;
   param.epochs=nepochs;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 16
   %%%% Exp 16 - mb-sgd with 1/L
   param.use_sgd=true;
   param.mb=round(sqrt(param.L/param.lambda));
   param.epochs=nepochs;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 17
   %%%% Exp 17 - search for minimum of the functions
   param.L=L/4;
   param.decreasing=true;
   param.search_min=true;
   param.epochs=1000;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
end

save(name,'logs_exp','w');
