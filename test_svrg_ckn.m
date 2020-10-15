%%%%% Dataset 1 - CKN %%%%%%
   load('/scratch2/clear/mairal/ckn_matrix.mat');
   X=psiTr;
   n=size(X,2);
   y=-ones(n,1);
   y(find(Ytr==0))=1;

%%%%% Dataset 2 - Gene %%%%%%
%  load('/scratch/clear/abietti/data/vant.mat');
%  X=X';
%  mex_normalize(X);
%  y=Y(:,2);

n=size(X,2);
param.lambda=1/(10*n);  %% This is the regularization parameter
% set up seed for all experiments
param.seed=0;
nepochs=300;

%%%% Exp 1 - without noise - SVRG with 1/12L
param.dropout=0; % Dropout rate
X=double(X);
y=double(y);
param.L=1;
param.accelerated=false;
param.threads=1;
param.decreasing=false;
param.averaging=false; % seems to always hurt
param.epochs=nepochs;
w0=zeros(size(X,1),1);
[w logs_exp1]=mex_svm_svrg(y,X,w0,param);

%%%% Exp 2 - without noise - SVRG with 1/3L (not allowed by theory)
param.L=0.25;
[w logs_exp2]=mex_svm_svrg(y,X,w0,param);

%%%% Exp 3 - without noise - accelerated SVRG - constant step size
param.L=1;
param.accelerated=true;
[w logs_exp3]=mex_svm_svrg(y,X,w0,param);

%%%% Exp 4 - with noise - SVRG based on 1/12L
param.dropout=0.01; % Dropout rate
param.L=1;
param.epochs=10;
param.accelerated=false;
param.decreasing=false;
[w logs_exp4a]=mex_svm_svrg(y,X,w0,param); % constant step size regime for 10 epochs
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp4b]=mex_svm_svrg(y,X,w,param);

%%%% Exp 5 - with noise - SVRG based on 1/3L
param.L=0.25;
param.epochs=10;
param.accelerated=false;
param.decreasing=false;
[w logs_exp5a]=mex_svm_svrg(y,X,w0,param); % constant step size regime for 10 epochs
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp5b]=mex_svm_svrg(y,X,w,param);

%%%% Exp 6 - with noise - accelerated SVRG based on 1/3L
param.L=1;
param.accelerated=true;
param.decreasing=false;
param.epochs=10;
[w logs_exp6a]=mex_svm_svrg(y,X,w0,param);
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp6b]=mex_svm_svrg(y,X,w,param);
save('ckn_dropout0.01.mat','logs_exp1','logs_exp2','logs_exp3','logs_exp4a','logs_exp4b','logs_exp5a','logs_exp5b','logs_exp6a','logs_exp6b');

%%%% Exp 4 - with noise - SVRG based on 1/12L
param.dropout=0.1; % Dropout rate
param.L=1;
param.epochs=10;
param.accelerated=false;
param.decreasing=false;
[w logs_exp4a]=mex_svm_svrg(y,X,w0,param); % constant step size regime for 10 epochs
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp4b]=mex_svm_svrg(y,X,w,param);

%%%% Exp 5 - with noise - SVRG based on 1/3L
param.L=0.25;
param.epochs=10;
param.accelerated=false;
param.decreasing=false;
[w logs_exp5a]=mex_svm_svrg(y,X,w0,param); % constant step size regime for 10 epochs
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp5b]=mex_svm_svrg(y,X,w,param);

%%%% Exp 6 - with noise - accelerated SVRG based on 1/3L
param.L=1;
param.accelerated=true;
param.decreasing=false;
param.epochs=10;
[w logs_exp6a]=mex_svm_svrg(y,X,w0,param);
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp6b]=mex_svm_svrg(y,X,w,param);
save('ckn_dropout0.1.mat','logs_exp1','logs_exp2','logs_exp3','logs_exp4a','logs_exp4b','logs_exp5a','logs_exp5b','logs_exp6a','logs_exp6b');

%%%% Exp 4 - with noise - SVRG based on 1/12L
param.dropout=0.3; % Dropout rate
param.L=1;
param.epochs=10;
param.accelerated=false;
param.decreasing=false;
[w logs_exp4a]=mex_svm_svrg(y,X,w0,param); % constant step size regime for 10 epochs
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp4b]=mex_svm_svrg(y,X,w,param);

%%%% Exp 5 - with noise - SVRG based on 1/3L
param.L=0.25;
param.epochs=10;
param.accelerated=false;
param.decreasing=false;
[w logs_exp5a]=mex_svm_svrg(y,X,w0,param); % constant step size regime for 10 epochs
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp5b]=mex_svm_svrg(y,X,w,param);

%%%% Exp 6 - with noise - accelerated SVRG based on 1/3L
param.L=1;
param.accelerated=true;
param.decreasing=false;
param.epochs=10;
[w logs_exp6a]=mex_svm_svrg(y,X,w0,param);
param.decreasing=true;
param.epochs=nepochs-10;
[w logs_exp6b]=mex_svm_svrg(y,X,w,param);
save('ckn_dropout0.3.mat','logs_exp1','logs_exp2','logs_exp3','logs_exp4a','logs_exp4b','logs_exp5a','logs_exp5b','logs_exp6a','logs_exp6b');


