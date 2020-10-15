loss=0;
format compact;
nepochs=300;
dropout=0.0;
lambda_factor=10;
init_epochs=0;
settings=1:12;

if dropout==0
   settings=[1 2 3 7 9 11];
else
   settings=[1 3 4 6 9 11 12 13];
end

tabdataset={'ckn-cifar','gene','alpha'};
colors={'b-','b--','b-.','r-','r--','r-.','g-','g--','g-.','c-','c--','c-.','m-','m--','m-.','k-','k--','k.-','y-','y.-','y--','b-.','r-.','g-.','c-.','m-.','k-.','y-.'};
colors={'b-','b--','r-','b-.','b-.','r--','m-','k-','g-','k--','k.-','k-.','c-'};
linesetting={'-','--','-','--','-.','--','-','-','-','-','.-','-','-'};
colors={[0 0 0.9],[0 0 0.9],[0.9 0 0],[0 0 0.9],[0 0 0.9],[0.9 0 0],[0.9 0 0.9],[0 0 0],[0 0.9 0],[0 0 0],[0 0 0],[0 0 0],[0 0.9 0.9]};
names_total={'rand-SVRG 1/12L','rand-SVRG 1/3L','acc-SVRG 1/3L','rand-SVRG-d','random-SVRG decr2','acc-SVRG-d','SGD 1/L','acc-SGD 1/L','SGD-d','acc-SGD-d','SAGA','SAGA-d','acc-mb-SGD-d'};
names=names_total(settings);
names

for dataset = [2]
   m=1;  
   M=0.5;
   clf;
   set(gcf,'position',[1000 500 400 250]);

   %%% find minimum
   for setting = settings
      sum_log=0;
      for seed = 0:100:400
         name=sprintf('exps/exp_data%d_s%d_d%d_l10_n300_0_seed%d_loss%d.mat',dataset,setting,dropout,seed,loss);
         name
         load(name);
         logs_exp=logs_exp(find(logs_exp));
         length(logs_exp)
         sum_log=logs_exp+sum_log;
      end
      sum_log=sum_log/5;
      m=min([m min(sum_log)]);
      m
      if (dropout ~= 0)
         sum_log=0;
         for seed = 0:100:400
            name=sprintf('exps/exp_data%d_s%d_d%d_l10_n1000_0_seed%d_loss%d.mat',dataset,2,dropout,seed,loss);
            name
            load(name);
            logs_exp=logs_exp(find(logs_exp));
            sum_log=logs_exp+sum_log;
         end
         sum_log=sum_log/5;
         m=min([m min(sum_log)]);
      end
   end

   for setting = settings
      sum_log=0;
      for seed = 0:100:400
         name=sprintf('exps/exp_data%d_s%d_d%d_l10_n300_0_seed%d_loss%d.mat',dataset,setting,dropout,seed,loss);
         load(name);
         sum_log=logs_exp+sum_log;
      end
      sum_log=sum_log/5;
      name
      if dropout
         subset=0:5:nepochs-1;
         sum_log=sum_log(find(sum_log));
         if setting <= 6
            subset=0:10:2*(nepochs-1);
         end
      else
         subset=0:nepochs-1;
         if setting <= 6
            subset=0:2:2*(nepochs-1);
         end
      end
      %   m=1.000001*m;
      %   sum_log'
      semilogy(subset,(sum_log/m-1),linesetting{setting},'Color',colors{setting},'LineWidth',2); hold on;
      axis([0 300 10^-5 1]);
   end

   set(gca,'YTick',10.^(-5:1:0)); 
   set(gca,'YMinorTick','off');  
   set(gca,'XTick',0:50:300); 
   set(gca,'FontSize',12);
   xlabel('Effective passes over data');
   ylabel('log(F/F^*-1)');

   h=legend(names);
   set(h,'FontSize',9);
   get(h,'position')
   if (dropout==0.0)
      set(h,'position',[ 0.6078    0.0800    0.3620    0.4550]);
   end
   if (dropout==0.01)
      set(h,'position',[0.6244    0.5965    0.3538    0.3925]);
   end
   if (dropout==0.1)
      set(h,'position',[  0.5769    0.0885    0.3538    0.3925]);
   end

   set(gcf,'PaperPositionMode','Auto');
   sprintf('figures/%s_%d_%d_%d.eps',tabdataset{dataset},lambda_factor,dropout,loss)
   print('-depsc2',sprintf('figures/%s_%d_%d_%d.eps',tabdataset{dataset},lambda_factor,dropout,loss));
end
