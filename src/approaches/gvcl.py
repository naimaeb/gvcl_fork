import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F

from approaches import ApprBase
import utils
import wandb

class Appr(ApprBase):
    """ Class implementing GVCL approach"""

    def __init__(self,model, device = "cpu", nepochs=[100], sbatch=64,lr=0.05, clipgrad=100, lamb = 1, beta = 1, reg_type = 'kl_g', q = 2, v = 1, train_samples = 10 , args=None, **kwargs):
        """
        Extra flags accepted: 
            - optimizer (str) \in ['sgd','adam']
            - weight_decay (float)
            - momentum (floa)
            - scheduler_type (str) \in ['step','cosine_anneal']
            - step_time (int)
            - discount_factor (float)
            - total_steps (int)
        """

        super().__init__(model, device=device)

        self.model_old=None
        self.fisher=None
        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.clipgrad=clipgrad

        self.beta = beta
        self.lamb = lamb
        # print("lambda", self.lamb)
        # print("beta", self.beta)
        # if len(args.parameter)>=1:
        #     params=args.parameter.split(',')
        #     self.beta= float(params[0])
        #     self.lamb= float(params[1]) 

        #terms relating to the type of regularizer that will be used
        self.reg_type = reg_type #construct the 4 possible regularization cases
        self.q = q #degree of renyi divergence (lambda = 1 - q)
        
        if self.reg_type == 't_st_k1' or self.reg_type == 't_st_mf':
            self.v = 2/(self.q-1)-1 #degrees of freedom of t distribution v = 2/(q-1)-1)
        else:
            self.v = v #dof set for t_st, but q_varies
        
        self.train_samples = train_samples

        self.equalize_epochs = True
        self.exp = kwargs.get("experiment", "")
    
        self.extra_arguments=dict(kwargs) # collecting all the extra flags into this dictionary (note: it may be empty)
        # example of extra flags: all optimizer hyperparameters or lr scheduller hyperparameters 
        self.lr_scheduling=kwargs.get("lr_schedule",False)

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        if self.lr_scheduling: self.setup_scheduler(**self.extra_arguments)




    def _get_optimizer(self, parameters = None, lr=None):
        if lr is None: lr=self.lr
        if parameters is None: parameters = self.model.parameters()
        opt = super()._get_optimizer(parameters=parameters, lr=lr, **self.extra_arguments)
        return opt
    


    #todo: implement get optimizer with the diagonal fisher or block version of it
    def train(self,t,xtrain,ytrain,xvalid,yvalid, step=None):

        num_epochs_to_train = self.get_training_epochs(len(xtrain), t)
        print('training for {} epochs'.format(num_epochs_to_train))

        lr=self.lr


        if t != 0:
            #update posterior to prior for everything except the first task
            self.model.add_task_body_params([t-1],keep_grad_mean = True)    


        parameters = self.model.get_task_specific_parameters(t)
        self.optimizer=self._get_optimizer(parameters, lr)
        total_steps = num_epochs_to_train * len(ytrain) // self.sbatch
        print(f"Total number of steps per task {t}: {total_steps}")
        if self.lr_scheduling: 
            print("Scheduling on.")
            self.setup_scheduler(**self.extra_arguments, total_steps=total_steps)
        
        if 'chasy' not in self.exp:
            #join train and validation sets because gvcl/vcl does not use early stopping
            #except for chasy experiments where the validation set is very large compared to the test set
            #this doesn't make a major difference - 1% max
            xtrain = torch.cat([xtrain, xvalid], dim = 0)
            ytrain = torch.cat([ytrain, yvalid], dim = 0)



        # Loop epochs
        for e in range(num_epochs_to_train):
            # Train
            clock0=time.time()
            class_loss, kl_loss, total_loss, train_acc  = self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()
            clock2=time.time()
            #include wandb logging for these terms
            wandb.log({
                'epoch': step+1,
                'class_loss': class_loss,
                'kl_loss': kl_loss,
                'total_loss': total_loss,
                'train_acc': train_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            step=step+1
            print('| Epoch {:3d}, time={:5.1f}ms| Train: class_loss={:.3f}  kl_loss={:.3f} total_loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),class_loss, kl_loss, total_loss,100*train_acc))
        return step



    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        train_samples = self.train_samples
        
        epoch_class_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0
        total_hits = 0

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            
            images = x[b]
            targets = y[b]
        

            task_labels = int(t) * torch.ones_like(targets)

            # Forward current model
            outputs=self.model(images, task_labels, self.reg_type, v = self.v, tasks = [t], num_samples = train_samples)
            output=outputs[t]

            #calculate loss for every MC sample
            stacked_targets = targets.repeat([train_samples])
            flattened_output = output.view(-1, output.shape[-1])
            class_loss = F.cross_entropy(flattened_output, stacked_targets, reduction = 'mean')
            
            #scale kl term by beta and dataset size
            kl_term = self.beta * self.model.get_reg(lamb = self.lamb, reg_type = self.reg_type, q = self.q, v = self.v)/(x.shape[0])
            loss = class_loss + kl_term


            #for calculating the accuracy
            probs = F.softmax(output, dim=2).mean(dim = 0)
            _,pred=probs.max(1)
            hits=(pred==targets).float()
            total_hits+=hits.sum().data.cpu().numpy().item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()
            if self.lr_scheduling: 
                self.scheduler_step()

            epoch_total_loss += loss.detach().data.item()
            epoch_class_loss += class_loss.detach().data.item()
            epoch_kl_loss += kl_term.detach().data.item()

        return epoch_class_loss/i, epoch_kl_loss/i, epoch_total_loss/i, total_hits/x.shape[0]

    def eval(self,t,x,y):
        with torch.no_grad():
            total_loss=0
            total_acc=0
            total_num=0
            self.model.eval()

            r=np.arange(x.size(0))
            r=torch.LongTensor(r).cuda()

            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                '''
                deperecated torch.autograd.Variable
                images=torch.autograd.Variable(x[b],volatile=True)
                targets=torch.autograd.Variable(y[b],volatile=True)
                '''
                images = x[b]
                targets = y[b]

                task_labels = int(t) * torch.ones_like(targets)

                # Forward
                outputs=self.model(images, task_labels, reg_type = self.reg_type, v = self.v, tasks = [t], num_samples = 20)
                output=outputs[t]
                probs = F.softmax(output, dim=2).mean(dim = 0)
                _,pred=probs.max(1)
                hits=(pred==targets).float()

                # Log
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=len(b)

            #not measuring loss for test set, just accuracy, so return -1 for loss
            return -1, total_acc/total_num

    def criterion(self,t,output,targets):
        return 0
    
    def ce_crit(self, t, output, targets):
        return 0

