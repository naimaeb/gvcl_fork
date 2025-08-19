import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F

from approaches import ApprBase
import utils
from utils import compute_u_lambda, log_lambda, kappa_lambda, generalized_elbo
import wandb

class Appr(ApprBase):
    """ Class implementing GVCL approach"""

    def __init__(self,model, device = "cpu", nepochs=[100], sbatch=64,lr=0.05, clipgrad=100, lamb = 1, beta = 1, reg_type = 'kl_g', use_deformed_likelihood = False, q = 2, v = 1, train_samples = 10 , args=None, **kwargs):
        """
        Extra flags accepted: 
            - optimizer (str) \in ['sgd','adam']
            - weight_decay (float)
            - momentum (floa)
            - scheduler_type (str) \in ['step','cosine_anneal']
            - step_time (int)
            - discount_factor (float)
            - total_steps (int)
            - use_deformed_likelihood (bool) - whether to use deformed likelihood for re_g
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
        self.use_deformed_likelihood = kwargs.get("use_deformed_likelihood", False)  # Default to False

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
    


    
    def train(self,t,xtrain,ytrain,xvalid,yvalid, step=None):
        num_epochs_to_train = self.get_training_epochs(len(xtrain), t)
        print('training for {} epochs'.format(num_epochs_to_train))

        lr=self.lr

        if t != 0:
            #update posterior to prior for everything except the first task
            self.model.add_task_body_params([t-1]) 
        '''
        # Log initial variances at the start of training
        init_vars = self.model.get_layer_variances(t, prior=True)
        wandb.log({
            'task': t,
            'epoch': 0,
            **{f'init_{k}': v for k, v in init_vars.items()}
        })
        '''
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
            class_loss, kl_loss, _, _, total_loss, train_acc, avg_grad_norm  = self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()
            clock2=time.time()

            #include wandb logging for these terms
            all_vars = self.model.collect_all_variances_vector(t, prior=False)
            wandb.log({
                'epoch': step+1,
                'class_loss': class_loss,
                'kl_loss': kl_loss,
                'total_loss': total_loss,
                'train_acc': train_acc,
                'lr': self.optimizer.param_groups[0]['lr'],
                'all_var_mean': all_vars.mean().item(),
                'all_var_var': all_vars.var().item(),
                'grad_norm': avg_grad_norm
            })
            
            step=step+1
            print('| Epoch {:3d}, time={:5.1f}ms| Train: class_loss={:.3f}  kl_loss={:.3f} total_loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),class_loss, kl_loss, total_loss,100*train_acc))
        '''
        # Log final variances at the end of training
        final_vars = self.model.get_layer_variances(t, prior=False)
        wandb.log({
            'task': t,
            'epoch': num_epochs_to_train,
            **{f'final_{k}': v for k, v in final_vars.items()}
        })
        '''       
        return step



    def train_epoch(self,t,x,y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        train_samples = self.train_samples
        
        epoch_class_loss = 0
        epoch_kl_loss = 0
        epoch_total_loss = 0
        total_hits = 0
        
        grad_norm_sum = 0.0
        grad_norm_count = 0
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            
            images = x[b]
            targets = y[b]
            task_labels = int(t) * torch.ones_like(targets)
            
            # Forward pass
            outputs = self.model(images, task_labels, self.reg_type, v=self.v, use_deformed_likelihood=self.use_deformed_likelihood, tasks=[t], num_samples=train_samples)
            output = outputs[t]
            
            # Reshape for loss computation
            stacked_targets = targets.repeat([train_samples])
            flattened_output = output.view(-1, output.shape[-1])
            
            # Compute classification loss based on reg_type and whether to use deformed likelihood
            if self.use_deformed_likelihood:
                # Deformed implementation for Renyi divergence
                probs = F.softmax(flattened_output, dim=-1)
                
                # Deformed logarithm: logq(x) = (x^(q) - 1)/q
                deformed_log_probs = (probs**(self.q) - 1)/self.q
                
                # Get target probabilities
                target_probs = F.one_hot(stacked_targets, num_classes=probs.shape[-1])
                
                # Compute expectation of deformed log likelihood
                expectation = (deformed_log_probs * target_probs).sum(dim=-1)
                
                # Kappa function: κλ(t) = log[λt + 1]/λ where λ = 1-q
                lambda_param = 1 - self.q
                class_loss = -torch.log(lambda_param * expectation + 1) / lambda_param
                class_loss = class_loss.mean()
            else:
                # Regular MC sampling for KL divergence
                class_loss = F.cross_entropy(flattened_output, stacked_targets, reduction='mean')
            
            #scale kl term by beta and dataset size
            kl_term, kl_term_mean, kl_term_var = self.model.get_reg(lamb = self.lamb, reg_type = self.reg_type, q = self.q, v = self.v)#/(x.shape[0])
            
            #divide by the size of the distribution, not necessary when get_reg doesn not return a tuple
            kl_term = self.beta*kl_term/(x.shape[0])
            kl_term_mean = self.beta*kl_term_mean/(x.shape[0])
            kl_term_var = self.beta*kl_term_var/(x.shape[0])
            #kl_val += kl_term.detach().data.item()

            '''
            renyi_term, renyi_term_mean, renyi_term_var = self.beta * self.model.get_reg(lamb = self.lamb, reg_type = "re_g", q = self.q, v = self.v)#/(x.shape[0])
            renyi_term = renyi_term/(x.shape[0])
            renyi_term_mean = renyi_term_mean/(x.shape[0])
            renyi_term_var = renyi_term_var/(x.shape[0])
            renyi_val += renyi_term.detach().data.item()
            '''
           
            

            
            #print(f"kl: {kl_val}, renyi: {renyi_val}")

            loss = class_loss + kl_term
            

            #for calculating the accuracy
            probs = F.softmax(output, dim=2).mean(dim = 0)
            _,pred=probs.max(1)
            hits=(pred==targets).float()
            total_hits+=hits.sum().data.cpu().numpy().item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # Compute grad norm for this batch
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            grad_norm_sum += grad_norm
            grad_norm_count += 1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()
            if self.lr_scheduling: 
                self.scheduler_step()

            epoch_total_loss += loss.detach().data.item()
            epoch_class_loss += class_loss.detach().data.item()

            #computes the KL divergence, and logs the parts that correspond to the mean and variance
            epoch_kl_loss += kl_term.detach().data.item()
            epoch_kl_loss_mean = kl_term_mean.detach().data.item()
            epoch_kl_loss_var = kl_term_var.detach().data.item()

            '''
            #computes the Renyi divergence, and logs the parts that correspond to the mean and variance
            epoch_renyi_loss = renyi_term.detach().data.item()
            epoch_renyi_loss_mean = renyi_term_mean.detach().data.item()
            epoch_renyi_loss_var = renyi_term_var.detach().data.item()
            '''
        avg_grad_norm = grad_norm_sum / grad_norm_count if grad_norm_count > 0 else 0.0
        return epoch_class_loss/i, epoch_kl_loss/i, epoch_kl_loss_mean/i, epoch_kl_loss_var/i, epoch_total_loss/i, total_hits/x.shape[0], avg_grad_norm

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
                outputs=self.model(images, task_labels, reg_type = self.reg_type, v = self.v, use_deformed_likelihood=self.use_deformed_likelihood, tasks = [t], num_samples = 20)
                output=outputs[t]
                probs = F.softmax(output, dim=2).mean(dim = 0)
                _,pred=probs.max(1)
                hits=(pred==targets).float()

                # Log
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=len(b)

            #not measuring loss for test set, just accuracy, so return -1 for loss
            return -1, total_acc/x.shape[0]

    def criterion(self,t,output,targets):
        return 0
    
    def ce_crit(self, t, output, targets):
        return 0

