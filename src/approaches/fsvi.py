import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F

from approaches import ApprBase
import utils
import wandb

from . import compute_kl_g, compute_re_g, compute_kl_qg, compute_t_st, sample_student_t

class Appr(ApprBase):
    """ Class implementing S-FSVI approach"""

    def __init__(self,model, device = "cpu", nepochs=100, sbatch=64,lr=0.05, clipgrad=100, lamb = 1, beta = 1, reg_type = 'kl_g', q = 2, v = 1, train_samples = 10 , args=None, **kwargs):
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
        self.v = v #degrees of freedom of t distribution (int)(v = 2/(q-1)-1)
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

        #making sure every dataset has the same # of gradient passes irrespective of dataset size
        if t == 0:
            self.first_train_size = len(xtrain)
            num_epochs_to_train = self.nepochs

            #correction if the task order is permuted (for mixture)
            if 'mixture' == self.exp:
                self.first_train_size = 20600 #size of facescrub
                num_epochs_to_train = int(round(self.nepochs * self.first_train_size/len(xtrain)))
        if t > 0 and self.equalize_epochs:
            num_epochs_to_train = int(round(self.nepochs * self.first_train_size/len(xtrain)))
        
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

            #scale kl term by beta and dataset size
            kl_term = self.beta * self.compute_functional_regularizer(t, images, targets)

            # Forward current model
            outputs=self.model(images, task_labels, self.reg_type, v = self.v, tasks = [t], num_samples = train_samples)
            output=outputs[t]
            #calculate loss for every MC sample
            stacked_targets = targets.repeat([train_samples])
            flattened_output = output.view(-1, output.shape[-1])
            class_loss = F.cross_entropy(flattened_output, stacked_targets, reduction = 'mean')

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

    def compute_functional_regularizer(self, t, x, y):

        # Forward prior model
        self.model.zero_grad()
        outputs_mean_prior=self.model.forward_mean(x, y, self.reg_type, v = self.v, tasks = [t], prior=True)
        output_mean_prior=outputs_mean_prior[t].mean(dim = 0)
        grad_prior = self.compute_grads(output_mean_prior, t, prior=True) 
            
        # Forward current model
        self.model.zero_grad()
        outputs_mean=self.model.forward_mean(x, y, self.reg_type, v = self.v, tasks = [t], prior=False)
        output_mean=outputs_mean[t].mean(dim = 0)
        grad = self.compute_grads(output_mean, t, prior=False) 
    
        prior_var = self.model.collect_all_variances_vector(t, prior=True)
        var = self.model.collect_all_variances_vector(t, prior=False)

        K_p =  torch.matmul(prior_var.unsqueeze(0)*grad_prior.t(),grad_prior) # equation (12) in the paper 
        K_q =  torch.matmul(var.unsqueeze(0)*grad.t(),grad) # equation (13) in the paper


        #scale kl term by beta and dataset size
        return self.get_reg(lamb = self.lamb, reg_type = self.reg_type, q = self.q, v = self.v, K_p = K_p, K_q = K_q, mu_p=output_mean_prior, mu_q=output_mean)

    def get_reg(self, reg_type, K_p, K_q, mu_p, mu_q, lamb=1, q=1, v=1):
        if reg_type == 'kl_g':
            kl_function = compute_kl_g
        elif reg_type == 're_g':
            kl_function = compute_re_g
        elif reg_type == "kl_qg":
            kl_function = compute_kl_qg
        elif reg_type == "t_st":
            kl_function = compute_t_st
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")

        return kl_function(mu_q, K_q, mu_p, K_p, q, v, lamb=lamb)
    
    def compute_grads(self, outputs, task, prior=False): 
        # Ensure output_prior requires gradients
        outputs.requires_grad_(True)

        # Compute the gradients
        gradients = []
        grad_names = []
        grad_output = torch.zeros_like(outputs)
        for i in range(outputs.shape[0]):
            grad_output[i] = 1.0
            outputs.backward(grad_output, retain_graph=True)
            grad = []
            for n,param in self.model.named_parameters():
                if param.grad is not None:
                    grad.append(param.grad.view(-1))
                    grad_names.append(n)
            gradients.append(torch.cat(grad))

        gradients = torch.stack(gradients).t()  # Shape: P x D
        return gradients

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

