import torch
from torch import nn
import pytorch_warmup as warmup

class ApprBase(object):
    """ Base class for any approach class. It implements some basic functionalities which can be useful during training and setting up the model: 
        - _get_optimizer: general setup keyword-arguments based
        - setup_scheduler: for now step or cosine scheduling 
        - step_scheduler: to call after each training step to advance the schedulling
        - ready_eval/train: putting the network into eval or train mode (important if using Batch Norm)
        - to_device: general function taking a list of (torch) objects and putting them on the device
    """

    def __init__(self, model, device="cpu", args=None):
        self.model=model
        self.device=device
        return

    def _get_optimizer(self, parameters = None, lr=None, **kwargs):
        """General initialization of optimizer. 
        Accepted arguments: optimizer, weight_decay, momentum"""

        optim_name = kwargs.get('optimizer', 'sgd')
        wd=kwargs.get('weight_decay', 0)
        mom=kwargs.get('momentum', 1)

        if optim_name=="sgd":
            opt = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=mom)
        elif optim_name=="adam":
            opt = torch.optim.Adam(parameters, lr = lr, weight_decay=wd)
        else: raise NotImplementedError

        return opt
    
    def setup_scheduler(self, **kwargs):
        """ Initializing scheduler """
        sched_type = kwargs.get('scheduler_type', 'step')
        step_time = kwargs.get('step_time',100) 
        gamma = kwargs.get('discount_factor',0.9) 
        total_steps = kwargs.get('total_steps',2000) 

        if sched_type=="step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_time, gamma=gamma)
        elif sched_type=="cosine_anneal":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-7)
        elif sched_type=="cosine_anneal_warmrest":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=total_steps, T_mult=1, eta_min=1e-7)
        else: raise NotImplementedError
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=10)   

    def scheduler_step(self):
        with self.warmup_scheduler.dampening(): 
            self.scheduler.step()
    
    def to_device(self, **objects):
        new_objects = []
        for object in objects:
            object = object.to(self.device)
            new_objects.append(object)
        return new_objects

    def ready_eval(self):
        """Puts the network in evaluation mode"""
        self.model.eval()

    def ready_train(self):
        """Puts the network in train mode"""
        self.model.train()

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        pass

    def train_epoch(self,t,x,y):
        pass

    def eval(self,t,x,y):
        pass

    def criterion(self,t,output,targets):
        return 0
    
    def ce_crit(self, t, output, targets):
        return 0

