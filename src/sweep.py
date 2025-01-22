import sys,os,argparse,time
import numpy as np
import torch
import wandb
from best_hyperparams import get_best_params

import utils

wandb_setup = {
    "project-name":'curvy-cl',
    "entity":'nagiu'
}

tstart=time.time()

root_path = './' #change to match running directory
# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--sweep_name',type=str,help='sweep-id')
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--experiment',default='',type=str,required=True,choices=['mnist2','pmnist','cifar','mixture', 'easy-chasy', 'hard-chasy', 'smnist'],help='(default=%(default)s)')
parser.add_argument('--approach',default='',type=str,required=True,choices=['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                            'imm-mode','sgd-restart', 'ewc2', 'ewc-film',
                                                                            'joint','hat','hat-test', 'gvcl', 'vcl', 'vclf', 'gvclf'],help='(default=%(default)s)')
parser.add_argument('--regularizer', default='',type=str,required=False,choices=['t_st', 'kl_g','re_g','kl_qg','re_qg'],help='(default=%(default)s)') 
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=-1,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--ntasks',type=int,default=-1,help='(default=%(default)s)')
parser.add_argument('--use-best-hyperparams',type=bool,default=True,help='(default=%(default)s)')
parser.add_argument('--train_samples', type=int, default=10, help='Number of samples from the posterior')
args=parser.parse_args()
if args.output=='':
    args.output=root_path+'res/'+args.experiment+'_'+args.approach+'_'+args.regularizer+'_'+str(args.seed)+'.txt' #change to parent or current directory depending if you run a test notebook or the run.py script directly
print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader
elif args.experiment=='easy-chasy':
    from dataloaders import easy_chasy as dataloader
elif args.experiment=='hard-chasy':
    from dataloaders import hard_chasy as dataloader
elif args.experiment=='smnist':
    from dataloaders import smnist as dataloader

# Args -- Approach
if args.approach=='random':
    from approaches import random as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach=='sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach=='lwf':
    from approaches import lwf as approach
elif args.approach=='lfl':
    from approaches import lfl as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='ewc-film':
    from approaches import ewc_film as approach
elif args.approach=='ewc2':
    from approaches import ewc2 as approach
elif args.approach=='imm-mean':
    from approaches import imm_mean as approach
elif args.approach=='imm-mode':
    from approaches import imm_mode as approach
elif args.approach=='progressive':
    from approaches import progressive as approach
elif args.approach=='pathnet':
    from approaches import pathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat as approach
elif 'vcl' in args.approach:
    print("approach VCL True")
    from approaches import gvcl as approach
elif args.approach=='joint':
    from approaches import joint as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist':
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    else:
        from networks import mlp as network
elif args.experiment == 'mixture':
    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='ewc-film':
        from networks import alexnet_ewc_film as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
    elif 'vclf' in args.approach:
        print("using vclf model")
        from networks.gvcl_models import AlexNetFiLM as network
    elif 'vcl' in args.approach:
        print("using vcl model")
        from networks.gvcl_models import AlexNetNoFiLM as network
    else:
        from networks import alexnet as network
        
elif args.experiment == 'cifar':
    if args.approach=='lfl':
        from networks import zenkenet_lfl as network
    elif args.approach=='hat':
        from networks import zenkenet_hat as network
    elif args.approach=='progressive':
        from networks import zenkenet_progressive as network
    elif args.approach=='pathnet':
        from networks import zenkenet_pathnet as network
    elif args.approach=='hat-test':
        from networks import zenkenet_hat_test as network
    elif args.approach=='ewc-film':
        from networks import zenkenet_ewc_film as network
    elif 'vclf' in args.approach:
        from networks.gvcl_models import ZenkeNetFiLM as network
    elif 'vcl' in args.approach:
        from networks.gvcl_models import ZenkeNetNoFiLM as network
    else:   
        from networks import zenkenet as network

elif 'chasy' in args.experiment:
    if args.approach=='lfl':
        from networks import babynet_lfl as network
    elif args.approach=='hat':
        from networks import babynet_hat as network
    elif args.approach=='progressive':
        from networks import babynet_progressive as network
    elif args.approach=='pathnet':
        from networks import babynet_pathnet as network
    elif args.approach=='ewc-film':
        from networks import babynet_ewc_film as network
    elif 'vclf' in args.approach:
        from networks.gvcl_models import BabyNetFiLM as network
    elif 'vcl' in args.approach:
        from networks.gvcl_models import BabyNetNoFiLM as network
    else:
        from networks import babynet as network

elif 'smnist' == args.experiment:
    if args.approach=='lfl':
        from networks import smnistnet_lfl as network
    elif args.approach=='hat':
        from networks import smnistnet_hat as network
    elif args.approach=='progressive':
        from networks import smnistnet_progressive as network
    elif args.approach=='pathnet':
        from networks import smnistnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import smnistnet_hat_test as network
    elif args.approach=='ewc-film':
        from networks import smnistnet_ewc_film as network
    elif args.approach=='ewc2':
        from networks import smnistnet_binary as network
    elif 'vclf' in args.approach:
        print("using vclf model smnist")
        from networks.gvcl_models import SMNISTNetFiLM as network
        #print("Film_type", network.get_film_type())
    elif 'vcl' in args.approach:
        print("using vcl model smnist")
        from networks.gvcl_models import SMNISTNetNoFiLM as network
       # print("Film_type", network.get_film_type())
    
    else:
        from networks import smnistnet as network


########################################################################################################################

wandb_config = {
    "name": args.sweep_name,
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "avg_accuracy"},
    "early_terminate":{
        "type": "hyperband",
        "min_iter": 25,
        "eta":2
    },
    'parameters': {
        'seed': {
            'value': args.seed
        },
        'train_samples': {
            'value': args.train_samples
        },
        'ntasks': {
            'value': args.ntasks
        },
        'approach': {
            'value': args.approach
        },
        'experiment': {
            'value': args.experiment
        },
        'nepochs': {
            'value': args.nepochs
        },
        'reg_type': {
            'value': args.regularizer 
        },
        'scheduler_type': {
            'value': 'cosine_anneal'
        },
        'lr_schedule': {'value': True},
        'lr': {'values': [1e-3, 1e-2, 1e-1, 1],},
        #'lamb': {"max": 1e4, "min": 1e-3},
        'beta': {"max": 1.0, "min": 1e-3},
        #'q': {"max": 3.0-1e-4, "min": 1.0+1e-4},
        'v': {"max": 50, "min": 1, "distribution": "int_uniform"},
        'weight_decay':{"max": 1e-3, "min": 1e-6},
        'momentum': {'values': [0, 0.9, 0.99, 1]},
    }
}

def training_and_testing(config=None):

    wandb.init(config=config)
    config = wandb.config

    # Load
    print('Load data...')
    default_path = root_path+"dat/" #pick this otherwise
    data,taskcla,inputsize=dataloader.get(seed=config.seed, path=default_path)
    if config.ntasks != -1:
        taskcla = taskcla[:config.ntasks]
    print('Input size =',inputsize,'\nTask info =',taskcla)

    # Inits
    print('Inits...')
    net=network.Net(inputsize,taskcla).cuda()
    utils.print_model_report(net)

    #Set hyperparameters
    config.total_steps = config.nepochs * len(data[0]['train']['y'])
    print(f"Total number of steps: {config.total_steps}")


    #set the regularizer if performing any vcl related approach
    if 'vcl' in config.approach:
        print("regularization happening")
    
    appr=approach.Appr(net, **config)
    
    # this will print a lot of stuff - turn it off to have less printing
    if True:
        print("Approach parameters:")
        for attr in dir(appr):
            if not attr.startswith('__'):
                print(f"approach.{attr} = {getattr(appr, attr)}")
        utils.print_optimizer_config(appr.optimizer)
        print('-'*100)


    # Loop taskki,l
    acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    step=0
    for t,ncla in taskcla:
        print('*'*100)
        print('Task {:2d} ({:s})'.format(t,data[t]['name']))
        print('*'*100)

        if args.approach == 'joint':
            # Get data. We do not put it to GPU
            if t==0:
                xtrain=data[t]['train']['x']
                ytrain=data[t]['train']['y']
                xvalid=data[t]['valid']['x']
                yvalid=data[t]['valid']['y']
                task_t=t*torch.ones(xtrain.size(0)).int()
                task_v=t*torch.ones(xvalid.size(0)).int()
                task=[task_t,task_v]
            else:
                xtrain=torch.cat((xtrain,data[t]['train']['x']))
                ytrain=torch.cat((ytrain,data[t]['train']['y']))
                xvalid=torch.cat((xvalid,data[t]['valid']['x']))
                yvalid=torch.cat((yvalid,data[t]['valid']['y']))
                task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
                task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
                task=[task_t,task_v]
        else:
            # Get data
            xtrain=data[t]['train']['x'].cuda()
            ytrain=data[t]['train']['y'].cuda()
            xvalid=data[t]['valid']['x'].cuda()
            yvalid=data[t]['valid']['y'].cuda()
            task=t

        # Train
        step=appr.train(task,xtrain,ytrain,xvalid,yvalid,step)
        print('-'*100)

        # Test
        for u in range(t+1):
            xtest=data[u]['test']['x'].cuda()
            ytest=data[u]['test']['y'].cuda()
            if args.approach == 'hat':
                test_loss,test_acc=appr.eval(u,xtest,ytest,save_preds = True, dset = args.experiment)
            else:
                test_loss,test_acc=appr.eval(u,xtest,ytest,)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            acc[t,u]=test_acc 
            lss[t,u]=test_loss
            wandb.log({
                'epoch': step,
                f"test_loss_task_{u}": test_loss,
                f"test_acc_task_{u}": test_acc
            })
        avg_accuracy = np.mean(acc[t, :])
        print(f'Average accuracy: {avg_accuracy * 100:.1f}%')
        wandb.log({"epoch":step, "avg_accuracy": avg_accuracy})

    # Done
    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100*acc[i,j]),end='')
        print()
    print('*'*100)
    print('Done!')
    print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


########################################################################################################################


# wandb sweep training
sweep_id = "uxj50na5"#wandb.sweep(wandb_config, project=wandb_setup['project-name'], entity=wandb_setup['entity'])
wandb.agent(sweep_id, function=training_and_testing, count=20, project=wandb_setup['project-name'], entity=wandb_setup['entity'])


########################################################################################################################

# example command: CUDA_VISIBLE_DEVICES=6 python ./src/sweep.py --sweep_name 'kl_g_gvcl-nofilm-max-mnist-2' --nepochs 10 --experiment smnist --train_samples 4 --approach gvcl --seed 14 --regularizer kl_g
# example command: CUDA_VISIBLE_DEVICES=1 python ./src/sweep.py --sweep_name 'kl_g_gvcl-nofilm-max-cifar' --nepochs 20 --experiment cifar --train_samples 4 --approach gvcl --seed 14 --regularizer kl_g
# example command: CUDA_VISIBLE_DEVICES=0 python ./src/sweep.py --sweep_name 't_st_gvcl-nofilm-max-mnist' --nepochs 10 --experiment smnist --train_samples 4 --approach gvcl --seed 14 --regularizer t_st
# example command: CUDA_VISIBLE_DEVICES=2 python ./src/sweep.py --sweep_name 't_st_gvcl-nofilm-max-cifar' --nepochs 20 --experiment cifar --train_samples 4 --approach gvcl --seed 14 --regularizer t_st
