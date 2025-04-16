"Parsing command line arguments and setting up the experiment"

import sys,os,argparse,time


def get_args():

    tstart=time.time()

    root_path = './' #change to match running directory
    # Arguments
    parser=argparse.ArgumentParser(description='xxx')
    parser.add_argument('--sweep_name',type=str,help='sweep-id')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--experiment',default='',type=str,required=True,choices=['mnist2','pmnist','cifar','mixture', 'easy-chasy', 'hard-chasy', 'smnist','omniglot','toy2d'],help='(default=%(default)s)')
    parser.add_argument('--approach',default='',type=str,required=True,choices=['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                                'imm-mode','sgd-restart', 'ewc2', 'ewc-film', 'fsvi', 'toy2d',
                                                                                'joint','hat','hat-test', 'gvcl', 'vcl', 'vclf', 'gvclf'],help='(default=%(default)s)')
    parser.add_argument('--reg_type', default='',type=str,required=False,choices=['t_st', 't_st_mf', 't_st_k1', 'kl_g','re_g','kl_qg','re_qg'],help='(default=%(default)s)') 
    parser.add_argument('--q', default = 1.01, type=float, required=False)
    parser.add_argument('--v',type=int,default=1,help='(default=%(default)f)')
    parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
    parser.add_argument('--nepochs', nargs='+', type=int, required=False, help='List of epochs for each task')
    parser.add_argument('--lr',default=-1,type=float,required=False,help='(default=%(default)f)')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--ntasks',type=int,default=-1,help='(default=%(default)s)')
    parser.add_argument('--context',type=int, help='Number of context points for FSVI')
    parser.add_argument('--momentum',type=float,default=0.9,help='(default=%(default)f)')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='(default=%(default)f)')
    parser.add_argument('--beta',type=float,default=1,help='(default=%(default)f)')
    parser.add_argument('--lamb',type=float,default=1,help='(default=%(default)f)')
    parser.add_argument('--root_path',type=str,default='./',help='(default=%(default)s)')
    parser.add_argument('--use-best-hyperparams',type=bool,default=False,help='(default=%(default)s)')
    parser.add_argument('--use-sweep',type=bool,default=False,help='(default=%(default)s)')
    parser.add_argument('--train_samples', type=int, default=10, help='Number of samples from the posterior')
    parser.add_argument('--lr_schedule', type=bool, default=False, help='Whether to use a learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='cosine_anneal', help='Type of learning rate scheduler') 
    parser.add_argument('--sbatch', type=int, default=64, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer')
    args=parser.parse_args()
    if args.output=='':
        args.output=root_path+'res/'+args.experiment+'/'+args.approach+'/'+args.reg_type+'/'+str(args.q)+'/'+str(args.seed)+'.txt' #change to parent or current directory depending if you run a test notebook or the run.py script directly
        if not os.path.exists(root_path+'res/'+args.experiment+'/'+args.approach+'/'+args.reg_type+'/'+str(args.q)+'/'): 
            os.makedirs(root_path+'res/'+args.experiment+'/'+args.approach+'/'+args.reg_type+'/'+str(args.q)+'/')
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)
        
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
    elif args.experiment=='omniglot':
        from dataloaders import omniglot as dataloader
    elif args.experiment=='toy2d':
        from dataloaders import toy_data as dataloader

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
    elif args.approach=='fsvi':
        from approaches import fsvi as approach
    elif args.approach=='toy2d':
        from approaches import toy2d as approach

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
        elif 'fsvi'in args.approach:
            from networks.fsvi_models import ZenkeNetNoFiLM as network
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
        elif 'vcl' or 'fsvi' in args.approach:
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
        elif 'fsvi'in args.approach:
            from networks.fsvi_models import SMNISTNetNoFiLM as network
        else:
            from networks import smnistnet as network

    elif 'omniglot' == args.experiment:
        if 'vcl' in args.approach:
            print("using vcl model smnist")
            from networks.gvcl_models import OmniglotNet as network
        # print("Film_type", network.get_film_type())
        elif 'fsvi'in args.approach:
            from networks.fsvi_models import OmniglotNet as network
   
    elif 'toy2d' == args.experiment:
        from networks.gvcl_models import Toy2DNet as network
            

    return args, network, approach, dataloader