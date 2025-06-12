import sys,os,argparse,time
import numpy as np
import torch
import wandb
from best_hyperparams import get_best_params

import setup
import utils

wandb_setup = {
    "project-name":'curvy-cl',
    "entity":'nagiu'
}

tstart=time.time()
root_path = './' #change to match running directory

args, network, approach, dataloader = setup.get_args()

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()


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
            'value': args.reg_type 
        },
        'context':{
            'value':20 # for fsvi
        },
        # 'scheduler_type': {
        #      'value': 'cosine_anneal'
        #  },
        'optimizer':{'value':'adam'},
        'lr_schedule': {'value': False},
        'lr': {'values': [1e-4, 1e-3],}, #{"max": 1.0, "min": 1e-4},
        'lamb': {'values': [1, 10, 100, 1000],}, # {"max": 1e3, "min": 10.1},
        #'beta': {'values': [0.01, 0.1, 0.5, 1],},#{"max": 1.0, "min": 1e-3},
        'q': {"max": 3.0 +1e-3, "min": 0.0 +1e-3},#{'values': [1.1, 1.8, 2.4, 2.9],},
        #'v': {"max": 10e4, "min": 1e-4},
        #'weight_decay':{"max": 1e-3, "min": 1e-6}, 
        #'momentum': {'values': [0, 0.9, 0.99, 1]},
        #'activation_fun': {'values': ['relu', 'tanh', 'leaky_relu']},
        #'width': {'values': [128, 256, 512, 780]},
        #'depth': {'values': [1, 2, 3, 4]},
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
    net=network.Net(inputsize,taskcla, **config).cuda()
    utils.print_model_report(net)


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
        avg_accuracy = np.mean(acc[t, :t+1])
        bwt = np.mean(acc[t, :t+1] - acc.diagonal()[:t+1])
        print(f'Average accuracy: {avg_accuracy * 100:.1f}% and BWT: {bwt * 100:.1f}%')
        wandb.log({
            "epoch": step,
            "avg_accuracy": avg_accuracy,
            "bwt": bwt
        })

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
sweep_id = wandb.sweep(wandb_config, project=wandb_setup['project-name'], entity=wandb_setup['entity'])
wandb.agent(sweep_id, function=training_and_testing, project=wandb_setup['project-name'], entity=wandb_setup['entity'])


########################################################################################################################

# example command: CUDA_VISIBLE_DEVICES=0 python3 ./src/sweep.py --sweep_name 'kl_g_gvcl-nofilm-cifar-cosine-v2' --nepochs 20 --experiment cifar --train_samples 4 --approach vcl --seed 14 --reg_type kl_g
# example command: CUDA_VISIBLE_DEVICES=3 python ./src/sweep.py --sweep_name 't_st_k1_vcl-nofilm-mnist-adam' --nepochs 10 --experiment smnist --train_samples 4 --approach vcl --seed 14 --reg_type t_st_k1
# example command: CUDA_VISIBLE_DEVICES=2 python3 ./src/sweep.py --sweep_name 't_st_mf_vcl-nofilm-mnist-adam' --nepochs 10 --experiment smnist --train_samples 4 --approach vcl --seed 14 --reg_type t_st_mf

# CUDA_VISIBLE_DEVICES=7 python ./src/sweep.py --sweep_name 'kl_g_gvcl_cifar-sgd-v2' --nepochs 20 --experiment cifar --train_samples 4 --approach gvcl --seed 14 --reg_type kl_g --ntasks 10
# CUDA_VISIBLE_DEVICES=0 python ./src/sweep.py --sweep_name 'kl_g_vcl_omniglot-adam' --nepochs 20 --experiment omniglot --train_samples 4 --approach gvcl --seed 14 --reg_type kl_g --ntasks 10
# CUDA_VISIBLE_DEVICES=5 python ./src/sweep.py --sweep_name 't_st_mf_gvcl_omniglot-adam-v2' --nepochs 20 --experiment omniglot --train_samples 4 --approach gvcl --seed 14 --reg_type t_st_mf --ntasks 10

# CUDA_VISIBLE_DEVICES=6 python ./src/sweep.py --sweep_name 're_g_vcl_mnist-adam-width' --nepochs 10 --experiment smnist --train_samples 4 --approach gvcl --seed 14 --reg_type re_g --ntasks 10

# CUDA_VISIBLE_DEVICES=0 python3 ./src/sweep.py --sweep_name 'kl_g_vcl-cifar' --nepochs 20 --experiment cifar --train_samples 4 --approach vcl --seed 14 --reg_type kl_g --context 40 
# CUDA_VISIBLE_DEVICES=1 python ./src/sweep.py --sweep_name 're_g_gvcl-smnist-v3' --nepochs 10 --experiment smnist --train_samples 4 --approach gvcl --seed 14 --reg_type re_g --context 40 