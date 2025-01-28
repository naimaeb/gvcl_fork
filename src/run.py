import sys,os,argparse,time
import numpy as np
import torch
from best_hyperparams import get_best_params, sweep_params

import utils
import setup
import wandb

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

########################################################################################################################

# Load
print('Load data...')
default_path = root_path+"dat/" #pick this otherwise
data,taskcla,inputsize=dataloader.get(seed=args.seed, path=default_path)
if args.ntasks != -1:
    taskcla = taskcla[:args.ntasks]
print('Input size =',inputsize,'\nTask info =',taskcla)

# Inits
print('Inits...')
net=network.Net(inputsize,taskcla).cuda()
utils.print_model_report(net)

#Set hyperparameters
if args.use_best_hyperparams:   
    best_param, best_lr, best_epochs = get_best_params(args.approach, args.experiment)
    args.nepochs = best_epochs
    print("using default # epochs of {}".format(best_epochs))
    args.lr = best_lr
    print("using default lr of {}".format(best_lr))
    args.parameter = best_param
    print("using default hyperparams of {}".format(best_param))

if args.use_sweep: 
    try: 
        print("args_reg",args.reg_type)
        sweep_best = sweep_params[args.experiment][args.approach][args.reg_type]
        use_sweep = True
        for key, value in sweep_best.items():
            setattr(args, key, value)
            print(f"Setting {key} to {value}")
    except Exception as e: 
        print(e)
        print("No sweep hyperparameters found for this approach and experiment. Resorting to default.")
        pass 


wandb.init(config=args, project=wandb_setup['project-name'], entity=wandb_setup['entity'])


#set the reg_type if performing any vcl related approach
if 'vcl' in args.approach:
    print("regularization happening")

print(vars(args))
appr=approach.Appr(net,**vars(args))

print("approach.beta", appr.beta)
print("approach.lamb", appr.lamb)
print("approach.q", appr.q)
print("approach.v", appr.v)
print("approach.reg", appr.reg_type)

print("criterion", appr.criterion)
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
    step = appr.train(task,xtrain,ytrain,xvalid,yvalid, step)
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


    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f')

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

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

wandb.finish()

########################################################################################################################

# example command: CUDA_VISIBLE_DEVICES=1 python ./src/run.py --train_samples 3 --use-sweep True --experiment cifar --approach gvcl --seed 42 --reg_type re_g
# CUDA_VISIBLE_DEVICES=1 python ./src/run.py --nepochs 10 --experiment smnist --train_samples 4 --approach gvcl --seed 14 --reg_type kl_g  --lr 0.1 --sbatch 64 --optimizer sgd 
# CUDA_VISIBLE_DEVICES=6 python ./src/run.py --nepochs 200 10 10 10 10 10 10 10 10 10 --experiment omniglot --train_samples 4 --approach gvcl --seed 14 --reg_type kl_g --lr 0.1 --sbatch 64 --optimizer sgd --ntasks 10
