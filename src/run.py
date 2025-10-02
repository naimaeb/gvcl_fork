import sys,os,argparse,time
import numpy as np
import torch
from best_hyperparams import get_best_params, sweep_params
import random
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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

########################################################################################################################

# Load
print('Load data...')
default_path = root_path+"dat/" #pick this otherwise

# Handle different dataloader parameters for new datasets
if args.experiment == 'tinyimagenet':
    data,taskcla,inputsize=dataloader.get(seed=args.seed, path=default_path, num_tasks=40, classes_per_task=5)
elif args.experiment == 'core50':
    data,taskcla,inputsize=dataloader.get(seed=args.seed, path=default_path, scenario='nc', num_tasks=10)
elif args.experiment in ['imagenet-r', 'imagenet-a']:
    variant = 'r' if args.experiment == 'imagenet-r' else 'a'
    data,taskcla,inputsize=dataloader.get(seed=args.seed, path=default_path, variant=variant, num_tasks=20, classes_per_task=10)
elif args.experiment == 'imagenet-subset':
    data,taskcla,inputsize=dataloader.get(seed=args.seed, path=default_path, variant='subset', num_tasks=50, classes_per_task=20)
else:
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
else: 
    if hasattr(args, 'sweep_name'):
        delattr(args, 'sweep_name')

# Set output path after all parameters are finalized
if args.output == '':
    args.output = args.root_path + 'res/' + args.experiment + '/' + args.approach + '/' + args.reg_type + '/' + str(args.q) + '/' + str(args.seed) + '.txt'
    if not os.path.exists(args.root_path + 'res/' + args.experiment + '/' + args.approach + '/' + args.reg_type + '/' + str(args.q) + '/'):
        os.makedirs(args.root_path + 'res/' + args.experiment + '/' + args.approach + '/' + args.reg_type + '/' + str(args.q) + '/')

# Print final arguments after all modifications
print('='*100)
print('Final Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

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
print("approach.use_deformed_likelihood", appr.use_deformed_likelihood)

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

    output_dict = {}
    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()
        xtrain=data[u]['train']['x'].cuda()
        ytrain = data[u]['train']['y'].cuda()
        if args.approach == 'hat':
            test_loss,test_acc=appr.eval(u,xtest,ytest,save_preds = True, dset = args.experiment)
        if args.approach == 'toy2d':
            test_loss,test_acc, out =appr.eval(u,xtest,ytest,)

            #create a grid to plot probabilities
            xtest_tmp = xtest.cpu().numpy()
            x_min, x_max = xtest_tmp[:, 0].min() - 1, xtest_tmp[:, 0].max() + 1
            y_min, y_max = xtest_tmp[:, 1].min() - 1, xtest_tmp[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                np.arange(y_min, y_max, 0.01))
            grid= np.c_[xx.ravel(), yy.ravel()]
            print("grid", grid.size)
            print("x_test_size", xtest.size())

            # Convert grid to a 2D tensor
            grid_tensor = torch.tensor(grid, dtype=torch.float32).cuda()
            print("grid_tensor", grid_tensor.size())

            _,_, out_grid =appr.eval(u,grid_tensor, None)
            out_grid_plot = out_grid[:,1].reshape(xx.shape)
            output_dict[f'task_{u}'] = {
                'output': out,
                'xtest': xtest.cpu().numpy(),
                'ytest': ytest.cpu().numpy(),
                'xtrain': xtrain.cpu().numpy(),
                'ytrain': ytrain.cpu().numpy(),
                'out_grid': out_grid_plot,
                'grid': grid,
                'xx': xx,
                'yy': yy
            }
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

    # Save output_dict if using toy2d approach
    if args.approach == 'toy2d':
        output_file = args.output.replace('.txt', "q_"+str(appr.q)+'_output_dict.npy')
        np.save(output_file, output_dict)

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

# example command: CUDA_VISIBLE_DEVICES=4 python ./src/run.py --train_samples 3 --use-sweep True --experiment cifar --approach gvcl --seed 42 --reg_type re_g 
# CUDA_VISIBLE_DEVICES=3 python ./src/run.py --nepochs 10 --experiment smnist --train_samples 4 --approach fsvi --seed 14 --reg_type kl_g  --lr 0.01 --sbatch 64 --optimizer sgd --context 40  --beta 0.1
# CUDA_VISIBLE_DEVICES=6 python ./src/run.py --nepochs 200 10 10 10 10 10 10 10 10 10 --experiment omniglot --train_samples 4 --approach gvcl --seed 14 --reg_type kl_g --lr 0.1 --sbatch 64 --optimizer sgd --ntasks 10
# CUDA_VISIBLE_DEVICES=4 python ./src/run.py --experiment smnist --approach ewc --seed 14 
