import argparse
import wandb
import sys
import os
import time
import numpy as np
import torch

import setup
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Sweep Agent')

    # Static arguments (the same across all runs)
    parser.add_argument('--sweep_id', type=str, required=True, help='Sweep ID to join')
    parser.add_argument('--entity', type=str, required=True, help='WandB entity')
    parser.add_argument('--project', type=str, required=True, help='WandB project')

    # Additional required arguments
    parser.add_argument('--experiment', type=str, required=True,
                        choices=["mnist2", "pmnist", "cifar", "mixture", "easy-chasy", "hard-chasy", "smnist", "omniglot", "toy2d"])
    parser.add_argument('--approach', type=str, required=True,
                        choices=["random", "sgd", "sgd-frozen", "lwf", "lfl", "ewc", "imm-mean", "progressive", "pathnet",
                                 "imm-mode", "sgd-restart", "ewc2", "ewc-film", "fsvi", "toy2d", "joint",
                                 "hat", "hat-test", "gvcl", "vcl", "vclf", "gvclf"])
    parser.add_argument('--reg_type', type=str, default=None,
                        choices=["t_st", "t_st_mf", "t_st_k1", "kl_g", "re_g", "kl_qg", "re_qg"])
    return parser.parse_args()

def training_and_testing(config=None):
    wandb.init(config=config)
    config = wandb.config
    print('Starting training and testing with config:', dict(config))

    # Load data
    print('Loading data...')
    root_path = './'
    default_path = os.path.join(root_path, "dat")

    # Combine sweep config and static args
    arg_list = [
        '--experiment', config.experiment,
        '--approach', config.approach,
    ]
    if getattr(config, "reg_type", None) is not None:
        arg_list += ['--reg_type', config.reg_type]

    args, network, approach, dataloader = setup.get_args(arg_list)

    data, taskcla, inputsize = dataloader.get(seed=config.seed, path=default_path)
    if config.ntasks != -1:
        taskcla = taskcla[:config.ntasks]
    print('Input size =', inputsize, '\nTask info =', taskcla)

    # Inits
    net = network.Net(inputsize, taskcla, **config).cuda()
    utils.print_model_report(net)
    appr = approach.Appr(net, **config)

    # Training Loop
    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    step = 0

    for t, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        if config.approach == 'joint':
            # Concatenate datasets
            if t == 0:
                xtrain, ytrain, xvalid, yvalid = data[t]['train']['x'], data[t]['train']['y'], data[t]['valid']['x'], data[t]['valid']['y']
                task_t = t * torch.ones(xtrain.size(0)).int()
                task_v = t * torch.ones(xvalid.size(0)).int()
            else:
                xtrain = torch.cat((xtrain, data[t]['train']['x']))
                ytrain = torch.cat((ytrain, data[t]['train']['y']))
                xvalid = torch.cat((xvalid, data[t]['valid']['x']))
                yvalid = torch.cat((yvalid, data[t]['valid']['y']))
                task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['x'].size(0)).int()))
                task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['x'].size(0)).int()))
            task = [task_t, task_v]
        else:
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task = t

        step = appr.train(task, xtrain, ytrain, xvalid, yvalid, step)

        # Evaluate
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))

            acc[t, u] = test_acc
            lss[t, u] = test_loss

            wandb.log({
                'epoch': step,
                f"test_loss_task_{u}": test_loss,
                f"test_acc_task_{u}": test_acc
            })

        avg_accuracy = np.mean(acc[t, :])
        print(f'Average accuracy: {avg_accuracy * 100:.1f}%')
        wandb.log({"epoch": step, "avg_accuracy": avg_accuracy})

    # Final reporting
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

def main():
    args = parse_args()
    wandb.agent(args.sweep_id, function=training_and_testing, project=args.project, entity=args.entity,
                config={
                    'experiment': args.experiment,
                    'approach': args.approach,
                    'reg_type': args.reg_type,
                })

if __name__ == "__main__":
    main()
