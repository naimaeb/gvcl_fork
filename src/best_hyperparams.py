def get_best_params(approach, experiment):
    #the best hyperparams for all the experiments
    param = None
    epochs = 200
    lr = 0.05
    if experiment == 'mixture':
        if approach in ['ewc', 'ewc-film']:
            param = 5
        if approach == 'hat':
            param = '0.75,400'
        if approach == 'imm-mean':
            param = 0.0001
        if approach == 'imm-mode':
            param = '1'
        if approach == 'lfl':
            param = 0.05
        if approach == 'lwf':
            param = '2,1'
        if approach == 'pathnet':
            param = 20
        
        if approach == 'gvclf':
            param = '0.2,100'
        if approach == 'gvcl':
            param = '0.2, 1000'
        if approach == 'vcl':
            param = '1,1'

        if 'vcl' in approach:
            epochs = 180
            lr = 1e-4

    elif experiment == 'cifar':
        if approach in ['ewc', 'ewc-film']:
            param = 100
        if approach == 'hat':
            param = '0.025,50'
        if approach == 'imm-mean':
            param = 0.0001
        if approach == 'imm-mode':
            param = '1e-5'
        if approach == 'lfl':
            param = 0.05
        if approach == 'lwf':
            param = '2,4'
        if approach == 'pathnet':
            param = 100
        if approach == 'gvclf':
            param = '0.2,100'
        if approach == 'gvcl':
            param = '0.2,1000'
        if approach == 'vcl':
            param = '1,1'


        if 'vcl' in approach:
            epochs = 60
            lr = 1e-3

    elif experiment == 'easy-chasy':
        epochs = 1000
        if approach in ['ewc', 'ewc-film']:
            param = 100
        if approach == 'hat':
            param = '1,10'
        if approach == 'imm-mean':
            param = 0.0005
        if approach == 'imm-mode':
            param = '1e-7'
        if approach == 'lfl':
            param = 0.1
        if approach == 'lwf':
            param = '0.5,4'
        if approach == 'pathnet':
            param = 20

        if approach == 'gvclf':
            param = '0.05,10'
        if approach == 'gvcl':
            param = '0.05,100'
        if approach == 'vcl':
            param = '1,1'

        if 'vcl' in approach:
            epochs = 1500
            lr = 1e-3

    elif experiment == 'hard-chasy':
        epochs = 1000
        if approach in ['ewc', 'ewc-film']:
            param = 500
        if approach == 'hat':
            param = '1,50'
        if approach == 'imm-mean':
            param = '1e-6'
        if approach == 'imm-mode':
            param = '0.1'
        if approach == 'lfl':
            param = 0.1
        if approach == 'lwf':
            param = '0.5,2'
        if approach == 'pathnet':
            param = 200

        if approach == 'gvclf':
            param = '0.05,10'
        if approach == 'gvcl':
            param = '0.05,100'
        if approach == 'vcl':
            param = '1,1'

        if 'vcl' in approach:
            epochs = 1500
            lr = 1e-3

    elif experiment == 'smnist':
        if approach in ['ewc', 'ewc-film']:
#             param = 1 #10000
            param = 10000
        if approach == 'hat':
            param = '0.1,50'
        if approach == 'imm-mean':
            param = 0.0005
        if approach == 'imm-mode':
            param = '0.1'
        if approach == 'lfl':
            param = 0.1
        if approach == 'lwf':
            param = '2,4'
        if approach == 'pathnet':
            param = 10
        if approach == 'gvclf':
            param = '0.1,100'
        if approach == 'gvcl':
            param = '0.1,1'
        if approach == 'vcl':
            param = '1,1'

        if 'vcl' in approach:
            epochs = 100
            lr = 1e-3

    return param, lr, epochs


sweep_params = {
    "cifar":{
        "gvcl": {
            "kl_g": {
                "lamb": 1705.3214827377574,
                "beta": 0.5640718556201451,
                "lr": 0.001,
                "nepochs": 20,
                "momentum": 0.99,
                "weight_decay": 0.0000939837539992802,
                "scheduler_type":'cosine_anneal',
                "lr_schedule":True,
            },
            "re_g": {
                "lamb": 1,
                "beta": 0.1514491405185915,
                "lr": 0.01,
                "q": 1.044502885378613,
                "nepochs": 20,
                "momentum": 0.9,
                "weight_decay": 0.0000612830313767832,
                "scheduler_type":'cosine_anneal',
                "lr_schedule":True,
            },
        }
    },
    "smnist":{
        "gvcl": {
            "kl_g": {
                "lamb": 94614.16139860584,
                "beta": 0.32338442121888156,
                "lr": 0.001,
                "nepochs": 20,
                "momentum": 0.99,
                "weight_decay": 0.00026155973578119684,
                "scheduler_type":'cosine_anneal',
                "lr_schedule":True,
            },
            "re_g": {
                "lamb": 1,
                "beta": 0.21902256095070907,
                "q":1.3438591027351827,
                "lr": 0.01,
                "nepochs": 10,
                "momentum": 0.9,
                "weight_decay": 0.0000530050707793805,
                "scheduler_type":'cosine_anneal',
                "lr_schedule":True,
            },
        }
    },
}