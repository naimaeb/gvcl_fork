from networks.fsvi_model_classes import MultiHeadCNN 

class ZenkeNetNoFiLM:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads, activation_fun='relu', prior_var=0.01) 

class SMNISTNetNoFiLM:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [256,256], heads, activation_fun='relu', prior_var=0.1)

class OmniglotNet:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [(64, 3),'pool', (64, 3), 'pool', (64, 3), 'pool', (64, 3), 'pool'], [512], heads, prior_var = 0.01)