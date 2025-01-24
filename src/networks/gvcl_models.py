from networks.gvcl_model_classes import MultiHeadFiLMCNN

#note that the default film type is point, so film and no film do the same
class BabyNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,32,32), [(16,3), 'pool', (32,3), 'pool'], [100], heads)

class BabyNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,32,32), [(16,3), 'pool', (32,3), 'pool'], [100], heads, film_type = 'scale')

class ZenkeNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads, film_type='none') 

class ZenkeNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads, film_type = 'scale')

class SMNISTNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [256,256], heads, film_type='none')

class SMNISTNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [256,256], heads, film_type = 'scale')

class AlexNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(64, 4, 0), 'pool', (128, 3, 0), 'pool', (256, 2, 0), 'pool'], [2048,2048], heads, prior_var = 0.01)

class AlexNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(64, 4, 0), 'pool', (128, 3, 0), 'pool', (256, 2, 0), 'pool'], [2048,2048], heads, film_type = 'scale', prior_var = 0.01)

class OmniglotNet:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [(64, 3), 'pool', (64, 3), 'pool', (64, 3), 'pool', (64, 3), 'pool'], [2048], heads, prior_var = 1.0)