from networks.gvcl_model_classes import MultiHeadFiLMCNN, MultiHeadCNN, MultiHeadMLP

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
    class Net(MultiHeadCNN):
        def __init__(self, inputsize,taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads, activation_fun='relu') 

class ZenkeNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(32,3), (32,3), 'pool', (64,3), (64,3), 'pool'], [512], heads, film_type = 'scale')

class SMNISTNetNoFiLM:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize, taskcla, width=256, depth=2, activation_fun='relu', **kwargs):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [width]*depth, heads, activation_fun=activation_fun)

class SMNISTNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [], [256,256], heads, film_type = 'scale')

class Toy2DNet:
    class Net(MultiHeadMLP):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((2), [], [32,32], heads, activation_fun='relu')

class AlexNetNoFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(64, 4, 0), 'pool', (128, 3, 0), 'pool', (256, 2, 0), 'pool'], [2048,2048], heads, prior_var = 0.01)

class AlexNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize,taskcla):
            heads = [t[1] for t in taskcla]
            super().__init__((3,32,32), [(64, 4, 0), 'pool', (128, 3, 0), 'pool', (256, 2, 0), 'pool'], [2048,2048], heads, film_type = 'scale', prior_var = 0.01)

class OmniglotNet:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize,taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            super().__init__((1,28,28), [(64, 3),'pool', (64, 3), 'pool', (64, 3), 'pool', (64, 3), 'pool'], [512], heads, prior_var = 1.)

# New model classes for larger datasets

class TinyImageNetNetNoFiLM:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize, taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            # Architecture for 64x64 images: deeper network with more capacity
            super().__init__((3,64,64), [
                (64, 3), (64, 3), 'pool',  # 64x64 -> 32x32
                (128, 3), (128, 3), 'pool',  # 32x32 -> 16x16
                (256, 3), (256, 3), 'pool',  # 16x16 -> 8x8
                (512, 3), (512, 3), 'pool'   # 8x8 -> 4x4
            ], [1024], heads, activation_fun='relu', prior_var=1.0)

class TinyImageNetNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize, taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            # Same architecture as NoFiLM but with FiLM layers
            super().__init__((3,64,64), [
                (64, 3), (64, 3), 'pool',  # 64x64 -> 32x32
                (128, 3), (128, 3), 'pool',  # 32x32 -> 16x16
                (256, 3), (256, 3), 'pool',  # 16x16 -> 8x8
                (512, 3), (512, 3), 'pool'   # 8x8 -> 4x4
            ], [1024], heads, film_type='scale', activation_fun='relu', prior_var=1.0)

class CORe50NetNoFiLM:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize, taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            # Architecture for 128x128 images: deeper network with more capacity
            super().__init__((3,128,128), [
                (64, 3), (64, 3), 'pool',   # 128x128 -> 64x64
                (128, 3), (128, 3), 'pool',  # 64x64 -> 32x32
                (256, 3), (256, 3), 'pool',  # 32x32 -> 16x16
                (512, 3), (512, 3), 'pool',  # 16x16 -> 8x8
                (1024, 3), (1024, 3), 'pool' # 8x8 -> 4x4
            ], [2048], heads, activation_fun='relu', prior_var=1.0)

class CORe50NetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize, taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            # Same architecture as NoFiLM but with FiLM layers
            super().__init__((3,128,128), [
                (64, 3), (64, 3), 'pool',   # 128x128 -> 64x64
                (128, 3), (128, 3), 'pool',  # 64x64 -> 32x32
                (256, 3), (256, 3), 'pool',  # 32x32 -> 16x16
                (512, 3), (512, 3), 'pool',  # 16x16 -> 8x8
                (1024, 3), (1024, 3), 'pool' # 8x8 -> 4x4
            ], [2048], heads, film_type='scale', activation_fun='relu', prior_var=1.0)

class ImageNetNetNoFiLM:
    class Net(MultiHeadCNN):
        def __init__(self, inputsize, taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            # Architecture for 224x224 images: ResNet-inspired architecture
            super().__init__((3,224,224), [
                (64, 7, 3), 'pool',          # 224x224 -> 112x112 (stride 2)
                (64, 3), (64, 3), 'pool',    # 112x112 -> 56x56
                (128, 3), (128, 3), 'pool',  # 56x56 -> 28x28
                (256, 3), (256, 3), 'pool',  # 28x28 -> 14x14
                (512, 3), (512, 3), 'pool',  # 14x14 -> 7x7
                (1024, 3), (1024, 3), 'pool' # 7x7 -> 4x4
            ], [4096], heads, activation_fun='relu', prior_var=1.0)

class ImageNetNetFiLM:
    class Net(MultiHeadFiLMCNN):
        def __init__(self, inputsize, taskcla, **kwargs):
            heads = [t[1] for t in taskcla]
            # Same architecture as NoFiLM but with FiLM layers
            super().__init__((3,224,224), [
                (64, 7, 3), 'pool',          # 224x224 -> 112x112 (stride 2)
                (64, 3), (64, 3), 'pool',    # 112x112 -> 56x56
                (128, 3), (128, 3), 'pool',  # 56x56 -> 28x28
                (256, 3), (256, 3), 'pool',  # 28x28 -> 14x14
                (512, 3), (512, 3), 'pool',  # 14x14 -> 7x7
                (1024, 3), (1024, 3), 'pool' # 7x7 -> 4x4
            ], [4096], heads, film_type='scale', activation_fun='relu', prior_var=1.0)