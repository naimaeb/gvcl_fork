# Integration Guide: New Datasets for Continual Learning

This guide explains how to use the newly integrated large-scale datasets with the existing GVCL framework.

## 🎯 Overview

Three new datasets have been integrated into the codebase:

1. **TinyImageNet** - 200 classes, 64×64 images
2. **CORe50** - 50 objects, 128×128 images  
3. **ImageNet Variants** - ImageNet-R, ImageNet-A, ImageNet-Subset (224×224 images)

## 🏗️ Architecture Compatibility

### Model Classes Added

New model classes have been created to handle the larger image sizes:

| Dataset | Image Size | Model Classes | Parameters |
|---------|------------|---------------|------------|
| **TinyImageNet** | 64×64 | `TinyImageNetNetNoFiLM`, `TinyImageNetNetFiLM` | ~2M |
| **CORe50** | 128×128 | `CORe50NetNoFiLM`, `CORe50NetFiLM` | ~8M |
| **ImageNet variants** | 224×224 | `ImageNetNetNoFiLM`, `ImageNetNetFiLM` | ~25M |

### Model Architecture Details

#### TinyImageNet Models (64×64)
```python
# Architecture: 4 conv blocks with pooling
(64, 3), (64, 3), 'pool',    # 64×64 → 32×32
(128, 3), (128, 3), 'pool',  # 32×32 → 16×16
(256, 3), (256, 3), 'pool',  # 16×16 → 8×8
(512, 3), (512, 3), 'pool'   # 8×8 → 4×4
FC: [1024]
```

#### CORe50 Models (128×128)
```python
# Architecture: 5 conv blocks with pooling
(64, 3), (64, 3), 'pool',     # 128×128 → 64×64
(128, 3), (128, 3), 'pool',   # 64×64 → 32×32
(256, 3), (256, 3), 'pool',   # 32×32 → 16×16
(512, 3), (512, 3), 'pool',   # 16×16 → 8×8
(1024, 3), (1024, 3), 'pool'  # 8×8 → 4×4
FC: [2048]
```

#### ImageNet Models (224×224)
```python
# Architecture: ResNet-inspired with larger initial conv
(64, 7, 3), 'pool',           # 224×224 → 112×112 (stride 2)
(64, 3), (64, 3), 'pool',     # 112×112 → 56×56
(128, 3), (128, 3), 'pool',   # 56×56 → 28×28
(256, 3), (256, 3), 'pool',   # 28×28 → 14×14
(512, 3), (512, 3), 'pool',   # 14×14 → 7×7
(1024, 3), (1024, 3), 'pool'  # 7×7 → 4×4
FC: [4096]
```

## 🚀 Usage Examples

### Basic Usage

```bash
# TinyImageNet: 40 tasks of 5 classes each
python src/run.py --experiment tinyimagenet --approach gvcl --reg_type kl_g --seed 42

# CORe50: New-class scenario, 10 tasks
python src/run.py --experiment core50 --approach gvcl --reg_type kl_g --seed 42

# ImageNet-R: 20 tasks of 10 classes each
python src/run.py --experiment imagenet-r --approach gvcl --reg_type kl_g --seed 42

# ImageNet-A: 20 tasks of 10 classes each
python src/run.py --experiment imagenet-a --approach gvcl --reg_type kl_g --seed 42

# ImageNet-Subset: 50 tasks of 20 classes each
python src/run.py --experiment imagenet-subset --approach gvcl --reg_type kl_g --seed 42
```

### Advanced Usage with Custom Parameters

```bash
# Custom number of tasks and epochs
python src/run.py \
    --experiment tinyimagenet \
    --approach gvcl \
    --reg_type kl_g \
    --seed 42 \
    --ntasks 20 \
    --nepochs 50 100 \
    --lr 0.001 \
    --sbatch 32

# CORe50 with different scenarios
python src/run.py \
    --experiment core50 \
    --approach gvcl \
    --reg_type kl_g \
    --seed 42 \
    --ntasks 8  # For new-domain scenario
```

### Using Different Approaches

```bash
# VCL with FiLM layers
python src/run.py --experiment tinyimagenet --approach gvclf --reg_type kl_g --seed 42

# FSVI approach
python src/run.py --experiment core50 --approach fsvi --reg_type kl_g --seed 42

# EWC approach
python src/run.py --experiment imagenet-r --approach ewc --seed 42
```

## 📊 Dataset Configurations

### TinyImageNet
- **Default**: 40 tasks × 5 classes = 200 classes
- **Image size**: 3×64×64
- **Download**: Automatic from Stanford CS231n
- **Memory**: ~2GB for binary files

### CORe50
- **Scenarios**:
  - **New-Class (nc)**: 10 tasks × 5 classes = 50 classes
  - **New-Instance (ni)**: 8 tasks × 50 classes = 50 classes
  - **New-Domain (nd)**: 8 tasks × 50 classes = 50 classes
- **Image size**: 3×128×128
- **Download**: Automatic from official repository
- **Memory**: ~4GB for binary files

### ImageNet Variants
- **ImageNet-R**: 20 tasks × 10 classes = 200 classes
- **ImageNet-A**: 20 tasks × 10 classes = 200 classes  
- **ImageNet-Subset**: 50 tasks × 20 classes = 1000 classes
- **Image size**: 3×224×224
- **Download**: Manual (requires ImageNet access)
- **Memory**: ~15GB for binary files

## 🔧 Configuration Options

### Command Line Arguments

All new datasets support the standard arguments:

```bash
--experiment     # Dataset name: tinyimagenet, core50, imagenet-r, imagenet-a, imagenet-subset
--approach       # Learning approach: gvcl, gvclf, fsvi, ewc, etc.
--reg_type       # Regularization type: kl_g, re_g, t_st, etc.
--seed           # Random seed for reproducibility
--ntasks         # Number of tasks to use (overrides default)
--nepochs        # Number of epochs per task
--lr             # Learning rate
--sbatch         # Batch size
--beta           # Regularization strength
--lamb           # Additional regularization parameter
--q              # Deformation parameter
--v              # Number of samples
```

### Dataloader Parameters

The dataloaders accept additional parameters that can be modified in `src/run.py`:

```python
# TinyImageNet
data,taskcla,inputsize=dataloader.get(
    seed=args.seed, 
    path=default_path, 
    num_tasks=40,           # Number of tasks
    classes_per_task=5      # Classes per task
)

# CORe50
data,taskcla,inputsize=dataloader.get(
    seed=args.seed, 
    path=default_path, 
    scenario='nc',          # 'nc', 'ni', 'nd'
    num_tasks=10            # Number of tasks
)

# ImageNet variants
data,taskcla,inputsize=dataloader.get(
    seed=args.seed, 
    path=default_path, 
    variant='r',            # 'r', 'a', 'subset'
    num_tasks=20,           # Number of tasks
    classes_per_task=10     # Classes per task
)
```

## 🧪 Testing

### Run Integration Tests

```bash
# Test all new datasets
python test_integration.py

# Test individual dataloaders
python test_new_dataloaders.py

# Test with small configurations
python src/run.py --experiment tinyimagenet --approach gvcl --reg_type kl_g --seed 42 --ntasks 2
```

### Expected Output

Successful integration should show:
```
✓ Setup successful for TinyImageNet
✓ TinyImageNetNetNoFiLM created successfully
✓ TinyImageNet dataloader works
✓ End-to-end test successful
```

## 💾 Data Management

### Binary File Locations

Processed data is cached as binary files:

```
dat/
├── binary_tinyimagenet/
│   ├── data0trainx.bin
│   ├── data0trainy.bin
│   ├── data0testx.bin
│   ├── data0testy.bin
│   └── tinyimagenet_task_map.json
├── binary_core50_nc/
│   ├── data0trainx.bin
│   ├── data0trainy.bin
│   └── core50_nc_task_map.json
└── binary_imagenet_r/
    ├── data0trainx.bin
    ├── data0trainy.bin
    └── imagenet_r_task_map.json
```

### Regenerating Binary Files

To regenerate binary files (e.g., after changing parameters):

```python
# In your code
data,taskcla,size = dataloader.get(seed=42, path="../dat/", regen=True)
```

Or modify the dataloader call in `src/run.py`:

```python
data,taskcla,inputsize=dataloader.get(
    seed=args.seed, 
    path=default_path, 
    regen=True,  # Force regeneration
    num_tasks=40,
    classes_per_task=5
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce batch size
   --sbatch 16
   
   # Use fewer tasks for testing
   --ntasks 5
   ```

2. **Download Errors**
   ```bash
   # Check internet connection
   # Verify dataset URLs in dataloader files
   # Manual download may be required for ImageNet variants
   ```

3. **CUDA Out of Memory**
   ```bash
   # Reduce model size or batch size
   # Use CPU for testing: modify run.py to use CPU
   ```

4. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install requests tqdm pillow
   ```

### Performance Tips

1. **Use SSD storage** for faster binary file loading
2. **Increase batch size** if memory allows for faster training
3. **Use mixed precision** for larger models (ImageNet variants)
4. **Monitor GPU memory** usage during training

## 📈 Expected Performance

### Training Time Estimates

| Dataset | Tasks | Epochs/Task | GPU | Estimated Time |
|---------|-------|-------------|-----|----------------|
| TinyImageNet | 40 | 50 | V100 | ~8 hours |
| CORe50 | 10 | 50 | V100 | ~4 hours |
| ImageNet-R | 20 | 50 | V100 | ~12 hours |

### Memory Requirements

| Dataset | Model Size | Batch Size | GPU Memory |
|---------|------------|------------|------------|
| TinyImageNet | ~2M params | 32 | ~4GB |
| CORe50 | ~8M params | 16 | ~8GB |
| ImageNet variants | ~25M params | 8 | ~16GB |

## 🔗 Integration with Existing Code

The new datasets are designed as drop-in replacements:

```python
# Before (CIFAR)
from dataloaders import cifar as dataloader
data,taskcla,size = dataloader.get(seed=42, path="../dat/")

# After (TinyImageNet)
from dataloaders import tinyimagenet as dataloader
data,taskcla,size = dataloader.get(seed=42, path="../dat/")

# Same interface, same data structure!
```

## 📚 References

- **TinyImageNet**: https://tiny-imagenet.herokuapp.com/
- **CORe50**: https://vlomonaco.github.io/core50/
- **ImageNet-R**: https://github.com/hendrycks/imagenet-r
- **ImageNet-A**: https://github.com/hendrycks/imagenet-a
- **GVCL Framework**: Original paper and implementation

## 🎉 Getting Started

1. **Test the integration**:
   ```bash
   python test_integration.py
   ```

2. **Run a small experiment**:
   ```bash
   python src/run.py --experiment tinyimagenet --approach gvcl --reg_type kl_g --seed 42 --ntasks 5
   ```

3. **Scale up to full experiments**:
   ```bash
   python src/run.py --experiment tinyimagenet --approach gvcl --reg_type kl_g --seed 42
   ```

The new datasets are now fully integrated and ready for your continual learning research! 🚀
