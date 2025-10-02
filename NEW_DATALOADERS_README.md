# New Dataloaders for Continual Learning

This document describes the new dataloaders that have been integrated into the codebase to support larger datasets for continual learning experiments.

## Overview

Three new dataloaders have been added to support larger-scale continual learning experiments:

1. **TinyImageNet** - 200 classes, configurable task splitting
2. **CORe50** - 50 objects with multiple continual learning scenarios
3. **ImageNet Variants** - ImageNet-R, ImageNet-A, and ImageNet subsets

## Dataloaders

### 1. TinyImageNet Dataloader

**File**: `src/dataloaders/tinyimagenet.py`

**Dataset**: TinyImageNet (200 classes, 64x64 images)

**Default Configuration**: 40 tasks of 5 classes each

**Usage**:
```python
from dataloaders.tinyimagenet import get

# Default: 40 tasks, 5 classes per task
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    num_tasks=40,
    classes_per_task=5
)

# Custom configuration: 20 tasks, 10 classes per task
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    num_tasks=20,
    classes_per_task=10
)
```

**Features**:
- Automatic download from Stanford CS231n
- Configurable task splitting
- Binary file caching for faster loading
- Standard ImageNet normalization

### 2. CORe50 Dataloader

**File**: `src/dataloaders/core50.py`

**Dataset**: CORe50 (50 objects, 128x128 images)

**Scenarios**:
- **New-Class (nc)**: Each session introduces new classes
- **New-Instance (ni)**: Each session introduces new instances of same classes
- **New-Domain (nd)**: Each session introduces new domains/conditions

**Usage**:
```python
from dataloaders.core50 import get

# New-Class scenario (default)
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    scenario='nc',  # 'nc', 'ni', or 'nd'
    num_tasks=10
)

# New-Instance scenario
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    scenario='ni',
    num_tasks=8  # 8 conditions
)

# New-Domain scenario
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    scenario='nd',
    num_tasks=8  # 8 conditions
)
```

**Features**:
- Three different continual learning scenarios
- Automatic download from official CORe50 repository
- Session-based task organization
- Binary file caching

### 3. ImageNet Variants Dataloader

**File**: `src/dataloaders/imagenet.py`

**Datasets**:
- **ImageNet Subset**: Subset of ImageNet classes (1000 classes → configurable tasks)
- **ImageNet-R**: Robustness variant (200 classes)
- **ImageNet-A**: Adversarial variant (200 classes)

**Usage**:
```python
from dataloaders.imagenet import get

# ImageNet subset: 50 tasks of 20 classes each
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    variant='subset',
    num_tasks=50,
    classes_per_task=20
)

# ImageNet-R: 20 tasks of 10 classes each
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    variant='r',
    num_tasks=20,
    classes_per_task=10
)

# ImageNet-A: 20 tasks of 10 classes each
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=False,
    variant='a',
    num_tasks=20,
    classes_per_task=10
)
```

**Features**:
- Support for multiple ImageNet variants
- Configurable task splitting
- Standard ImageNet preprocessing (224x224, ImageNet normalization)
- Binary file caching

## Common Parameters

All dataloaders share these common parameters:

- `seed`: Random seed for reproducibility
- `pc_valid`: Percentage of training data to use for validation (default: 0.10)
- `path`: Path to data directory (default: "../dat/")
- `regen`: Whether to regenerate binary files (default: False)

## Data Structure

All dataloaders return the same data structure:

```python
data, taskcla, size = get(...)

# data: Dictionary with data for each task
# data[task_id]['train']['x']: Training images
# data[task_id]['train']['y']: Training labels
# data[task_id]['test']['x']: Test images
# data[task_id]['test']['y']: Test labels
# data[task_id]['valid']['x']: Validation images
# data[task_id]['valid']['y']: Validation labels
# data[task_id]['ncla']: Number of classes in this task
# data[task_id]['name']: Task name

# taskcla: List of (task_id, num_classes) tuples
# size: Image size [channels, height, width]
```

## Installation and Setup

### Dependencies

The new dataloaders require these additional dependencies:

```bash
pip install requests tqdm pillow
```

### Dataset Downloads

1. **TinyImageNet**: Automatically downloaded from Stanford CS231n
2. **CORe50**: Automatically downloaded from official repository
3. **ImageNet variants**: Manual download required (see below)

### Manual Dataset Setup

For ImageNet variants, you need to manually download and organize the datasets:

```bash
# Create data directory
mkdir -p ../dat/

# Download ImageNet-R (example)
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xzf imagenet-r.tar -C ../dat/
mv ../dat/imagenet-r ../dat/imagenet-r

# Download ImageNet-A (example)
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xzf imagenet-a.tar -C ../dat/
mv ../dat/imagenet-a ../dat/imagenet-a
```

## Testing

Run the test script to verify all dataloaders work correctly:

```bash
python test_new_dataloaders.py
```

This will test all three dataloaders and provide a summary of results.

## Integration with Existing Code

The new dataloaders follow the same interface as existing dataloaders (CIFAR, Omniglot), so they can be used as drop-in replacements:

```python
# Replace existing dataloader import
# from dataloaders.cifar import get
from dataloaders.tinyimagenet import get

# Use the same way
data, taskcla, size = get(seed=42, path="../dat/")
```

## Performance Considerations

- **Binary caching**: All dataloaders cache processed data as binary files for faster subsequent loading
- **Memory usage**: Large datasets may require significant RAM
- **Disk space**: Binary files can be large (several GB for full datasets)
- **First run**: Initial processing may take several minutes

## Troubleshooting

### Common Issues

1. **Download errors**: Check internet connection and dataset URLs
2. **Memory errors**: Reduce batch size or use smaller dataset subsets
3. **Disk space**: Ensure sufficient space for binary files
4. **Import errors**: Check that all dependencies are installed

### Regenerating Binary Files

To regenerate binary files (e.g., after changing parameters):

```python
data, taskcla, size = get(
    seed=42,
    path="../dat/",
    regen=True  # Force regeneration
)
```

## Examples

### Example 1: TinyImageNet with 20 tasks
```python
from dataloaders.tinyimagenet import get

data, taskcla, size = get(
    seed=42,
    num_tasks=20,
    classes_per_task=10  # 200 classes / 20 tasks = 10 classes per task
)

print(f"Dataset size: {size}")
print(f"Number of tasks: {len(taskcla)}")
print(f"Total classes: {data['ncla']}")
```

### Example 2: CORe50 New-Domain scenario
```python
from dataloaders.core50 import get

data, taskcla, size = get(
    seed=42,
    scenario='nd',  # New-domain scenario
    num_tasks=8     # 8 different conditions
)

print(f"CORe50 New-Domain: {len(taskcla)} tasks")
```

### Example 3: ImageNet-R with custom splitting
```python
from dataloaders.imagenet import get

data, taskcla, size = get(
    seed=42,
    variant='r',
    num_tasks=25,
    classes_per_task=8  # 200 classes / 25 tasks = 8 classes per task
)

print(f"ImageNet-R: {len(taskcla)} tasks")
```

## Contributing

To add new dataloaders:

1. Follow the same interface as existing dataloaders
2. Implement binary file caching
3. Add proper error handling
4. Include documentation
5. Add tests to the test script

## References

- **TinyImageNet**: https://tiny-imagenet.herokuapp.com/
- **CORe50**: https://vlomonaco.github.io/core50/
- **ImageNet-R**: https://github.com/hendrycks/imagenet-r
- **ImageNet-A**: https://github.com/hendrycks/imagenet-a
