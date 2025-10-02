"""
Utilities for processing and loading data for Sequential iNaturalist with 50 tasks.
"""

import os
import torch
import numpy as np
from torchvision.datasets import INaturalist
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from typing import Tuple, Dict, List
from tqdm import tqdm
from collections import defaultdict

# Image normalization (ImageNet stats)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_size = (224, 224)


def get(
    path: str = "./dat/",
    num_tasks: int = 50,
    train_p: float = 0.6,
    seed: int = 42,
    permute_tasks: bool = True,
    permute_seed: int = 42,
    augmentation_factor: int = 2,
    **kwargs
) -> Tuple[Dict, List[Tuple[int,int]], Tuple[int,int,int]]:
    """
    Returns data and meta data for Sequential iNaturalist.
    Automatically downloads the dataset if missing.
    """
    torch.manual_seed(seed)
    data, class_splits = _load_data(
        path,
        num_tasks=num_tasks,
        train_p=train_p,
        augmentation_factor=augmentation_factor
    )

    # Optionally permute tasks
    if permute_tasks:
        rng = torch.Generator()
        rng.manual_seed(permute_seed)
        perm = torch.randperm(len(class_splits), generator=rng).tolist()
        data = {i: data[p] for i, p in enumerate(perm)}
        class_splits = [class_splits[p] for p in perm]

    # Build taskcla list
    taskcla = [(t, data[t]['ncla']) for t in range(num_tasks)]
    data['ncla'] = sum([nc for _, nc in taskcla])
    return data, taskcla, (3,) + image_size


def _load_data(
    path: str,
    num_tasks: int = 50,
    train_p: float = 0.6,
    augmentation_factor: int = 2
) -> Tuple[Dict, List[List[int]]]:
    """
    Loads iNaturalist mini and splits categories into tasks.
    Downloads it if missing.
    """
    # Ensure dataset is downloaded
    version = '2021_train_mini'
    dataset_dir = os.path.join(path, version)

    if not os.path.exists(dataset_dir):
        print(f"iNaturalist dataset not found at {dataset_dir}, downloading...")
        INaturalist(root=path, version=version, download=True)
    else:
        print(f"iNaturalist dataset found at {dataset_dir}, skipping download.")

    # Define transform
    transform_base = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load the dataset
    dataset = INaturalist(root=path, version=version, download=False, transform=transform_base)

    # Get class list and indices
    # iNaturalist uses all_categories and categories_index
    classes = np.array(dataset.all_categories)
    n_classes = len(classes)
    if n_classes < num_tasks:
        raise ValueError(f"Dataset has only {n_classes} classes, fewer than num_tasks={num_tasks}.")

    splits = np.array_split(np.arange(n_classes), num_tasks)

    # Map class -> sample indices
    targets = np.array(dataset.targets)
    class_to_idx = defaultdict(list)
    for i, cls in enumerate(targets):
        class_to_idx[int(cls)].append(i)

    data = {}
    for t, split in enumerate(splits):
        # Indices for this task
        task_indices = [i for cls in split for i in class_to_idx[int(cls)]]
        subset = Subset(dataset, task_indices)

        # Train/valid/test split
        n_total = len(subset)
        n_train = int(train_p * n_total)
        n_val = (n_total - n_train) // 2
        n_test = n_total - n_train - n_val
        train_set, val_set, test_set = random_split(subset, [n_train, n_val, n_test])

        def process_split(split_set, augment=False):
            loader = DataLoader(split_set, batch_size=1, shuffle=False)
            x, y = [], []
            for img, label in tqdm(loader, desc=f"Task {t}"):
                x.append(img.squeeze(0))
                y.append(label)
                if augment:
                    # Simple augmentation: horizontal flip
                    for _ in range(augmentation_factor):
                        aug_img = transforms.RandomHorizontalFlip()(img)
                        x.append(aug_img.squeeze(0))
                        y.append(label)
            return torch.stack(x), torch.LongTensor(y)

        x_train, y_train = process_split(train_set, augment=True)
        x_val, y_val = process_split(val_set)
        x_test, y_test = process_split(test_set)

        data[t] = {
            'name': f'inat-task-{t}',
            'ncla': len(split),
            'train': {'x': x_train, 'y': y_train},
            'valid': {'x': x_val, 'y': y_val},
            'test': {'x': x_test, 'y': y_test}
        }

    return data, splits
