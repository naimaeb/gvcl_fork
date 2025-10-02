"""Utilities for processing and loading data for Sequential Caltech256."""
import os
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from torchvision.datasets import Caltech256
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split

import numpy as np
from PIL import Image
from tqdm import tqdm

class Caltech256WithCategory(Caltech256):
    """ Modified version of the Caltech256 dataset that also returns the category label for each image. """
    size = [3, 224, 224]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root, transform=None, target_transform=None, download=False, resize=(224, 224)):
        transform = transforms.Compose([transforms.Resize(resize), transform])
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.num_categories = len(self.categories)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target category class.
        """
        image, target = super().__getitem__(index)
        return image, target, target  # category label is the same as target for Caltech256
    
def get(path: str = "../dat/", seed=42, permute_tasks=True, permute_seed=42, **kwargs):
    """Returns data and meta data for Sequential Caltech256.
    
    Args:
        path: Path to data directory
        seed: Random seed for data splitting
        permute_tasks: If True, permutes the order of tasks
        permute_seed: Specific seed for task permutation (uses seed if None)
        **kwargs: Additional arguments
        
    Returns:
        data: Dictionary with data for each task
        taskcla: List of (task_id, num_classes) tuples
        size: Size of images
    """
    torch.manual_seed(seed) 
    print(f"permute_tasks = {permute_tasks}")

    
    data = {} # dictionary task->{'train'|'valid' -> {'x'|'y'|'a': torch.tensor}} 
    taskcla = [] # number of classes for each task
    download_path = path + 'caltech256/'

    raw_data, size = _load_data(download_path, num_tasks=50, 
                               train_p=kwargs.get('train_p', 0.6), 
                               resize=kwargs.get('resize', True), 
                               augmentation_factor=5)
    
    # Apply task permutation if requested
    if permute_tasks:
        permute_rng = torch.Generator()
        permute_rng.manual_seed(permute_seed if permute_seed is not None else seed)
        
        # Generate permutation of task indices
        num_tasks = len(raw_data) - 1 if 'ncla' in raw_data else len(raw_data)
        task_permutation = torch.randperm(num_tasks, generator=permute_rng).tolist()
        
        # Apply permutation to data
        permuted_data = {}
        for new_idx, original_idx in enumerate(task_permutation):
            permuted_data[new_idx] = raw_data[original_idx]
        
        # Keep any metadata
        for key in raw_data:
            if not isinstance(key, int):
                permuted_data[key] = raw_data[key]
                
        data = permuted_data
        print(f"Tasks permuted with seed {permute_seed if permute_seed is not None else seed}")
        print(f"Task permutation: {task_permutation}")
    else:
        data = raw_data

    # Calculating total number of classes in the dataset
    n = 0
    for t in data.keys():
        if isinstance(t, int):  # Skip non-integer keys like 'ncla'
            taskcla.append((t, data[t]['ncla']))
            n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size 


def _load_data(download_dir: str, num_tasks=50, train_p=0.6, resize=False, augmentation_factor=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Down)Load raw Caltech256 data.""" 
    new_size = (224, 224) if resize else (224, 224)

    # Load Caltech256 dataset (similar to Omniglot approach)
    caltech256_dataset = Caltech256WithCategory(
        download_dir, 
        download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(Caltech256WithCategory.mean, Caltech256WithCategory.std)
        ]), 
        resize=new_size
    )
    
    # Caltech256 has 256 categories, we'll use 250 (50 tasks * 5 classes per task)
    # Filter to only include first 250 categories
    filtered_indices = []
    for idx in range(len(caltech256_dataset)):
        _, target, _ = caltech256_dataset[idx]
        if target < 250:  # Only use first 250 categories
            filtered_indices.append(idx)
    
    # Create subset dataset
    subset_dataset = torch.utils.data.Subset(caltech256_dataset, filtered_indices)

    binaries_dir = os.path.join(download_dir, 'binaries/')
    binaries_exist = all(os.path.isfile(os.path.join(binaries_dir, f'data{t}{s}x.bin')) for t in range(num_tasks) for s in ['train', 'test', 'valid'])
    
    if not binaries_exist:
        if not os.path.exists(download_dir): 
            os.makedirs(download_dir)
        if not os.path.exists(binaries_dir):    
            os.makedirs(binaries_dir)

        data = {} # final data dictionary
        # CALTECH256
        dat={} # temporary holder of data
        # Split the data into train, test, and validation
        dat['train'], dat['test'], dat['valid'] = random_split(subset_dataset, [train_p, (1.0-train_p)/2, (1.0-train_p)/2])

        print("Generating dataset files... (this may take a while)")
        for n in range(num_tasks):
            data[n]={}
            data[n]['train']={'x': [],'y': [], 'a': []}
            data[n]['test']={'x': [],'y': [], 'a': []}
            data[n]['valid']={'x': [],'y': [], 'a': []}

        for s in ['train','test','valid']:
            loader=DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target,category in tqdm(loader):
                # Map category to task (5 categories per task)
                task = category.item() // 5
                if task >= num_tasks:  # Skip if task index is out of range
                    continue
                    
                # Map category to class within task (0-4)
                class_within_task = category.item() % 5
                
                data[task][s]['x'].append(image)
                data[task][s]['y'].append(torch.tensor(class_within_task))
                data[task][s]['a'].append(category)
                
                if s == 'train': # augment training data by a factor of augmentation_factor
                    for _ in range(augmentation_factor):
                        augmented_image = transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1))(image)
                        data[task][s]['x'].append(augmented_image)
                        data[task][s]['y'].append(torch.tensor(class_within_task))
                        data[task][s]['a'].append(category)
        
        # "Unify" and save
        for t in data.keys():
            for s in ['train','test','valid']:
                if data[t][s]['x']:  # Only process if there's data
                    data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,*image.size()[1:])
                    data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                    data[t][s]['a']=torch.LongTensor(np.array(data[t][s]['a'],dtype=int)).view(-1)
                    torch.save(data[t][s]['x'], os.path.join(binaries_dir,'data'+str(t)+s+'x.bin'))
                    torch.save(data[t][s]['y'], os.path.join(binaries_dir,'data'+str(t)+s+'y.bin'))
                    torch.save(data[t][s]['a'], os.path.join(binaries_dir,'data'+str(t)+s+'a.bin'))
            data[t]['ncla']=len(np.unique(data[t]['train']['y'].numpy())) if data[t]['train']['y'] else 0
            data[t]['name']='caltech256-task-'+str(t)  
    else: 
        # Load binary files
        data={}
        for i in range(num_tasks):
            data[i] = dict.fromkeys(['name','ncla','train','test'])
            for s in ['train','test','valid']:
                data[i][s]={'x':[],'y':[],'a':[]}
                data[i][s]['x']=torch.load(os.path.join(binaries_dir,'data'+str(i)+s+'x.bin'))
                data[i][s]['y']=torch.load(os.path.join(binaries_dir,'data'+str(i)+s+'y.bin'))
                data[i][s]['y'] = data[i][s]['y'] - data[i][s]['y'].min()  # Normalize labels like Omniglot
                data[i][s]['a']=torch.load(os.path.join(binaries_dir,'data'+str(i)+s+'a.bin'))
            
            data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
            data[i]['name']='caltech256-task-'+str(i)  

    return data, new_size
