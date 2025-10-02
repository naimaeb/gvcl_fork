"""Utilities for processing and loading data for Sequential ImageNet variants."""
import os
import json
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import shuffle
from PIL import Image
import requests
import zipfile
from tqdm import tqdm

class ImageNetVariantDataset(Dataset):
    """Custom ImageNet variant dataset loader."""
    size = [3, 224, 224]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, root, variant='subset', train=True, transform=None, download=True, num_classes=1000):
        self.root = root
        self.variant = variant  # 'subset', 'r', 'a'
        self.train = train
        self.transform = transform
        self.num_classes = num_classes
        self.data = []
        self.targets = []
        
        if download:
            self._download()
        
        self._load_data()
    
    def _download(self):
        """Download ImageNet variant dataset if not present."""
        if self.variant == 'subset':
            # Use standard ImageNet
            if os.path.exists(os.path.join(self.root, 'imagenet')):
                return
        elif self.variant == 'r':
            # ImageNet-R
            if os.path.exists(os.path.join(self.root, 'imagenet-r')):
                return
        elif self.variant == 'a':
            # ImageNet-A
            if os.path.exists(os.path.join(self.root, 'imagenet-a')):
                return
        
        os.makedirs(self.root, exist_ok=True)
        
        if self.variant == 'r':
            # Download ImageNet-R
            url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
            print(f"Downloading ImageNet-R to {self.root}...")
            # Note: This is a placeholder URL, actual download would need proper implementation
        elif self.variant == 'a':
            # Download ImageNet-A
            url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
            print(f"Downloading ImageNet-A to {self.root}...")
            # Note: This is a placeholder URL, actual download would need proper implementation
    
    def _load_data(self):
        """Load images and labels from the dataset."""
        if self.variant == 'subset':
            # Use standard ImageNet with subset of classes
            self._load_imagenet_subset()
        elif self.variant == 'r':
            # Load ImageNet-R
            self._load_imagenet_r()
        elif self.variant == 'a':
            # Load ImageNet-A
            self._load_imagenet_a()
    
    def _load_imagenet_subset(self):
        """Load subset of ImageNet classes."""
        # This is a simplified implementation
        # In practice, you would need the actual ImageNet dataset
        imagenet_dir = os.path.join(self.root, 'imagenet')
        
        if not os.path.exists(imagenet_dir):
            print("ImageNet dataset not found. Please download ImageNet manually.")
            return
        
        # Get class directories (first num_classes)
        class_dirs = sorted([d for d in os.listdir(imagenet_dir) 
                           if os.path.isdir(os.path.join(imagenet_dir, d))])[:self.num_classes]
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(imagenet_dir, class_dir)
            
            if self.train:
                # Training data
                train_dir = os.path.join(class_path, 'train')
            else:
                # Validation data
                train_dir = os.path.join(class_path, 'val')
            
            if os.path.exists(train_dir):
                for img_name in os.listdir(train_dir):
                    if img_name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(train_dir, img_name)
                        self.data.append(img_path)
                        self.targets.append(class_idx)
    
    def _load_imagenet_r(self):
        """Load ImageNet-R dataset."""
        # Simplified implementation for ImageNet-R
        imagenet_r_dir = os.path.join(self.root, 'imagenet-r')
        
        if not os.path.exists(imagenet_r_dir):
            print("ImageNet-R dataset not found. Please download ImageNet-R manually.")
            return
        
        # ImageNet-R has 200 classes
        class_dirs = sorted([d for d in os.listdir(imagenet_r_dir) 
                           if os.path.isdir(os.path.join(imagenet_r_dir, d))])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(imagenet_r_dir, class_dir)
            
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.targets.append(class_idx)
    
    def _load_imagenet_a(self):
        """Load ImageNet-A dataset."""
        # Simplified implementation for ImageNet-A
        imagenet_a_dir = os.path.join(self.root, 'imagenet-a')
        
        if not os.path.exists(imagenet_a_dir):
            print("ImageNet-A dataset not found. Please download ImageNet-A manually.")
            return
        
        # ImageNet-A has 200 classes
        class_dirs = sorted([d for d in os.listdir(imagenet_a_dir) 
                           if os.path.isdir(os.path.join(imagenet_a_dir, d))])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(imagenet_a_dir, class_dir)
            
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.targets.append(class_idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get(seed=42, pc_valid=0.10, path="../dat/", regen=False, variant='subset', num_tasks=50, classes_per_task=20):
    """
    ImageNet variant dataloader with configurable task splitting.
    
    Args:
        seed: Random seed for reproducibility
        pc_valid: Percentage of training data to use for validation
        path: Path to data directory
        regen: Whether to regenerate the binary files
        variant: 'subset' (ImageNet subset), 'r' (ImageNet-R), 'a' (ImageNet-A)
        num_tasks: Number of tasks to split the dataset into
        classes_per_task: Number of classes per task
    
    Returns:
        data: Dictionary with data for each task
        taskcla: List of (task_id, num_classes) tuples
        size: Size of images
    """
    rng = np.random.RandomState(seed)
    size = [3, 224, 224]
    
    # Set total classes based on variant
    if variant == 'subset':
        total_classes = 1000
    elif variant in ['r', 'a']:
        total_classes = 200
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    assert num_tasks * classes_per_task <= total_classes, f"num_tasks * classes_per_task must be <= {total_classes}"
    
    download_path = os.path.join(path, f"binary_imagenet_{variant}")
    os.makedirs(download_path, exist_ok=True)
    
    # Files we expect
    task_ids = list(range(num_tasks))
    expected_files = [
        os.path.join(download_path, f"data{t}{s}{xy}.bin")
        for t in task_ids for s in ["train", "test"] for xy in ["x", "y"]
    ]
    map_path = os.path.join(download_path, f"imagenet_{variant}_task_map.json")
    
    need_build = regen or any(not os.path.isfile(f) for f in expected_files) or (not os.path.isfile(map_path))
    
    if need_build:
        print(f"Building ImageNet {variant} dataset...")
        
        # Define transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Load datasets
        train_dataset = ImageNetVariantDataset(path, variant=variant, train=True, 
                                             transform=tfm, download=True, num_classes=total_classes)
        test_dataset = ImageNetVariantDataset(path, variant=variant, train=False, 
                                            transform=tfm, download=True, num_classes=total_classes)
        
        # Create task mapping
        classes = np.arange(total_classes)
        perm = rng.permutation(classes)
        task_groups = perm[:num_tasks * classes_per_task].reshape(num_tasks, classes_per_task)
        
        task_map = {}
        for task_idx, class_group in enumerate(task_groups):
            for class_idx, original_class in enumerate(class_group):
                task_map[int(original_class)] = {"task": task_idx, "label": class_idx}
        
        # Initialize data structure
        data = {}
        for t in range(num_tasks):
            data[t] = {}
            data[t]['name'] = f'imagenet-{variant}-task-{t}'
            data[t]['ncla'] = classes_per_task
            data[t]['train'] = {'x': [], 'y': []}
            data[t]['test'] = {'x': [], 'y': []}
        
        # Process training data
        print("Processing training data...")
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        for image, target in tqdm(train_loader):
            class_id = target.item()
            if class_id in task_map:
                task_info = task_map[class_id]
                task_id = task_info["task"]
                label = task_info["label"]
                data[task_id]['train']['x'].append(image)
                data[task_id]['train']['y'].append(label)
        
        # Process test data
        print("Processing test data...")
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for image, target in tqdm(test_loader):
            class_id = target.item()
            if class_id in task_map:
                task_info = task_map[class_id]
                task_id = task_info["task"]
                label = task_info["label"]
                data[task_id]['test']['x'].append(image)
                data[task_id]['test']['y'].append(label)
        
        # Save data
        print("Saving binary files...")
        for t in data.keys():
            for s in ['train', 'test']:
                if data[t][s]['x']:  # Only save if there's data
                    data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                    data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                    torch.save(data[t][s]['x'], os.path.join(download_path, f'data{t}{s}x.bin'))
                    torch.save(data[t][s]['y'], os.path.join(download_path, f'data{t}{s}y.bin'))
        
        # Save task mapping
        with open(map_path, "w") as f:
            json.dump({
                "seed": int(seed),
                "variant": variant,
                "num_tasks": num_tasks,
                "classes_per_task": classes_per_task,
                "total_classes": total_classes,
                "task_groups": task_groups.tolist(),
                "map": task_map
            }, f, indent=2)
    
    # Load binary files
    data = {}
    print(f"Loading ImageNet {variant} data for {num_tasks} tasks...")
    for i in task_ids:
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(download_path, f'data{i}{s}x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(download_path, f'data{i}{s}y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = f"imagenet-{variant}-task-{i}"
    
    # Create validation split
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()
    
    # Create task list and calculate total classes
    taskcla, total = [], 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        total += data[t]['ncla']
    data['ncla'] = total
    
    return data, taskcla, size
