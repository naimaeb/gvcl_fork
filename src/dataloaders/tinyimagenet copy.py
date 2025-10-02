"""Utilities for processing and loading data for Sequential TinyImageNet."""
import os
import json
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import shuffle
from PIL import Image
import requests
import zipfile
from tqdm import tqdm

class TinyImageNetDataset(Dataset):
    """Custom TinyImageNet dataset loader."""
    size = [3, 64, 64]
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]
    
    def __init__(self, root, train=True, transform=None, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        
        if download:
            self._download()
        
        self._load_data()
    
    def _download(self):
        """Download TinyImageNet dataset if not present."""
        if os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
            return
            
        os.makedirs(self.root, exist_ok=True)
        
        # TinyImageNet download URL (same for both train and test)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        
        print(f"Downloading TinyImageNet to {self.root}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        zip_path = os.path.join(self.root, 'tiny-imagenet-200.zip')
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Extracting TinyImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        # Clean up zip file
        os.remove(zip_path)
    
    def _load_data(self):
        """Load images and labels from the dataset."""
        data_dir = os.path.join(self.root, 'tiny-imagenet-200')
        
        if self.train:
            train_dir = os.path.join(data_dir, 'train')
        else:
            train_dir = os.path.join(data_dir, 'val')
        
        # Get class directories
        class_dirs = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(train_dir, class_dir)
            
            if self.train:
                # Training data is in subdirectories
                images_dir = os.path.join(class_path, 'images')
            else:
                # Validation data is directly in class directory
                images_dir = class_path
            
            if os.path.exists(images_dir):
                for img_name in os.listdir(images_dir):
                    if img_name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(images_dir, img_name)
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

def get(seed=42, pc_valid=0.10, path="../dat/", regen=False, num_tasks=40, classes_per_task=5):
    """
    TinyImageNet dataloader with configurable task splitting.
    
    Args:
        seed: Random seed for reproducibility
        pc_valid: Percentage of training data to use for validation
        path: Path to data directory
        regen: Whether to regenerate the binary files
        num_tasks: Number of tasks to split the dataset into
        classes_per_task: Number of classes per task
    
    Returns:
        data: Dictionary with data for each task
        taskcla: List of (task_id, num_classes) tuples
        size: Size of images
    """
    rng = np.random.RandomState(seed)
    size = [3, 64, 64]
    # Convert relative path to absolute path
    abs_path = os.path.abspath(path)
    download_path = os.path.join(abs_path, "binary_tinyimagenet")
    os.makedirs(download_path, exist_ok=True)
    
    total_classes = 200
    assert num_tasks * classes_per_task == total_classes, f"num_tasks * classes_per_task must equal {total_classes}"
    
    # Files we expect
    task_ids = list(range(num_tasks))
    expected_files = [
        os.path.join(download_path, f"data{t}{s}{xy}.bin")
        for t in task_ids for s in ["train", "test"] for xy in ["x", "y"]
    ]
    map_path = os.path.join(download_path, "tinyimagenet_task_map.json")
    
    need_build = regen or any(not os.path.isfile(f) for f in expected_files) or (not os.path.isfile(map_path))
    
    if need_build:
        print("Building TinyImageNet dataset...")
        
        # Define transforms
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Load datasets
        train_dataset = TinyImageNetDataset(abs_path, train=True, transform=tfm, download=True)
        test_dataset = TinyImageNetDataset(abs_path, train=False, transform=tfm, download=True)
        
        # Create task mapping
        classes = np.arange(total_classes)
        perm = rng.permutation(classes)
        task_groups = perm.reshape(num_tasks, classes_per_task)
        
        task_map = {}
        for task_idx, class_group in enumerate(task_groups):
            for class_idx, original_class in enumerate(class_group):
                task_map[int(original_class)] = {"task": task_idx, "label": class_idx}
        
        # Initialize data structure
        data = {}
        for t in range(num_tasks):
            data[t] = {}
            data[t]['name'] = f'tinyimagenet-task-{t}'
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
                "num_tasks": num_tasks,
                "classes_per_task": classes_per_task,
                "task_groups": task_groups.tolist(),
                "map": task_map
            }, f, indent=2)
    
    # Load binary files
    data = {}
    print(f"Loading TinyImageNet data for {num_tasks} tasks...")
    for i in task_ids:
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(download_path, f'data{i}{s}x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(download_path, f'data{i}{s}y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = f"tinyimagenet-task-{i}"
    
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
