"""Utilities for processing and loading data for Sequential CORe50."""
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

class CORe50Dataset(Dataset):
    """Custom CORe50 dataset loader."""
    size = [3, 128, 128]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def __init__(self, root, train=True, transform=None, download=True, scenario='nc'):
        self.root = root
        self.train = train
        self.transform = transform
        self.scenario = scenario  # 'nc' (new-class), 'ni' (new-instance), 'nd' (new-domain)
        self.data = []
        self.targets = []
        self.sessions = []
        
        if download:
            self._download()
        
        self._load_data()
    
    def _download(self):
        """Download CORe50 dataset if not present."""
        if os.path.exists(os.path.join(self.root, 'core50')):
            return
            
        os.makedirs(self.root, exist_ok=True)
        
        # CORe50 download URL - Official download link from vlomonaco.github.io/core50
        url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
        
        print(f"Downloading CORe50 to {self.root}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        zip_path = os.path.join(self.root, 'core50_128x128.zip')
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Extracting CORe50...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        # Clean up zip file
        os.remove(zip_path)
    
    def _load_data(self):
        """Load images and labels from the dataset."""
        data_dir = os.path.join(self.root, 'core50')
        
        # CORe50 has 50 objects, 11 sessions, 8 different conditions
        # We'll organize by sessions for continual learning
        
        if self.scenario == 'nc':
            # New-class scenario: each session introduces new classes
            self._load_new_class_scenario(data_dir)
        elif self.scenario == 'ni':
            # New-instance scenario: each session introduces new instances of same classes
            self._load_new_instance_scenario(data_dir)
        elif self.scenario == 'nd':
            # New-domain scenario: each session introduces new domains/conditions
            self._load_new_domain_scenario(data_dir)
    
    def _load_new_class_scenario(self, data_dir):
        """Load data for new-class scenario."""
        # Group objects by sessions (each session introduces new objects)
        objects_per_session = 5  # 50 objects / 10 sessions = 5 objects per session
        
        for session in range(10):  # 10 sessions for new-class
            start_obj = session * objects_per_session
            end_obj = start_obj + objects_per_session
            
            for obj_id in range(start_obj, end_obj):
                obj_dir = os.path.join(data_dir, f'obj_{obj_id:02d}')
                if os.path.exists(obj_dir):
                    for condition in range(8):  # 8 conditions
                        cond_dir = os.path.join(obj_dir, f'cond_{condition}')
                        if os.path.exists(cond_dir):
                            for img_name in os.listdir(cond_dir):
                                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                    img_path = os.path.join(cond_dir, img_name)
                                    self.data.append(img_path)
                                    self.targets.append(obj_id)
                                    self.sessions.append(session)
    
    def _load_new_instance_scenario(self, data_dir):
        """Load data for new-instance scenario."""
        # Each session introduces new instances of the same 50 objects
        for obj_id in range(50):
            obj_dir = os.path.join(data_dir, f'obj_{obj_id:02d}')
            if os.path.exists(obj_dir):
                for condition in range(8):  # 8 conditions
                    cond_dir = os.path.join(obj_dir, f'cond_{condition}')
                    if os.path.exists(cond_dir):
                        for img_name in os.listdir(cond_dir):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(cond_dir, img_name)
                                self.data.append(img_path)
                                self.targets.append(obj_id)
                                self.sessions.append(condition)  # Use condition as session
    
    def _load_new_domain_scenario(self, data_dir):
        """Load data for new-domain scenario."""
        # Each session introduces new domains/conditions
        for obj_id in range(50):
            obj_dir = os.path.join(data_dir, f'obj_{obj_id:02d}')
            if os.path.exists(obj_dir):
                for condition in range(8):  # 8 conditions
                    cond_dir = os.path.join(obj_dir, f'cond_{condition}')
                    if os.path.exists(cond_dir):
                        for img_name in os.listdir(cond_dir):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(cond_dir, img_name)
                                self.data.append(img_path)
                                self.targets.append(obj_id)
                                self.sessions.append(condition)  # Use condition as session
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def get(seed=42, pc_valid=0.10, path="../dat/", regen=False, scenario='nc', num_tasks=10):
    """
    CORe50 dataloader with different continual learning scenarios.
    
    Args:
        seed: Random seed for reproducibility
        pc_valid: Percentage of training data to use for validation
        path: Path to data directory
        regen: Whether to regenerate the binary files
        scenario: 'nc' (new-class), 'ni' (new-instance), 'nd' (new-domain)
        num_tasks: Number of tasks to split the dataset into
    
    Returns:
        data: Dictionary with data for each task
        taskcla: List of (task_id, num_classes) tuples
        size: Size of images
    """
    rng = np.random.RandomState(seed)
    size = [3, 128, 128]
    # Convert relative path to absolute path
    abs_path = os.path.abspath(path)
    download_path = os.path.join(abs_path, f"binary_core50_{scenario}")
    os.makedirs(download_path, exist_ok=True)
    
    # Files we expect
    task_ids = list(range(num_tasks))
    expected_files = [
        os.path.join(download_path, f"data{t}{s}{xy}.bin")
        for t in task_ids for s in ["train", "test"] for xy in ["x", "y"]
    ]
    map_path = os.path.join(download_path, f"core50_{scenario}_task_map.json")
    
    need_build = regen or any(not os.path.isfile(f) for f in expected_files) or (not os.path.isfile(map_path))
    
    if need_build:
        print(f"Building CORe50 dataset for scenario: {scenario}...")
        
        # Define transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Load dataset
        dataset = CORe50Dataset(abs_path, train=True, transform=tfm, download=True, scenario=scenario)
        
        # Create task mapping based on sessions
        unique_sessions = sorted(list(set(dataset.sessions)))
        print(f"Found {len(unique_sessions)} unique sessions: {unique_sessions}")
        print(f"Dataset loaded {len(dataset)} samples")
        
        if not unique_sessions:
            raise ValueError("No data loaded from CORe50 dataset. Check if the dataset was downloaded and extracted correctly.")
        
        session_to_task = {}
        
        if scenario == 'nc':
            # New-class: each session is a task with new classes
            classes_per_task = 5  # 50 objects / 10 tasks = 5 objects per task
            for session in unique_sessions:
                session_to_task[session] = session
        elif scenario == 'ni':
            # New-instance: each session has instances of all classes
            classes_per_task = 50  # All 50 objects in each task
            for session in unique_sessions:
                session_to_task[session] = session
        elif scenario == 'nd':
            # New-domain: each session has different conditions
            classes_per_task = 50  # All 50 objects in each task
            for session in unique_sessions:
                session_to_task[session] = session
        
        # Initialize data structure
        data = {}
        for t in range(num_tasks):
            data[t] = {}
            data[t]['name'] = f'core50-{scenario}-task-{t}'
            data[t]['ncla'] = classes_per_task
            data[t]['train'] = {'x': [], 'y': []}
            data[t]['test'] = {'x': [], 'y': []}
        
        # Process data
        print("Processing data...")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for (image, target), session in tqdm(zip(loader, dataset.sessions)):
            task_id = session_to_task.get(session, 0)
            if task_id < num_tasks:
                data[task_id]['train']['x'].append(image)
                data[task_id]['train']['y'].append(target.item())
        
        # Create test split (use last session as test)
        test_session = max(unique_sessions)
        test_data = []
        test_targets = []
        
        for idx, session in enumerate(dataset.sessions):
            if session == test_session:
                test_data.append(dataset.data[idx])
                test_targets.append(dataset.targets[idx])
        
        # Load test images
        for img_path, target in tqdm(zip(test_data, test_targets), desc="Loading test data"):
            image = Image.open(img_path).convert('RGB')
            image = tfm(image).unsqueeze(0)
            
            # Assign to appropriate task
            task_id = session_to_task.get(test_session, 0)
            if task_id < num_tasks:
                data[task_id]['test']['x'].append(image)
                data[task_id]['test']['y'].append(target)
        
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
                "scenario": scenario,
                "num_tasks": num_tasks,
                "session_to_task": session_to_task,
                "unique_sessions": unique_sessions
            }, f, indent=2)
    
    # Load binary files
    data = {}
    print(f"Loading CORe50 data for {num_tasks} tasks (scenario: {scenario})...")
    for i in task_ids:
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(download_path, f'data{i}{s}x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(download_path, f'data{i}{s}y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = f"core50-{scenario}-task-{i}"
    
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
