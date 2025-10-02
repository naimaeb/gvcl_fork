import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle

def get(seed=0, pc_valid=0.10, path='../dat/', num_tasks=40, classes_per_task=5):
    """
    TinyImageNet Sequential Loader:
    - Splits 200 classes into 40 tasks with 5 classes each.
    - Preprocesses and caches binary files for efficiency.
    """
    data = {}
    taskcla = []
    size = [3, 64, 64]
    download_path = os.path.join(path, 'binary_tinyimagenet')
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]

     # Check if binary files exist for the requested configuration
     expected_files = []
     for i in range(num_tasks):
         for s in ['train', 'test']:
             expected_files.append(os.path.join(download_path, f'data{i}{s}x.bin'))
             expected_files.append(os.path.join(download_path, f'data{i}{s}y.bin'))
     
     files_exist = all(os.path.exists(f) for f in expected_files)
     
     if not files_exist:
         print("Binary files not found or incomplete. Creating new ones...")
         os.makedirs(download_path, exist_ok=True)

         # --- Load TinyImageNet training and validation sets ---
         train_dir = os.path.join(path, 'tiny-imagenet-200', 'train')
         val_dir = os.path.join(path, 'tiny-imagenet-200', 'val')

         transform = transforms.Compose([
             transforms.Resize((64, 64)),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)
         ])

         dat = {}
         dat['train'] = datasets.ImageFolder(train_dir, transform=transform)
         dat['test'] = datasets.ImageFolder(val_dir, transform=transform)

         # Shuffle and group classes
         all_classes = list(range(len(dat['train'].classes)))  # 200 classes
         all_classes = shuffle(all_classes, random_state=seed)
         grouped = [all_classes[i:i + classes_per_task] for i in range(0, num_tasks * classes_per_task, classes_per_task)]

         # Create tasks
         for t, cls_group in enumerate(grouped):
             data[t] = {}
             data[t]['name'] = f'tinyimagenet-{t}'
             data[t]['ncla'] = len(cls_group)
             data[t]['train'] = {'x': [], 'y': []}
             data[t]['test'] = {'x': [], 'y': []}

         # --- Assign images to tasks ---
         for s in ['train', 'test']:
             loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
             for image, target in loader:
                 for t, cls_group in enumerate(grouped):
                     if target.item() in cls_group:
                         new_label = cls_group.index(target.item())
                         data[t][s]['x'].append(image)
                         data[t][s]['y'].append(new_label)
                         break

         # --- Save to binary files ---
         for t in data.keys():
             for s in ['train', 'test']:
                 data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, *size)
                 data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                 torch.save(data[t][s]['x'], os.path.join(download_path, f'data{t}{s}x.bin'))
                 torch.save(data[t][s]['y'], os.path.join(download_path, f'data{t}{s}y.bin'))
     else:
         print("Loading existing binary files...")
         # --- Load from binary files ---
         ids = list(range(num_tasks))
         print('Task order =', ids)
         data = {}
         for i in ids:
             data[i] = {'name': f'tinyimagenet-{i}', 'ncla': classes_per_task}
             for s in ['train', 'test']:
                 data[i][s] = {'x': [], 'y': []}
                 data[i][s]['x'] = torch.load(os.path.join(download_path, f'data{i}{s}x.bin'))
                 data[i][s]['y'] = torch.load(os.path.join(download_path, f'data{i}{s}y.bin'))

    # --- Create validation split ---
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

    # --- Build taskcla and ncla ---
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size
