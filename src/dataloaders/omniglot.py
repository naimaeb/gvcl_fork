"""Utilities for processing and loading data for Sequential Omniglot."""
import os
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from torchvision.datasets import Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split

import numpy as np
from PIL import Image
from tqdm import tqdm

class OmniglotWithAlphabet(Omniglot):
    """ Modified version of the Omniglot dataset that also returns the alphabet label for each image. """
    size = [1,105,105]
    mean=[0.9221]
    std=[0.2681]

    def __init__(self, root, background=True, transform=None, target_transform=None, download=False, resize=(105,105)):
        transform = transforms.Compose([transforms.Resize(resize), transform])
        super().__init__(root, background=background, transform=transform, target_transform=target_transform, download=download)
        self.num_alphabets = len(self._alphabets)
        

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        alphabet_name = self._characters[character_class].split('/')[0]
        alphabet_label = self._alphabets.index(alphabet_name) 
        if not self.background: alphabet_label +=30 #counting the background ones first
        image = Image.open(image_path, mode="r").convert("L")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class, alphabet_label
    
def get(path: str = "../dat/", seed=42, **kwargs):
    """Returns data and meta data for Sequential Omniglot."""
    torch.manual_seed(42) 
    
    data={} # dictionary task->{'train'|'valid' -> {'x'|'y'|'a': torch.tensor}} 
    taskcla=[] # number of classes for each task
    download_path=path+'omniglot/'

    
    data, size = _load_data(download_path, num_tasks=50, train_p=kwargs.get('train_p' , 0.6), resize=kwargs.get('resize', True), augmentation_factor=20)


    # Calculating total number of classes in the dataset
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size 


def _load_data(download_dir: str, num_tasks=50, train_p=0.6, resize=False, augmentation_factor=20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Down)Load raw Omniglot data.""" 
    new_size = (28,28) if resize else (105,105)

    training_alphabets=OmniglotWithAlphabet(download_dir,background=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(OmniglotWithAlphabet.mean,OmniglotWithAlphabet.std)]), resize=new_size)
    test_alphabets=OmniglotWithAlphabet(download_dir,background=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(OmniglotWithAlphabet.mean,OmniglotWithAlphabet.std)]), resize=new_size)
    all_alphabets = ConcatDataset([training_alphabets, test_alphabets])
    alphabet_names= training_alphabets._alphabets + test_alphabets._alphabets
    


    binaries_dir = os.path.join(download_dir, 'binaries/')
    binaries_exist = all(os.path.isfile(os.path.join(binaries_dir, f'data{t}{s}x.bin')) for t in range(num_tasks) for s in ['train', 'test', 'valid'])
    if not binaries_exist:
        if not os.path.exists(download_dir): 
            os.makedirs(download_dir)
        if not os.path.exists(binaries_dir):    
            os.makedirs(binaries_dir)

        data = {} # final data dictionary
        # OMNIGLOT
        dat={} # temporary holder of data
        # Need to split the data manually because the Omniglot dataset does not have a predefined train-test split
        # Define the sizes for train and test splits
        dat['train'], dat['test'], dat['valid'] = random_split(all_alphabets, [train_p, (1.0-train_p)/2, (1.0-train_p)/2])

        print("Generating dataset files... (this may take a while)")
        for n in range(num_tasks):
            data[n]={}
            data[n]['train']={'x': [],'y': [], 'a': []}
            data[n]['test']={'x': [],'y': [], 'a': []}
            data[n]['valid']={'x': [],'y': [], 'a': []}

        for s in ['train','test','valid']:
            loader=DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target,alphabet in tqdm(loader):
                task = alphabet.item()
                data[task][s]['x'].append(image)
                data[task][s]['y'].append(target)
                data[task][s]['a'].append(alphabet)
                if s == 'train': # augment training data by a factor of augmentation_factor
                    for _ in range(augmentation_factor):
                        augmented_image=transforms.RandomAffine(30, translate=(0.2,0/2))(image)
                        data[task][s]['x'].append(augmented_image)
                        data[task][s]['y'].append(target)
                        data[task][s]['a'].append(alphabet)
        # "Unify" and save
        for t in data.keys():
            for s in ['train','test','valid']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,*image.size()[1:])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                data[t][s]['a']=torch.LongTensor(np.array(data[t][s]['a'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(binaries_dir,'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(binaries_dir,'data'+str(t)+s+'y.bin'))
                torch.save(data[t][s]['a'], os.path.join(binaries_dir,'data'+str(t)+s+'a.bin'))
            data[t]['ncla']=len(np.unique(data[t]['train']['y'].numpy()))
            data[t]['name']='omniglot-'+alphabet_names[t]  
    else: 
        # Load binary files
        data={}
        for i in range(num_tasks):
            data[i] = dict.fromkeys(['name','ncla','train','test'])
            for s in ['train','test','valid']:
                data[i][s]={'x':[],'y':[],'a':[]}
                data[i][s]['x']=torch.load(os.path.join(binaries_dir,'data'+str(i)+s+'x.bin'))
                data[i][s]['y']=torch.load(os.path.join(binaries_dir,'data'+str(i)+s+'y.bin'))
                data[i][s]['y'] = data[i][s]['y'] - data[i][s]['y'].min()
                data[i][s]['a']=torch.load(os.path.join(binaries_dir,'data'+str(i)+s+'a.bin'))
            
            data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
            data[i]['name']='omniglot-'+alphabet_names[i]  

    return data, new_size
    