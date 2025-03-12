import os,sys
import numpy as np
import torch
#from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, Subset
import pickle
import gzip
from copy import deepcopy

########################################################################################################################
class ToydataGenerator():
    def __init__(self, max_iter=5, num_samples=2000, option=0):

        self.offset = 5  # Offset when loading data in next_task()

        # Generate data
        if option == 0:
            # Standard settings
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 1:
            # Six tasks
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3], [1.65, 0.1],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1], [0.7, 0.25]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16], [0.14, 0.14],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22], [0.14, 0.14]]

        elif option == 2:
            # All std devs increased
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.12, 0.22], [0.24, 0.12], [0.07, 0.2], [0.16, 0.08], [0.08, 0.16],
                [0.12, 0.16], [0.16, 0.12], [0.08, 0.16], [0.24, 0.08], [0.08, 0.22]]

        elif option == 3:
            # Tougher to separate
            centers = [[0, 0.2], [0.6, 0.65], [1.3, 0.4], [1.6, -0.22], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.55], [1., 0.1], [1.7, -0.3], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        self.centers = centers
        self.std = std



class TaskDatasetGenerator:
    def __init__(self, n_tasks=5, n_samples=2000, task_centers=None, overlap_ratio=0.3, random_state=42):
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.task_centers = task_centers if task_centers is not None else self._default_task_centers()
        self.overlap_ratio = overlap_ratio
        self.random_state = random_state

        # Calculate numbers for each class
        self.n_per_class = self.n_samples // 2
        self.n_overlap = int(self.n_per_class * self.overlap_ratio)
        self.n_core = self.n_per_class - self.n_overlap

        self.rng = np.random.RandomState(random_state)
    

    def _default_task_centers(self):
        
        '''
        return [ [[0, 0.2], [0.6, 0.65]],
                [[1.3, 0.4], [1.6, -0.22]], 
                [[2.0, 0.3], [0.45, 0]],
                [[0.7, 0.55], [1., 0.1]], 
                [[1.7, -0.3], [2.3, 0.1]]]
        '''
        return [
            [[-2, 0], [1, 0]],
            [[0, -2], [0, 1]],
            [[2, 2], [-2, -2]],
            [[-1, 1], [1, -1]],
            [[3, 0], [0, 3]]
        ]
        

    def _generate_class_points(self, centers,class_idx):
            # Generate core points with smaller std
            X_core, _ = make_blobs(n_samples=self.n_core,
                                 centers=[centers[class_idx]],
                                 cluster_std=0.5,
                                 random_state=self.rng.randint(1000))
            
            # Generate overlapping points with larger std, centered between the two classes
            overlap_center = np.mean([centers[class_idx], centers[1-class_idx]], axis=0)
            X_overlap, _ = make_blobs(n_samples=self.n_overlap,
                                    centers=[overlap_center],
                                    cluster_std=1.5,
                                    random_state=self.rng.randint(1000))
            
            # Combine points for this class
            X_class = np.vstack([X_core, X_overlap])
            y_class = np.full(self.n_per_class, class_idx)
            
            return X_class, y_class
    
    def _generate_task_data(self, centers):
        #always have 2 classes and want to stack them

        X0, y0 = self._generate_class_points(centers, 0)
        X1, y1 = self._generate_class_points(centers, 1)

        # Combine classes
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])  

        return X, y

    def _generate_datasets(self):
        datasets = []
        for i in range(self.n_tasks):
            # Generate data for both tasks
            centers = self.task_centers[i]
            X1, y1 = self._generate_task_data(centers)
            
            # Shuffle all datasets with the same permutation
            shuffle_idx = self.rng.permutation(len(X1))
            X1 = X1[shuffle_idx]
            y1 = y1[shuffle_idx]

                # Save X1 and y1 as np.ndarray
            X1 = np.array(X1)
            y1 = np.array(y1)
            datasets.append(X1)
            datasets.append(y1)
        return datasets

    def get_datasets(self):
        return self._generate_datasets()



def get(seed=0, n_tasks = 3, n_samples = 2000, fixed_order=False, option = 0, pc_valid=0.1, path=None):
    data = {}
    taskcla = []
    size = [2]  # Each data point is a 2D coordinate (x, y)

    torch.manual_seed(42) 
    num_samples_per_task = n_samples
    num_tasks = n_tasks
    std = 0.5  # Standard deviation of the clusters

    # Example usage
    generator = TaskDatasetGenerator(n_tasks=n_tasks, n_samples=n_samples)
    task_data = generator.get_datasets()
    print(task_data)
    #task_data = generate_two_task_dataset()

    # Instantiate ToydataGenerator to get centers and std, not used currently
    toy_data_generator = ToydataGenerator(option=option)
    centers = toy_data_generator.centers
    std = toy_data_generator.std

    for t in range(num_tasks):
        data[t] = {}
        data[t]['name'] = f'toy-2d-binary-{t}'
        data[t]['ncla'] = 2
        data[t]['train'] = {'x': [], 'y': []}
        data[t]['test'] = {'x': [], 'y': []}
        data[t]['valid'] = {'x': [], 'y': []}

        task_centers = [centers[2*t], centers[2*t + 1]]
        task_std = [std[2*t], std[2*t + 1]]
        #x, y = make_blobs(n_samples=num_samples_per_task, centers=task_centers, cluster_std=task_std, random_state=seed)

        #print("good_shapes",x.shape, y.shape)
        x, y = task_data[2*t], task_data[2*t + 1]
        #print("current_shapes",x1.shape, y1.shape)
        # Split data into train, validation, and test sets
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=pc_valid + 0.1, random_state=seed)
        x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=seed)

        # Store data in the format required
        data[t]['train']['x'] = torch.tensor(x_train, dtype=torch.float32)
        data[t]['train']['y'] = torch.tensor(y_train, dtype=torch.long)
        data[t]['valid']['x'] = torch.tensor(x_valid, dtype=torch.float32)
        data[t]['valid']['y'] = torch.tensor(y_valid, dtype=torch.long)
        data[t]['test']['x'] = torch.tensor(x_test, dtype=torch.float32)
        data[t]['test']['y'] = torch.tensor(y_test, dtype=torch.long)

        taskcla.append((t, 2))

    return data, taskcla, size

def visualize_data(data):
    num_tasks = len(data)
    fig, axes = plt.subplots(1, num_tasks, figsize=(15, 3))
    for t in range(num_tasks):
        x_train = data[t]['test']['x'].numpy()
        y_train = data[t]['test']['y'].numpy()
        axes[t].scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], label='Class 0', alpha=0.5)
        axes[t].scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], label='Class 1', alpha=0.5)
        axes[t].set_title(f'Task {t}')
        axes[t].legend()
    plt.show()

