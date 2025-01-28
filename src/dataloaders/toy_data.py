import os,sys
import numpy as np
import torch
from sklearn.datasets._samples_generator import make_blobs
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




def get(seed=0, fixed_order=False, option = 0, pc_valid=0.1, path=None):
    data = {}
    taskcla = []
    size = [2]  # Each data point is a 2D coordinate (x, y)

    torch.manual_seed(42) 
    num_samples_per_task = 2000
    num_tasks = 5
    std = 0.5  # Standard deviation of the clusters


    # Instantiate ToydataGenerator to get centers and std
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
        x, y = make_blobs(n_samples=num_samples_per_task, centers=task_centers, cluster_std=task_std, random_state=seed)

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

