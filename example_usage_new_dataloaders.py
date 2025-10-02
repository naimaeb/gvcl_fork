#!/usr/bin/env python3
"""
Example usage of the new dataloaders in continual learning experiments.
This script demonstrates how to integrate the new dataloaders into existing code.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def example_tinyimagenet_experiment():
    """Example experiment using TinyImageNet dataloader."""
    print("=" * 60)
    print("Example: TinyImageNet Continual Learning Experiment")
    print("=" * 60)
    
    from dataloaders.tinyimagenet import get
    import torch
    
    # Load TinyImageNet data with 20 tasks of 10 classes each
    data, taskcla, size = get(
        seed=42,
        path="../dat/",
        regen=False,
        num_tasks=20,
        classes_per_task=10
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Image size: {size}")
    print(f"Number of tasks: {len(taskcla)}")
    print(f"Total classes: {data['ncla']}")
    
    # Example: Access data for task 0
    task_0_train_x = data[0]['train']['x']
    task_0_train_y = data[0]['train']['y']
    task_0_test_x = data[0]['test']['x']
    task_0_test_y = data[0]['test']['y']
    
    print(f"\nTask 0 statistics:")
    print(f"  Training samples: {task_0_train_x.shape[0]}")
    print(f"  Test samples: {task_0_test_x.shape[0]}")
    print(f"  Classes: {data[0]['ncla']}")
    print(f"  Image shape: {task_0_train_x.shape[1:]}")
    
    # Example: Create a simple dataloader for task 0
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(task_0_train_x, task_0_train_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"\nCreated DataLoader for task 0 with batch_size=32")
    print(f"Number of batches: {len(train_loader)}")
    
    return data, taskcla, size

def example_core50_experiment():
    """Example experiment using CORe50 dataloader."""
    print("\n" + "=" * 60)
    print("Example: CORe50 New-Class Scenario Experiment")
    print("=" * 60)
    
    from dataloaders.core50 import get
    import torch
    
    # Load CORe50 data with new-class scenario
    data, taskcla, size = get(
        seed=42,
        path="../dat/",
        regen=False,
        scenario='nc',  # New-class scenario
        num_tasks=10
    )
    
    print(f"CORe50 dataset loaded successfully!")
    print(f"Image size: {size}")
    print(f"Number of tasks: {len(taskcla)}")
    print(f"Total classes: {data['ncla']}")
    
    # Example: Analyze class distribution across tasks
    print(f"\nClass distribution across tasks:")
    for task_id, num_classes in taskcla:
        train_samples = data[task_id]['train']['x'].shape[0]
        test_samples = data[task_id]['test']['x'].shape[0]
        print(f"  Task {task_id}: {num_classes} classes, "
              f"{train_samples} train, {test_samples} test samples")
    
    return data, taskcla, size

def example_imagenet_experiment():
    """Example experiment using ImageNet variant dataloader."""
    print("\n" + "=" * 60)
    print("Example: ImageNet-R Experiment")
    print("=" * 60)
    
    from dataloaders.imagenet import get
    import torch
    
    # Load ImageNet-R data with 20 tasks of 10 classes each
    data, taskcla, size = get(
        seed=42,
        path="../dat/",
        regen=False,
        variant='r',  # ImageNet-R
        num_tasks=20,
        classes_per_task=10
    )
    
    print(f"ImageNet-R dataset loaded successfully!")
    print(f"Image size: {size}")
    print(f"Number of tasks: {len(taskcla)}")
    print(f"Total classes: {data['ncla']}")
    
    # Example: Check memory usage
    total_memory = 0
    for task_id in data.keys():
        if isinstance(task_id, int):
            task_memory = (data[task_id]['train']['x'].numel() + 
                          data[task_id]['test']['x'].numel()) * 4  # 4 bytes per float32
            total_memory += task_memory
    
    print(f"\nMemory usage: {total_memory / (1024**3):.2f} GB")
    
    return data, taskcla, size

def example_integration_with_existing_code():
    """Example showing how to integrate new dataloaders with existing code."""
    print("\n" + "=" * 60)
    print("Example: Integration with Existing Code")
    print("=" * 60)
    
    # This shows how to replace existing dataloader imports
    # with the new ones in existing code
    
    print("Original code might look like:")
    print("""
    # Original import
    from dataloaders.cifar import get
    
    # Load data
    data, taskcla, size = get(seed=42, path="../dat/")
    """)
    
    print("\nTo use new dataloaders, simply change the import:")
    print("""
    # New import for TinyImageNet
    from dataloaders.tinyimagenet import get
    
    # Load data (same interface!)
    data, taskcla, size = get(seed=42, path="../dat/")
    """)
    
    print("\nOr for CORe50:")
    print("""
    # New import for CORe50
    from dataloaders.core50 import get
    
    # Load data with scenario
    data, taskcla, size = get(seed=42, path="../dat/", scenario='nc')
    """)
    
    print("\nThe data structure remains the same, so existing code")
    print("should work without modification!")

def example_continual_learning_loop():
    """Example continual learning training loop with new dataloaders."""
    print("\n" + "=" * 60)
    print("Example: Continual Learning Training Loop")
    print("=" * 60)
    
    from dataloaders.tinyimagenet import get
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Load data
    data, taskcla, size = get(
        seed=42,
        path="../dat/",
        regen=False,
        num_tasks=5,  # Use fewer tasks for demonstration
        classes_per_task=40  # 200 classes / 5 tasks = 40 classes per task
    )
    
    print(f"Loaded {len(taskcla)} tasks for continual learning experiment")
    
    # Simple model (for demonstration)
    class SimpleModel(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleModel, self).__init__()
            self.features = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    # Continual learning loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model for first task
    input_size = size[0] * size[1] * size[2]  # 3 * 64 * 64 = 12288
    model = SimpleModel(input_size, data[0]['ncla']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nStarting continual learning training...")
    
    for task_id, num_classes in taskcla:
        print(f"\nTraining on Task {task_id} ({num_classes} classes)")
        
        # Get data for current task
        train_x = data[task_id]['train']['x'].to(device)
        train_y = data[task_id]['train']['y'].to(device)
        test_x = data[task_id]['test']['x'].to(device)
        test_y = data[task_id]['test']['y'].to(device)
        
        # Create dataloader
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Training loop (simplified)
        model.train()
        for epoch in range(3):  # 3 epochs per task for demonstration
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_x)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == test_y).float().mean().item()
            print(f"  Task {task_id} Test Accuracy: {accuracy:.4f}")
    
    print(f"\nContinual learning experiment completed!")

def main():
    """Run all examples."""
    print("New Dataloaders Integration Examples")
    print("=" * 80)
    
    try:
        # Example 1: TinyImageNet
        example_tinyimagenet_experiment()
        
        # Example 2: CORe50
        example_core50_experiment()
        
        # Example 3: ImageNet-R
        example_imagenet_experiment()
        
        # Example 4: Integration guide
        example_integration_with_existing_code()
        
        # Example 5: Continual learning loop
        example_continual_learning_loop()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nThe new dataloaders are ready to use in your continual learning experiments.")
        print("They follow the same interface as existing dataloaders, making integration seamless.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install requests tqdm pillow")
        print("Also ensure you have sufficient disk space for dataset downloads.")

if __name__ == "__main__":
    main()
