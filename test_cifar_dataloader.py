#!/usr/bin/env python3
"""
Test script for the cifar.py dataloader
"""

import sys
import os
sys.path.append('.')

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.dataloaders import cifar

def test_cifar_dataloader():
    """Test the CIFAR dataloader functionality"""
    
    print("=== CIFAR DATALOADER TEST ===\n")
    
    # Test the dataloader
    print("Loading CIFAR data...")
    data, taskcla, size = cifar.get(seed=42, path='../dat/')
    print(f"Data loaded successfully!")
    print(f"Image size: {size}")
    print(f"Number of tasks: {len(taskcla)}")
    print(f"Total classes: {data['ncla']}\n")
    
    # Check task structure
    print("Task structure:")
    for task_id, n_classes in taskcla:
        print(f"  Task {task_id}: {data[task_id]['name']} - {n_classes} classes")
        print(f"    Train samples: {data[task_id]['train']['x'].shape[0]}")
        print(f"    Test samples: {data[task_id]['test']['x'].shape[0]}")
        print(f"    Valid samples: {data[task_id]['valid']['x'].shape[0]}")
        print(f"    Unique labels: {torch.unique(data[task_id]['train']['y']).tolist()}")
        print()
    
    # Verify CIFAR-10 task (task 0)
    print("CIFAR-10 Task (Task 0):")
    task0 = data[0]
    print(f"  Name: {task0['name']}")
    print(f"  Classes: {task0['ncla']}")
    print(f"  Train labels: {torch.unique(task0['train']['y']).tolist()}")
    print(f"  Test labels: {torch.unique(task0['test']['y']).tolist()}")
    print(f"  Expected: 0-9")
    assert task0['ncla'] == 10, f"CIFAR-10 should have 10 classes, got {task0['ncla']}"
    assert set(torch.unique(task0['train']['y']).tolist()) == set(range(10)), "CIFAR-10 should have labels 0-9"
    print("✓ CIFAR-10 task is correct!\n")
    
    # Verify binary CIFAR-100 tasks (tasks 1-50)
    print("Binary CIFAR-100 Tasks (Tasks 1-50):")
    binary_tasks = [data[i] for i in range(1, 51)]
    print(f"  Number of binary tasks: {len(binary_tasks)}")
    
    # Check that all binary tasks have exactly 2 classes
    for i, task in enumerate(binary_tasks, 1):
        assert task['ncla'] == 2, f"Task {i} should have 2 classes, got {task['ncla']}"
        assert set(torch.unique(task['train']['y']).tolist()) == {0, 1}, f"Task {i} should have binary labels 0,1"
        assert set(torch.unique(task['test']['y']).tolist()) == {0, 1}, f"Task {i} should have binary labels 0,1"
    
    print("✓ All binary tasks have exactly 2 classes with labels 0,1!\n")
    
    # Check data shapes and types
    print("Data shape verification:")
    print("Note: CIFAR uses ImageNet normalization (mean=[125.3/255, 123.0/255, 113.9/255], std=[63.0/255, 62.1/255, 66.7/255])")
    print("This produces values outside the [0,1] range, which is normal for ImageNet-normalized data.\n")
    for task_id in range(51):
        task = data[task_id]
        for split in ['train', 'test', 'valid']:
            x = task[split]['x']
            y = task[split]['y']
            
            # Check shapes
            assert x.shape[1:] == tuple(size), f"Task {task_id} {split}: expected shape {size}, got {x.shape[1:]}"
            assert x.shape[0] == y.shape[0], f"Task {task_id} {split}: x and y should have same number of samples"
            
            # Check types
            assert x.dtype == torch.float32, f"Task {task_id} {split}: x should be float32, got {x.dtype}"
            assert y.dtype == torch.long, f"Task {task_id} {split}: y should be long, got {y.dtype}"
            
            # Check value ranges
            # CIFAR uses ImageNet normalization, so values can be outside [0,1]
            # Check that values are reasonable (not extreme outliers)
            assert x.min() > -5 and x.max() < 5, f"Task {task_id} {split}: x values seem unreasonable (min={x.min():.3f}, max={x.max():.3f})"
            assert y.min() >= 0, f"Task {task_id} {split}: y should be non-negative"
    
    print("✓ All data shapes and types are correct!\n")
    
    # Check that all CIFAR-100 classes are used exactly once
    print("CIFAR-100 class coverage verification:")
    
    # Load the mapping to see which classes are used
    import json
    map_path = '../dat/binary_cifar/cifar100_binary_map.json'
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            mapping = json.load(f)
        
        used_classes = set()
        for class_id, info in mapping['map'].items():
            used_classes.add(int(class_id))
        
        print(f"  Classes used: {len(used_classes)}")
        print(f"  Expected: 100")
        assert len(used_classes) == 100, f"Should use all 100 CIFAR-100 classes, got {len(used_classes)}"
        assert used_classes == set(range(100)), "Should use classes 0-99"
        print("✓ All 100 CIFAR-100 classes are used exactly once!\n")
    else:
        print("  Mapping file not found, skipping class coverage check\n")
    
    # Summary statistics
    print("=== DATALOADER SUMMARY ===")
    print(f"Total tasks: {len(taskcla)}")
    print(f"Total classes: {data['ncla']}")
    print(f"Image size: {size}")
    print()
    
    print("Task breakdown:")
    print(f"  CIFAR-10 task: 1 task with 10 classes")
    print(f"  CIFAR-100 binary tasks: 50 tasks with 2 classes each")
    print(f"  Total: 1 + 50 = 51 tasks, 10 + (50 × 2) = 110 classes")
    print()
    
    print("Data splits:")
    total_train = sum(data[i]['train']['x'].shape[0] for i in range(51))
    total_test = sum(data[i]['test']['x'].shape[0] for i in range(51))
    total_valid = sum(data[i]['valid']['x'].shape[0] for i in range(51))
    print(f"  Train samples: {total_train:,}")
    print(f"  Test samples: {total_test:,}")
    print(f"  Valid samples: {total_valid:,}")
    print()
    
    print("✓ DATALOADER TEST PASSED! ✓")
    
    return True

if __name__ == "__main__":
    try:
        test_cifar_dataloader()
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

