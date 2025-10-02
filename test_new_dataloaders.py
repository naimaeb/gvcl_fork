#!/usr/bin/env python3
"""
Test script for the new dataloaders: TinyImageNet, CORe50, and Caltech256.
Usage: python test_new_dataloaders.py [dataset_name]
Available datasets: tinyimagenet, core50, caltech256, omniglot, all
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_tinyimagenet(regen=False):
    """Test TinyImageNet dataloader."""
    print("=" * 50)
    print("Testing TinyImageNet Dataloader")
    print("=" * 50)
    
    try:
        from dataloaders.tinyimagenet import get
        
        # Test with default settings (40 tasks, 5 classes per task)
        data, taskcla, size = get(
            seed=42,
            path="./dat/",  # Fixed: use ./dat/ instead of ../dat/
            regen=regen,  # Use the regen parameter
            num_tasks=40,
            classes_per_task=5
        )
        
        print(f"Dataset size: {size}")
        print(f"Number of tasks: {len(taskcla)}")
        print(f"Total classes: {data['ncla']}")
        
        # Print task information
        for task_id, num_classes in taskcla[:5]:  # Show first 5 tasks
            print(f"Task {task_id}: {num_classes} classes, "
                  f"train samples: {data[task_id]['train']['x'].shape[0]}, "
                  f"test samples: {data[task_id]['test']['x'].shape[0]}")
        
        print("TinyImageNet test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing TinyImageNet: {e}")
        return False

def test_core50(regen=False):
    """Test CORe50 dataloader."""
    print("\n" + "=" * 50)
    print("Testing CORe50 Dataloader")
    print("=" * 50)
    
    try:
        from dataloaders.core50 import get
        
        # Test with new-class scenario
        data, taskcla, size = get(
            seed=42,
            path="./dat/",  # Fixed: use ./dat/ instead of ../dat/
            regen=regen,  # Use the regen parameter
            scenario='nc',  # 'nc' (new-class), 'ni' (new-instance), 'nd' (new-domain)
            num_tasks=10
        )
        
        print(f"Dataset size: {size}")
        print(f"Number of tasks: {len(taskcla)}")
        print(f"Total classes: {data['ncla']}")
        
        # Print task information
        for task_id, num_classes in taskcla[:3]:  # Show first 3 tasks
            print(f"Task {task_id}: {num_classes} classes, "
                  f"train samples: {data[task_id]['train']['x'].shape[0]}, "
                  f"test samples: {data[task_id]['test']['x'].shape[0]}")
        
        print("CORe50 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing CORe50: {e}")
        return False

def test_cifar100(regen=False):
    """Test CIFAR-100 dataloader."""
    print("\n" + "=" * 50)
    print("Testing CIFAR-100 Dataloader")
    print("=" * 50)
    
    try:
        from dataloaders.cifar100 import get
        
        # Test with CIFAR-100 (20 tasks, 5 classes per task)
        data, taskcla, size = get(
            seed=42,
            path="./dat/",  # Fixed: use ./dat/ instead of ../dat/
            regen=regen,  # Use the regen parameter
            num_tasks=20,
            classes_per_task=5
        )
        
        print(f"Dataset size: {size}")
        print(f"Number of tasks: {len(taskcla)}")
        print(f"Total classes: {data['ncla']}")
        
        # Print task information
        for task_id, num_classes in taskcla[:3]:  # Show first 3 tasks
            print(f"Task {task_id}: {num_classes} classes, "
                  f"train samples: {data[task_id]['train']['x'].shape[0]}, "
                  f"test samples: {data[task_id]['test']['x'].shape[0]}")
        
        print("CIFAR-100 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing CIFAR-100: {e}")
        return False

def test_caltech256(regen=False):
    """Test Caltech256 dataloader."""
    print("\n" + "=" * 50)
    print("Testing Caltech256 Dataloader")
    print("=" * 50)
    
    try:
        from dataloaders.caltech256 import get
        
        # Test with Caltech256 (50 tasks, 5 classes per task)
        data, taskcla, size = get(
            seed=42,
            path="./dat/",  # Fixed: use ./dat/ instead of ../dat/
            regen=regen,  # Use the regen parameter
            num_tasks=50,
            classes_per_task=5
        )
        
        print(f"Dataset size: {size}")
        print(f"Number of tasks: {len(taskcla)}")
        print(f"Total classes: {data['ncla']}")
        
        # Print task information
        for task_id, num_classes in taskcla[:3]:  # Show first 3 tasks
            print(f"Task {task_id}: {num_classes} classes, "
                  f"train samples: {data[task_id]['train']['x'].shape[0]}, "
                  f"test samples: {data[task_id]['test']['x'].shape[0]}")
        
        print("Caltech256 test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing Caltech256: {e}")
        return False

def test_omniglot(regen=False):
    """Test Omniglot dataloader."""
    print("\n" + "=" * 50)
    print("Testing Omniglot Dataloader")
    print("=" * 50)
    
    try:
        from dataloaders.omniglot import get
        
        # Test with Omniglot (50 tasks)
        data, taskcla, size = get(
            seed=42,
            path="./dat/",
            regen=regen,  # Use the regen parameter
            num_tasks=50
        )
        
        print(f"Dataset size: {size}")
        print(f"Number of tasks: {len(taskcla)}")
        print(f"Total classes: {data['ncla']}")
        
        # Print task information
        for task_id, num_classes in taskcla[:3]:  # Show first 3 tasks
            print(f"Task {task_id}: {num_classes} classes, "
                  f"train samples: {data[task_id]['train']['x'].shape[0]}, "
                  f"test samples: {data[task_id]['test']['x'].shape[0]}")
        
        print("Omniglot test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing Omniglot: {e}")
        return False

def test_inaturalist(regen=False):
    """Test iNaturalist dataloader."""
    print("\n" + "=" * 50)
    print("Testing iNaturalist Dataloader")
    print("=" * 50)
    
    try:
        from dataloaders.inat50 import get
        
        # Test with iNaturalist (50 tasks)
        data, taskcla, size = get(
            seed=42,
            path="./dat/",
            num_tasks=50,
            permute_tasks=True
        )
        
        print(f"Dataset size: {size}")
        print(f"Number of tasks: {len(taskcla)}")
        print(f"Total classes: {data['ncla']}")
        
        # Print task information
        for task_id, num_classes in taskcla[:3]:  # Show first 3 tasks
            print(f"Task {task_id}: {num_classes} classes, "
                  f"train samples: {data[task_id]['train']['x'].shape[0]}, "
                  f"test samples: {data[task_id]['test']['x'].shape[0]}")
        
        print("iNaturalist test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing iNaturalist: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Run tests for specified datasets."""
    parser = argparse.ArgumentParser(description='Test dataloaders for continual learning')
    parser.add_argument('dataset', nargs='?', default='all', 
                       choices=['tinyimagenet', 'core50', 'cifar100', 'caltech256', 'omniglot', 'inaturalist', 'all'],
                       help='Dataset to test (default: all)')
    parser.add_argument('--regen', action='store_true', 
                       help='Regenerate binary files')
    
    # Add usage examples to help
    parser.epilog = """
Examples:
  python test_new_dataloaders.py                    # Test all datasets
  python test_new_dataloaders.py omniglot           # Test only Omniglot
  python test_new_dataloaders.py caltech256 --regen # Test Caltech256 and regenerate files
  python test_new_dataloaders.py tinyimagenet       # Test only TinyImageNet
  python test_new_dataloaders.py core50             # Test only CORe50
  python test_new_dataloaders.py inaturalist        # Test only iNaturalist
"""
    
    args = parser.parse_args()
    
    # Available test functions
    test_functions = {
        'tinyimagenet': test_tinyimagenet,
        'core50': test_core50,
        'cifar100': test_cifar100,
        'caltech256': test_caltech256,
        'omniglot': test_omniglot,
        'inaturalist': test_inaturalist
    }
    
    if args.dataset == 'all':
        datasets_to_test = list(test_functions.keys())
    else:
        datasets_to_test = [args.dataset]
    
    print("Testing Dataloaders for Continual Learning")
    print("=" * 60)
    print(f"Testing: {', '.join(datasets_to_test)}")
    if args.regen:
        print("Regenerating binary files...")
    print("=" * 60)
    
    results = []
    test_names = []
    
    for dataset in datasets_to_test:
        if dataset in test_functions:
            result = test_functions[dataset](regen=args.regen)
            results.append(result)
            test_names.append(dataset.title())
        else:
            print(f"Unknown dataset: {dataset}")
            results.append(False)
            test_names.append(dataset.title())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in zip(test_names, results):
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\nAll tested dataloaders are ready to use!")
        print("\nUsage examples:")
        print("1. TinyImageNet: 40 tasks of 5 classes each")
        print("2. CORe50: 10 tasks with different scenarios (nc/ni/nd)")
        print("3. CIFAR-100: 20 tasks of 5 classes each (100 classes total)")
        print("4. Caltech256: 50 tasks of 5 classes each (250 classes total)")
        print("5. Omniglot: 50 tasks with different alphabets")
        print("6. iNaturalist: 50 tasks with natural species classification")
    else:
        print("\nPlease check the error messages above and ensure datasets are available.")

if __name__ == "__main__":
    main()
