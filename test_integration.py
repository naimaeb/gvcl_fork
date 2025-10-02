#!/usr/bin/env python3
"""
Test script to verify integration of new datasets with the existing framework.
This script tests the setup.py configuration and model compatibility.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_setup_integration():
    """Test that setup.py can properly import and configure new datasets."""
    print("=" * 60)
    print("Testing Setup Integration")
    print("=" * 60)
    
    try:
        import setup
        
        # Test argument parsing for new datasets
        test_args = [
            '--experiment', 'tinyimagenet', '--approach', 'gvcl', '--reg_type', 'kl_g'
        ]
        
        # Mock sys.argv for testing
        original_argv = sys.argv
        sys.argv = ['test_integration.py'] + test_args
        
        try:
            args, network, approach, dataloader = setup.get_args()
            print(f"✓ Setup successful for TinyImageNet")
            print(f"  - Network: {network.__name__}")
            print(f"  - Approach: {approach.__name__}")
            print(f"  - Dataloader: {dataloader.__name__}")
        except Exception as e:
            print(f"✗ Setup failed for TinyImageNet: {e}")
            return False
        finally:
            sys.argv = original_argv
        
        # Test CORe50
        test_args = [
            '--experiment', 'core50', '--approach', 'gvcl', '--reg_type', 'kl_g'
        ]
        sys.argv = ['test_integration.py'] + test_args
        
        try:
            args, network, approach, dataloader = setup.get_args()
            print(f"✓ Setup successful for CORe50")
            print(f"  - Network: {network.__name__}")
            print(f"  - Approach: {approach.__name__}")
            print(f"  - Dataloader: {dataloader.__name__}")
        except Exception as e:
            print(f"✗ Setup failed for CORe50: {e}")
            return False
        finally:
            sys.argv = original_argv
        
        # Test ImageNet-R
        test_args = [
            '--experiment', 'imagenet-r', '--approach', 'gvcl', '--reg_type', 'kl_g'
        ]
        sys.argv = ['test_integration.py'] + test_args
        
        try:
            args, network, approach, dataloader = setup.get_args()
            print(f"✓ Setup successful for ImageNet-R")
            print(f"  - Network: {network.__name__}")
            print(f"  - Approach: {approach.__name__}")
            print(f"  - Dataloader: {dataloader.__name__}")
        except Exception as e:
            print(f"✗ Setup failed for ImageNet-R: {e}")
            return False
        finally:
            sys.argv = original_argv
        
        return True
        
    except Exception as e:
        print(f"✗ Setup integration test failed: {e}")
        return False

def test_model_compatibility():
    """Test that new models are compatible with the framework."""
    print("\n" + "=" * 60)
    print("Testing Model Compatibility")
    print("=" * 60)
    
    try:
        from networks.gvcl_models import (
            TinyImageNetNetNoFiLM, TinyImageNetNetFiLM,
            CORe50NetNoFiLM, CORe50NetFiLM,
            ImageNetNetNoFiLM, ImageNetNetFiLM
        )
        
        # Test TinyImageNet models
        print("Testing TinyImageNet models...")
        taskcla = [(0, 5), (1, 5), (2, 5)]  # 3 tasks, 5 classes each
        inputsize = (3, 64, 64)
        
        # Test NoFiLM model
        model = TinyImageNetNetNoFiLM.Net(inputsize, taskcla)
        print(f"✓ TinyImageNetNetNoFiLM created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test FiLM model
        model = TinyImageNetNetFiLM.Net(inputsize, taskcla)
        print(f"✓ TinyImageNetNetFiLM created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test CORe50 models
        print("\nTesting CORe50 models...")
        taskcla = [(0, 5), (1, 5)]  # 2 tasks, 5 classes each
        inputsize = (3, 128, 128)
        
        # Test NoFiLM model
        model = CORe50NetNoFiLM.Net(inputsize, taskcla)
        print(f"✓ CORe50NetNoFiLM created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test FiLM model
        model = CORe50NetFiLM.Net(inputsize, taskcla)
        print(f"✓ CORe50NetFiLM created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test ImageNet models
        print("\nTesting ImageNet models...")
        taskcla = [(0, 10), (1, 10)]  # 2 tasks, 10 classes each
        inputsize = (3, 224, 224)
        
        # Test NoFiLM model
        model = ImageNetNetNoFiLM.Net(inputsize, taskcla)
        print(f"✓ ImageNetNetNoFiLM created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test FiLM model
        model = ImageNetNetFiLM.Net(inputsize, taskcla)
        print(f"✓ ImageNetNetFiLM created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        return False

def test_dataloader_integration():
    """Test that dataloaders work with the framework."""
    print("\n" + "=" * 60)
    print("Testing Dataloader Integration")
    print("=" * 60)
    
    try:
        # Test TinyImageNet dataloader
        print("Testing TinyImageNet dataloader...")
        from dataloaders.tinyimagenet import get as get_tinyimagenet
        
        # Use small configuration for testing
        data, taskcla, size = get_tinyimagenet(
            seed=42,
            path="../dat/",
            regen=False,
            num_tasks=2,  # Small number for testing
            classes_per_task=10
        )
        
        print(f"✓ TinyImageNet dataloader works")
        print(f"  - Tasks: {len(taskcla)}")
        print(f"  - Image size: {size}")
        print(f"  - Total classes: {data['ncla']}")
        
        # Test CORe50 dataloader
        print("\nTesting CORe50 dataloader...")
        from dataloaders.core50 import get as get_core50
        
        data, taskcla, size = get_core50(
            seed=42,
            path="../dat/",
            regen=False,
            scenario='nc',
            num_tasks=2  # Small number for testing
        )
        
        print(f"✓ CORe50 dataloader works")
        print(f"  - Tasks: {len(taskcla)}")
        print(f"  - Image size: {size}")
        print(f"  - Total classes: {data['ncla']}")
        
        # Test ImageNet dataloader
        print("\nTesting ImageNet dataloader...")
        from dataloaders.imagenet import get as get_imagenet
        
        data, taskcla, size = get_imagenet(
            seed=42,
            path="../dat/",
            regen=False,
            variant='r',
            num_tasks=2,  # Small number for testing
            classes_per_task=10
        )
        
        print(f"✓ ImageNet dataloader works")
        print(f"  - Tasks: {len(taskcla)}")
        print(f"  - Image size: {size}")
        print(f"  - Total classes: {data['ncla']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataloader integration test failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end integration with a small example."""
    print("\n" + "=" * 60)
    print("Testing End-to-End Integration")
    print("=" * 60)
    
    try:
        # Test with TinyImageNet (smallest dataset)
        print("Testing end-to-end with TinyImageNet...")
        
        # Import required modules
        from dataloaders.tinyimagenet import get as get_tinyimagenet
        from networks.gvcl_models import TinyImageNetNetNoFiLM
        import torch
        
        # Load data
        data, taskcla, inputsize = get_tinyimagenet(
            seed=42,
            path="../dat/",
            regen=False,
            num_tasks=2,  # Small number for testing
            classes_per_task=10
        )
        
        # Create model
        net = TinyImageNetNetNoFiLM.Net(inputsize, taskcla)
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(4, *inputsize)  # 4 samples
        task_labels = torch.tensor([0, 0, 1, 1])  # 2 samples per task
        
        # Test forward pass
        outputs = net(dummy_input, task_labels, 'kl_g', 1, False)
        
        print(f"✓ End-to-end test successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shapes: {[out.shape for out in outputs if out is not None]}")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("New Datasets Integration Tests")
    print("=" * 80)
    
    results = []
    
    # Test 1: Setup integration
    results.append(test_setup_integration())
    
    # Test 2: Model compatibility
    results.append(test_model_compatibility())
    
    # Test 3: Dataloader integration
    results.append(test_dataloader_integration())
    
    # Test 4: End-to-end integration
    results.append(test_end_to_end())
    
    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    tests = ["Setup Integration", "Model Compatibility", "Dataloader Integration", "End-to-End"]
    for test_name, result in zip(tests, results):
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Integration successful! New datasets are ready to use.")
        print("\nExample usage:")
        print("python src/run.py --experiment tinyimagenet --approach gvcl --reg_type kl_g --seed 42")
        print("python src/run.py --experiment core50 --approach gvcl --reg_type kl_g --seed 42")
        print("python src/run.py --experiment imagenet-r --approach gvcl --reg_type kl_g --seed 42")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
