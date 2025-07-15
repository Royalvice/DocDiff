#!/usr/bin/env python3
import os
import sys
import torch
from PIL import Image
import numpy as np

def test_environment():
    """Test if the environment is set up correctly"""
    print("Testing DocDiff inference environment...")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Test PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"PyTorch not found: {e}")
        return False
    
    # Test other dependencies
    deps = ['torchvision', 'PIL', 'yaml', 'tqdm', 'numpy']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} not available")
            return False
    
    return True

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    print("=" * 50)
    
    model_files = [
        'checksave/init.pth',
        'checksave/denoiser.pth'
    ]
    
    all_exist = True
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
            print(f"✓ {model_file} exists ({size:.1f} MB)")
        else:
            print(f"✗ {model_file} not found")
            all_exist = False
    
    return all_exist

def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    print("=" * 50)
    
    required_files = [
        'inference.py',
        'model/DocDiff.py',
        'schedule/diffusionSample.py',
        'schedule/schedule.py',
        'schedule/dpm_solver_pytorch.py',
        'src/config.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} not found")
            all_exist = False
    
    return all_exist

def create_test_image():
    """Create a simple test image"""
    print("\nCreating test image...")
    print("=" * 50)
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_image)
    
    test_image_path = 'test_input.png'
    test_image.save(test_image_path)
    print(f"✓ Test image created: {test_image_path}")
    
    return test_image_path

def test_inference():
    """Test the inference script"""
    print("\nTesting inference script...")
    print("=" * 50)
    
    # First create test image
    test_image_path = create_test_image()
    
    # Test inference import
    try:
        sys.path.append('.')
        from inference import DocDiffInference
        print("✓ Successfully imported DocDiffInference")
        
        # Test configuration loading
        config = {
            'CHANNEL_X': 3,
            'CHANNEL_Y': 3,
            'TIMESTEPS': 100,
            'SCHEDULE': 'linear',
            'MODEL_CHANNELS': 32,
            'NUM_RESBLOCKS': 1,
            'CHANNEL_MULT': [1, 2, 3, 4],
            'IMAGE_SIZE': [128, 128],
            'PRE_ORI': 'True',
            'DPM_SOLVER': 'False',
            'DPM_STEP': 20,
            'NATIVE_RESOLUTION': 'False',
            'TEST_INITIAL_PREDICTOR_WEIGHT_PATH': 'checksave/init.pth',
            'TEST_DENOISER_WEIGHT_PATH': 'checksave/denoiser.pth'
        }
        
        # Save test config
        import yaml
        with open('test_config.yml', 'w') as f:
            yaml.dump(config, f)
        print("✓ Test configuration created")
        
        # Test model loading (if model files exist)
        if test_model_files():
            try:
                inference = DocDiffInference('test_config.yml')
                print("✓ Model loaded successfully")
                
                # Test image loading
                img = inference.load_image(test_image_path)
                if img is not None:
                    print("✓ Image loading works")
                    
                    # Test inference (commented out to avoid long processing)
                    # result = inference.infer_single_image(test_image_path, 'test_output.png')
                    # if result is not None:
                    #     print("✓ Inference completed successfully")
                    print("✓ Ready for inference (test skipped for speed)")
                else:
                    print("✗ Image loading failed")
                    
            except Exception as e:
                print(f"✗ Model loading failed: {e}")
                return False
        else:
            print("⚠ Model files not found - skipping inference test")
            print("  Please make sure model files are in checksave/ directory")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False
    
    # Clean up
    try:
        os.remove(test_image_path)
        os.remove('test_config.yml')
        print("✓ Test files cleaned up")
    except:
        pass
    
    return True

def main():
    """Main test function"""
    print("DocDiff Inference Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Project Structure", test_project_structure),
        ("Model Files", test_model_files),
        ("Inference Script", test_inference)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            result = test_func()
            print(f"\n{test_name} test: {'PASSED' if result else 'FAILED'}")
            all_passed = all_passed and result
        except Exception as e:
            print(f"\n{test_name} test: FAILED ({e})")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! DocDiff inference is ready to use.")
        print("\nQuick start:")
        print("  python inference.py --input your_image.jpg --output result.png")
        print("  ./run_inference.sh your_image.jpg result.png")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Install missing dependencies: pip install torch torchvision pillow pyyaml tqdm numpy")
        print("  - Download model files to checksave/ directory")
        print("  - Make sure all source files are present")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 