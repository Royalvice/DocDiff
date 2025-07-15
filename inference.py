#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
import glob

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.DocDiff import DocDiff
from schedule.schedule import Schedule
from schedule.diffusionSample import GaussianDiffusion
from schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

class DocDiffInference:
    def __init__(self, config_path='conf.yml'):
        """Initialize DocDiff inference with configuration"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.setup_model()
        
        # Setup image transformations
        self.setup_transforms()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model(self):
        """Setup the DocDiff model and load pretrained weights"""
        # Model parameters
        in_channels = self.config['CHANNEL_X'] + self.config['CHANNEL_Y']
        out_channels = self.config['CHANNEL_Y']
        
        # Initialize model
        self.model = DocDiff(
            input_channels=in_channels,
            output_channels=out_channels,
            n_channels=self.config['MODEL_CHANNELS'],
            ch_mults=self.config['CHANNEL_MULT'],
            n_blocks=self.config['NUM_RESBLOCKS']
        ).to(self.device)
        
        # Initialize diffusion
        self.schedule = Schedule(self.config['SCHEDULE'], self.config['TIMESTEPS'])
        self.diffusion = GaussianDiffusion(
            self.model.denoiser, 
            self.config['TIMESTEPS'], 
            self.schedule
        ).to(self.device)
        
        # Load pretrained weights
        self.load_pretrained_weights()
        self.model.eval()
        
    def load_pretrained_weights(self):
        """Load pretrained model weights"""
        # Default paths for pretrained models
        init_predictor_path = self.config.get('TEST_INITIAL_PREDICTOR_WEIGHT_PATH', 'checksave/init.pth')
        denoiser_path = self.config.get('TEST_DENOISER_WEIGHT_PATH', 'checksave/denoiser.pth')
        
        # Load weights
        try:
            self.model.init_predictor.load_state_dict(torch.load(init_predictor_path, map_location=self.device))
            self.model.denoiser.load_state_dict(torch.load(denoiser_path, map_location=self.device))
            print(f"Model loaded successfully from {init_predictor_path} and {denoiser_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Please check the model paths in configuration")
            sys.exit(1)
    
    def setup_transforms(self):
        """Setup image transformations"""
        self.image_size = self.config['IMAGE_SIZE']
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def crop_for_inference(self, img, size=128):
        """Crop image for inference when using native resolution"""
        shape = img.shape
        correct_shape = (size*(shape[2]//size+1), size*(shape[3]//size+1))
        one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
        one[:, :, :shape[2], :shape[3]] = img
        
        # Crop into patches
        patches = []
        for i in range(shape[2]//size+1):
            for j in range(shape[3]//size+1):
                patch = one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]
                patches.append(patch)
        
        return torch.cat(patches, dim=0)
    
    def reconstruct_from_patches(self, img, prediction, size=128):
        """Reconstruct image from patches"""
        shape = img.shape
        batch_size = shape[0]
        
        rows = []
        for i in range(shape[2]//size+1):
            row_patches = []
            for j in range(shape[3]//size+1):
                patch_idx = i*(shape[3]//size+1) + j
                patch = prediction[patch_idx*batch_size:(patch_idx+1)*batch_size, :, :, :]
                row_patches.append(patch)
            row = torch.cat(row_patches, dim=3)
            rows.append(row)
        
        result = torch.cat(rows, dim=2)
        return result[:, :, :shape[2], :shape[3]]
    
    def dpm_solver_inference(self, noisy_image, condition):
        """Run DPM solver for inference"""
        def model_fn(x_t, t_input):
            return self.model.denoiser(torch.cat((x_t, condition), dim=1), t_input)
        
        # Setup DPM solver
        betas = self.schedule.get_betas()
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
        model_wrapper_fn = model_wrapper(
            model_fn,
            noise_schedule,
            model_type="noise",
            model_kwargs={},
        )
        
        dpm_solver = DPM_Solver(
            model_wrapper_fn, 
            noise_schedule, 
            algorithm_type="dpmsolver++",
            correcting_x0_fn="dynamic_thresholding"
        )
        
        # Sample
        x_sample = dpm_solver.sample(
            noisy_image,
            steps=self.config.get('DPM_STEP', 20),
            order=1,
            skip_type="time_uniform",
            method="singlestep",
        )
        
        return x_sample
    
    def normalize_tensor(self, tensor):
        """Normalize tensor to [0,1] range"""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    def infer_single_image(self, image_path, output_path=None, save_intermediate=False):
        """Perform inference on a single image"""
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return None
            
        img = img.to(self.device)
        
        # Handle native resolution if needed
        native_resolution = self.config.get('NATIVE_RESOLUTION', 'False') == 'True'
        if native_resolution:
            original_img = img
            img = self.crop_for_inference(img)
        
        with torch.no_grad():
            # Generate random noise
            noisy_image = torch.randn_like(img).to(self.device)
            
            # Get initial prediction
            init_predict = self.model.init_predictor(img, torch.tensor([0]).to(self.device))
            
            # Run diffusion sampling
            use_dpm = self.config.get('DPM_SOLVER', 'False') == 'True'
            if use_dpm:
                sampled_imgs = self.dpm_solver_inference(noisy_image, init_predict)
            else:
                pre_ori = self.config.get('PRE_ORI', 'True')
                sampled_imgs = self.diffusion(noisy_image, init_predict, pre_ori)
            
            # Combine results
            final_imgs = sampled_imgs + init_predict
            
            # Reconstruct if using native resolution
            if native_resolution:
                final_imgs = self.reconstruct_from_patches(original_img, final_imgs)
                init_predict = self.reconstruct_from_patches(original_img, init_predict)
                sampled_imgs = self.reconstruct_from_patches(original_img, sampled_imgs)
                img = original_img
        
        # Save results
        if output_path:
            if save_intermediate:
                # Save concatenated results (input, initial_pred, sampled, final)
                combined = torch.cat([
                    img.cpu(),
                    init_predict.cpu(),
                    self.normalize_tensor(sampled_imgs.cpu()),
                    final_imgs.cpu()
                ], dim=3)
                save_image(combined, output_path, nrow=1)
            else:
                # Save only final result
                save_image(final_imgs.cpu(), output_path, nrow=1)
        
        return final_imgs.cpu()
    
    def infer_batch(self, input_dir, output_dir, save_intermediate=False):
        """Perform batch inference on all images in a directory"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        if not image_paths:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_paths)} images for inference")
        
        # Process each image
        for image_path in tqdm(image_paths, desc="Processing images"):
            image_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_name)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_enhanced.png")
            
            try:
                self.infer_single_image(image_path, output_path, save_intermediate)
                print(f"Processed: {image_name}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="DocDiff Inference Script")
    parser.add_argument('--config', type=str, default='conf.yml', help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True, help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image path or directory')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate results')
    parser.add_argument('--init_model', type=str, default='checksave/init.pth', help='Path to initial predictor model')
    parser.add_argument('--denoiser_model', type=str, default='checksave/denoiser.pth', help='Path to denoiser model')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found, using default parameters")
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
            'NATIVE_RESOLUTION': 'False'
        }
    
    # Set model paths
    config['TEST_INITIAL_PREDICTOR_WEIGHT_PATH'] = args.init_model
    config['TEST_DENOISER_WEIGHT_PATH'] = args.denoiser_model
    
    # Save updated config
    with open('inference_config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize inference
    inference = DocDiffInference('inference_config.yml')
    
    # Run inference
    if os.path.isfile(args.input):
        # Single image inference
        print(f"Processing single image: {args.input}")
        inference.infer_single_image(args.input, args.output, args.save_intermediate)
        print(f"Result saved to: {args.output}")
    elif os.path.isdir(args.input):
        # Batch inference
        print(f"Processing batch images from: {args.input}")
        inference.infer_batch(args.input, args.output, args.save_intermediate)
        print(f"Results saved to: {args.output}")
    else:
        print(f"Input path {args.input} does not exist")
        sys.exit(1)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main() 