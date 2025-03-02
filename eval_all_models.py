import os
import argparse
import subprocess
from pathlib import Path
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Test all trained VAE models on specified datasets')
    parser.add_argument('-t', '--test_datasets', nargs='+', required=True,
                        help='List of test datasets to evaluate models on')
    parser.add_argument('-m', '--models_dir', type=str, default='trained_models',
                        help='Path to the directory containing trained models (default: trained_models)')
    parser.add_argument('-c', '--config_dir', type=str, default='configs',
                        help='Path to the config directory (default: configs)')
    parser.add_argument('-n', '--checkpoint_name', type=str, default='last.ckpt',
                        help='Checkpoint filename to use (default: last.ckpt)')
    parser.add_argument('-s', '--side_by_side_only', action='store_true',help='Only save side-by-side images', default=False)
    parser.add_argument('-o', '--output_dir', type=str, default='test_outputs', help='Directory to save test outputs')
    return parser.parse_args()

def find_trained_models(models_dir, checkpoint_name):
    trained_models = []
    
    # Look directly in models_dir for model directories
    for model_dir in Path(models_dir).iterdir():
        if not model_dir.is_dir():
            continue
            
        # Check if this is a model directory with the right format
        exp_name = model_dir.name
        match = re.match(r'([^-]+)-(\d+)-kl_([\d\.e\-\+]+)-train_(.+)', exp_name)
        if not match:
            continue
            
        model_type, latent_dim, kl_penalty, train_dataset = match.groups()
        
        if "_test_" in train_dataset:
            train_dataset = train_dataset.split("_test_")[0]
        
        # Find version subdirectory (assuming there's only one)
        version_dirs = list(model_dir.glob("version_*"))
        if not version_dirs:
            print(f"No version subdirectory found for model: {model_dir}")
            continue
            
        # Take the most recent version
        version_dir = sorted(version_dirs)[-1]
        
        # Check for checkpoint
        checkpoint_path = version_dir / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            print(f"Checkpoint {checkpoint_name} not found for model: {model_dir}")
            continue
            
        trained_models.append({
            'model_type': model_type,
            'latent_dim': latent_dim,
            'kl_penalty': kl_penalty,
            'train_dataset': train_dataset,
            'checkpoint_path': str(checkpoint_path),
            'exp_dir': str(model_dir)
        })
            
    return trained_models

def run_test(model_info, test_dataset, config_dir, side_by_side_only=False, output_dir='test_outputs', idx=None, total=None):
    config_map = {'VanillaVAE':'vae.yaml',
                  'MIWAE':'miwae.yaml',
                  'DFCVAE':'dfc_vae.yaml',
                  'MSSIMVAE':'mssim_vae.yaml',}
    cmd = [
        "python", "run.py",
        "--config", os.path.join(config_dir, config_map[model_info['model_type']]),
        "--train_dataset", model_info['train_dataset'],
        "--test_dataset", test_dataset,
        "--latent_dim", model_info['latent_dim'],
        "--kl_penalty", model_info['kl_penalty'],
        "--trained_model_path", model_info['checkpoint_path'],
        "--test_output_dir", output_dir,
    ]
    if side_by_side_only:
        cmd.append("--side_by_side_only")
    

    print(f"================\nRUNNING TEST COMMAND {idx+1}/{total}: " + " ".join(cmd) + "\n================")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Test completed successfully for {model_info['model_type']} on {test_dataset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error testing {model_info['model_type']} on {test_dataset}: {e}")
        return False

def main():
    args = parse_args()
    
    print(f"Searching for trained models in {args.models_dir}...")
    trained_models = find_trained_models(args.models_dir, args.checkpoint_name)
    
    if not trained_models:
        print("No trained models found in the logs directory.")
        return
    
    print(f"Found {len(trained_models)} trained models:")
    for i, model in enumerate(trained_models, 1):
        print(f"{i}. {model['model_type']} (latent_dim={model['latent_dim']}, kl_penalty={model['kl_penalty']}) trained on {model['train_dataset']}")
        print(f"   Checkpoint: {model['checkpoint_path']}")
    
    results = []
    idx = 0
    for model in trained_models:
        for test_dataset in args.test_datasets:
            success = run_test(
                model_info=model,
                test_dataset=test_dataset,
                config_dir=args.config_dir,
                side_by_side_only=args.side_by_side_only,
                output_dir=args.output_dir,
                idx=idx,
                total=len(trained_models) * len(args.test_datasets)
            )
            idx += 1
            
            results.append({
                'model_type': model['model_type'],
                'latent_dim': model['latent_dim'],
                'kl_penalty': model['kl_penalty'],
                'train_dataset': model['train_dataset'],
                'test_dataset': test_dataset,
                'success': success
            })
    
    print("\n===== Test Results Summary =====")
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{result['model_type']} (latent_dim={result['latent_dim']}, kl_penalty={result['kl_penalty']}) " +
              f"trained on {result['train_dataset']}, " +
              f"tested on {result['test_dataset']}: {status}")
    print("Now copying the tex pdf outputs...")
    gif_dir = "output_gifs"
    os.makedirs(gif_dir, exist_ok=True)
    os.system(f"rm {gif_dir}/*.pdf")
    os.system(f"rm output_gifs.zip")
    os.system(f"cp {args.output_dir}/*.pdf {gif_dir}")
    os.system(f"zip -r output_gifs.zip {gif_dir}")

if __name__ == "__main__":
    main()
