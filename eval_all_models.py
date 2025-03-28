import os
import argparse
import subprocess
from pathlib import Path
import re

config_map = {'VanillaVAE':'vae.yaml',
              'MIWAE':'miwae.yaml',
              'DFCVAE':'dfc_vae.yaml',
              'BaseAE':'base_ae.yaml',
              'BigAE':'big_ae.yaml',
              'VGGAE':'vgg_ae.yaml',
              'AdaptiveAE':'adaptive_lr_ae.yaml',
              'PatientAE':'patient_ae.yaml',
              'PatientBigAE':'patient_big_ae.yaml',
              'PatientVBigAE':'patient_vbig_ae.yaml',
              'VPatientBigAE':'vpatient_big_ae.yaml',
              'CenterFocusedAE':'center_focused_ae.yaml',
              'CenterVFocusedAE':'center_vfocused_ae.yaml',
              'CenterVVFocusedAE':'center_vvfocused_ae.yaml',
              'PatientVVBigAE':'patient_vvbig_ae.yaml',
              'MSSIM_AE':'mssim_ae.yaml',
              'DifficultySamplerAE':'difficulty_sampler.yaml',
              'LinearDifficultySamplerAE':'difficulty_sampler.yaml',
              'MSSIMVAE':'mssim_vae.yaml'}

def parse_args():
    parser = argparse.ArgumentParser(description='Test all trained VAE models on specified datasets')
    parser.add_argument('-t', '--test_datasets', nargs='+', required=True,
                        help='List of test datasets to evaluate models on')
    parser.add_argument('-m', '--models_dir', type=str, default='trained_models',
                        help='Path to the directory containing trained models (default: trained_models)')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to the config file. Default is to use whatever config is based on the model name (which is what you should nearly all of the time)')
    parser.add_argument('-n', '--checkpoint_name', type=str, default='best.ckpt',
                        help='Checkpoint filename to use (default: best.ckpt)')
    parser.add_argument('-e', '--extra_image_outputs', action='store_true', help='Output samples and individual reconstructions in addition to side-by-side comparisons', default=False)
    parser.add_argument('-o', '--output_dir', type=str, default='full_test_output', help='Directory to save test outputs')
    parser.add_argument('-u', '--cleanup', action='store_true', help='Delete the output directory after running tests', default=False)
    parser.add_argument('-a', '--dont_annotate_loss', action='store_true', help='Annotate the output images with the loss', default=False)
    parser.add_argument('-g', '--histogram_only', action='store_true', help='Only save histogram images in test', default=False)
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
        checkpoint_path1 = version_dir / "checkpoints" / checkpoint_name
        checkpoint_path2 = version_dir / "checkpoints" / "last.ckpt"
        if checkpoint_path1.exists():
            checkpoint_path = checkpoint_path1
        else:
            print(f"Checkpoint {checkpoint_name} not found for model: {model_dir}. Looking for last.ckpt")
            if checkpoint_path2.exists():
                checkpoint_path = checkpoint_path2
            else:
                print(f"Checkpoint last.ckpt not found for model: {model_dir}. Skipping model")
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

def run_test(model_info, args, test_dataset, idx=None, total=None):
    config_file = args.config if args.config is not None else os.path.join('configs', config_map[model_info['model_type']])
    cmd = [
        "python", "run.py",
        "--config", config_file,
        "--test_dataset", test_dataset,
        "--latent_dim", model_info['latent_dim'],
        "--kl_penalty", model_info['kl_penalty'],
        "--trained_model_path", model_info['checkpoint_path'],
        "--test_output_dir", args.output_dir,
    ]
    if args.extra_image_outputs:
        cmd.append("--extra_image_outputs")
    if args.dont_annotate_loss:
        cmd.append("--dont_annotate_loss")
    if args.histogram_only:
        cmd.append("--histogram_only")
    

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
            success = run_test(model_info=model, args=args, test_dataset=test_dataset, idx=idx, total=len(trained_models) * len(args.test_datasets))
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
    print("Now copying the outputs: gif pdfs and histogram pngs")
    final_output_dir = "output"
    os.makedirs(final_output_dir, exist_ok=True)
    os.system(f"rm {final_output_dir}/*")
    os.system(f"rm output.zip")
    os.system(f"cp {args.output_dir}/*.pdf {final_output_dir}")
    os.system(f"cp {args.output_dir}/*.png {final_output_dir}")
    os.system(f"zip -r output.zip {final_output_dir}")
    if args.cleanup:
        os.system(f"rm -r {args.output_dir}")
        os.system(f"rm -r {final_output_dir}")

if __name__ == "__main__":
    main()
