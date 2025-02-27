import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, MyDataset
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',dest="filename", metavar='FILE'
                    ,help =  'path to the config file',default='configs/vae.yaml')
parser.add_argument('-r', '--train_dataset', type=str, help='Dataset to use for training')
parser.add_argument('-t', '--test_dataset', type=str, help='Dataset to use for testing')
parser.add_argument('-d', '--latent_dim', type=int, help='Latent dimension of the model. If provided, it will override the value in the config file')
parser.add_argument('-p', '--trained_model_path', type=str, help='Path to the checkpoint to use for testing. If provided, training will be skipped')
parser.add_argument('-k', '--kl_penalty', type=float, help='KL penalty to use for training. If provided, it will override the value in the config file')
parser.add_argument('-s', '--save_samples', action='store_true', help='Save generated samples in addition to reconstructions', default=False)
parser.add_argument('-o', '--test_output_dir', type=str, help='Where to save the output images from test', default='test_outputs')


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Setup up parameters
config['exp_params']['save_samples'] = args.save_samples
if args.latent_dim is not None:
    config['model_params']['latent_dim'] = args.latent_dim
if args.kl_penalty is not None:
    config['exp_params']['kld_weight'] = args.kl_penalty
exp_name = f"{config['logging_params']['name']}-{config['model_params']['latent_dim']}-kl_{config['exp_params']['kld_weight']}"
if not args.trained_model_path is None:
    exp_name += f"-train_{args.train_dataset}"
if args.test_dataset is not None:
    exp_name += f"-test_{args.test_dataset}"
config['exp_params']['test_output_dir'] = os.path.join(args.test_output_dir, exp_name)

# Set up main stuff
tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=exp_name)
Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)

seed_everything(config['exp_params']['manual_seed'], True)
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,config['exp_params'])
data = VAEDataset(**config["data_params"], 
                 pin_memory=len(config['trainer_params']['gpus']) != 0, 
                 train_dataset=args.train_dataset,
                 test_dataset=args.test_dataset,
                 test_batch_size=16)  # Adjust test batch size as needed
data.setup()

runner = Trainer(logger=tb_logger,
                callbacks=[
                    LearningRateMonitor(),
                    ModelCheckpoint(save_top_k=2, 
                                   dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), 
                                   monitor="val_loss",
                                   save_last=True),
                ],
                strategy=DDPPlugin(find_unused_parameters=False),
                **config['trainer_params'])


if args.trained_model_path is None:
    # No model checkpoint provided, train the model
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)
    
    # After training, test the model on the test dataset if provided
    if args.test_dataset is not None:
        print(f"======= Testing {config['model_params']['name']} on {args.test_dataset} =======")
        # Use the last checkpoint for testing
        checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints", "last.ckpt")
        runner.test(experiment, datamodule=data, ckpt_path=checkpoint_path)
else:
    print(f"======= Testing {config['model_params']['name']} using checkpoint {args.trained_model_path} =======")
    runner.test(experiment, datamodule=data, ckpt_path=args.trained_model_path)
