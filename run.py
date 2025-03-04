import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
from make_tex import make_tex
from print_epoch_summary import EpochSummary
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
parser.add_argument('--side_by_side_only', action='store_true', help='Only save side-by-side images in testing', default=False)
parser.add_argument('-a', '--annotate_loss', action='store_true', help='Annotate the output images with the loss', default=False)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# print args:
print("Arguments:")
for arg in vars(args):
    print(f"\t{arg}: {getattr(args, arg)}")

# Setup up parameters
if args.train_dataset is None and args.test_dataset is None:
    raise ValueError("At least one of train_dataset and test_dataset must be provided")
config['exp_params']['save_samples'] = args.save_samples
config['exp_params']['side_by_side_only'] = args.side_by_side_only
config['exp_params']['annotate_loss'] = args.annotate_loss
if args.latent_dim is not None:
    config['model_params']['latent_dim'] = args.latent_dim
if args.kl_penalty is not None:
    config['exp_params']['kld_weight'] = args.kl_penalty
exp_name = f"{config['logging_params']['name']}-{config['model_params']['latent_dim']}-kl_{config['exp_params']['kld_weight']}"
if args.trained_model_path is None:
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
                 test_batch_size=16)
data.setup()

runner = Trainer(logger=tb_logger,
                callbacks=[
                    LearningRateMonitor(),
                    ModelCheckpoint(save_top_k=2, 
                                   dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), 
                                   monitor="val_loss",
                                   save_last=True),
                    EpochSummary()
                ],
                strategy=DDPPlugin(find_unused_parameters=False),
                **config['trainer_params'])


if args.trained_model_path is None:
    # No model checkpoint provided, train the model
    print(f"----\nTraining {exp_name}\n----")
    runner.fit(experiment, datamodule=data)
    checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints", "last.ckpt")
else:
    checkpoint_path = args.trained_model_path
if args.test_dataset is not None:
    print(f"----\nTesting {exp_name}\n----")
    runner.test(experiment, datamodule=data, ckpt_path=args.trained_model_path)
    # Make gifs of side-by-side images
    test_output_dir = os.path.join(args.test_output_dir, exp_name) if args.side_by_side_only else os.path.join(args.test_output_dir, exp_name, 'side-by-side')
    make_tex(test_output_dir, exp_name + '.tex')
    # Compile the tex file. For some reason we need to do it twice to make the gifs work
    os.system(f"cd {args.test_output_dir}; pdflatex {exp_name}.tex")
    os.system(f"cd {args.test_output_dir}; pdflatex {exp_name}.tex")    

# Cleanup
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
