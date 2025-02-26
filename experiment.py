import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def test_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        test_loss = self.model.loss_function(*results,
                                           M_N=1.0,
                                           optimizer_idx=0,
                                           batch_idx=batch_idx)

        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

        # Save reconstructions for each batch in test set
        if batch_idx % 10 == 0:  # Save every 10th batch to avoid too many images
            recons = results[0]  # Reconstructed images from model output

            # Create the batch save directory if it doesn't exist
            save_dir = os.path.join(self.logger.log_dir, "TestReconstructions")
            os.makedirs(save_dir, exist_ok=True)

            # Save original images
            vutils.save_image(real_img.data,
                            os.path.join(save_dir, f"original_batch_{batch_idx}.png"),
                            normalize=True,
                            nrow=int(math.sqrt(real_img.size(0))))

            # Save reconstructed images
            vutils.save_image(recons.data,
                            os.path.join(save_dir, f"recons_batch_{batch_idx}.png"),
                            normalize=True,
                            nrow=int(math.sqrt(recons.size(0))))

        return test_loss

    def on_test_end(self):
        """
        Function called at the end of test to generate a summary image grid
        containing random samples and their reconstructions
        """
        print("Generating test reconstructions summary...")

        # Create test directory if it doesn't exist
        test_dir = os.path.join(self.logger.log_dir, "TestReconstructions")
        os.makedirs(test_dir, exist_ok=True)

        # Get a batch of test data
        try:
            test_data = next(iter(self.trainer.datamodule.test_dataloader()))
            test_input, test_label = test_data
            test_input = test_input.to(self.curr_device)
            test_label = test_label.to(self.curr_device)

            # Get reconstructions
            with torch.no_grad():
                recons = self.model.generate(test_input, labels=test_label)

            # Create comparison grid: original | reconstruction
            comparison = torch.cat([test_input, recons], dim=0)

            # Save grid image
            vutils.save_image(comparison.cpu().data,
                            os.path.join(test_dir, f"test_reconstructions_summary.png"),
                            normalize=True,
                            nrow=test_input.size(0))

            # Generate some random samples from the latent space
            try:
                samples = self.model.sample(64, self.curr_device, labels=test_label)
                vutils.save_image(samples.cpu().data,
                                os.path.join(test_dir, f"test_random_samples.png"),
                                normalize=True,
                                nrow=8)
            except Exception as e:
                print(f"Could not generate random samples: {e}")

        except Exception as e:
            print(f"Error generating test summary: {e}")
        
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append({"scheduler": scheduler, "interval": "epoch"})
                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append({"scheduler": scheduler2, "interval": "epoch"})
                except:
                    pass
                return optims, scheds
        except:
            return optims
