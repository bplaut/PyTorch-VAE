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
        """
        Generate and save sample reconstructions and random samples during training
        using the validation dataset, not the test dataset.
        """
        val_input, val_label = next(iter(self.trainer.datamodule.val_dataloader()))
        val_input = val_input.to(self.curr_device)
        val_label = val_label.to(self.curr_device)

        recons = self.model.generate(val_input, labels=val_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir, 
                                       "reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        if self.params['save_samples']:
            samples = self.model.sample(144,
                                       self.curr_device,
                                       labels=val_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir, 
                                           "samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)

    def test_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        test_loss = self.model.loss_function(*results,
                                           M_N=1.0,
                                           optimizer_idx=0,
                                           batch_idx=batch_idx)

        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

        # Process each image individually
        recons = results[0]  # Reconstructed images from model output

        # Create save directories if they don't exist
        original_dir = os.path.join(self.params['test_output_dir'], "originals")
        recon_dir = os.path.join(self.params['test_output_dir'], "reconstructions")
        comparison_dir = os.path.join(self.params['test_output_dir'], "side-by-side")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)

        # Save each image individually
        for i in range(real_img.size(0)):
            img_idx = batch_idx * real_img.size(0) + i

            # Save original image
            original = real_img[i].unsqueeze(0)  # Add batch dimension
            vutils.save_image(original.data,
                             os.path.join(original_dir, f"original_{img_idx}.png"),
                             normalize=True)

            # Save reconstructed image
            reconstruction = recons[i].unsqueeze(0)  # Add batch dimension
            vutils.save_image(reconstruction.data,
                             os.path.join(recon_dir, f"recon_{img_idx}.png"),
                             normalize=True)

            # Create side-by-side comparison
            # Pad images if dimensions don't match perfectly (avoid potential errors)
            max_height = max(original.size(2), reconstruction.size(2))
            max_width = max(original.size(3), reconstruction.size(3))

            padded_original = torch.nn.functional.pad(
                original, 
                (0, max(0, max_width - original.size(3)), 0, max(0, max_height - original.size(2)))
            )

            padded_recon = torch.nn.functional.pad(
                reconstruction, 
                (0, max(0, max_width - reconstruction.size(3)), 0, max(0, max_height - reconstruction.size(2)))
            )

            # Concatenate horizontally (along width dimension)
            comparison = torch.cat([padded_original, padded_recon], dim=3)

            vutils.save_image(comparison.data,
                             os.path.join(comparison_dir, f"comparison_{img_idx}.png"),
                             normalize=True)

        return test_loss

    def on_test_end(self):
        """
        Function called at the end of test to generate summary statistics
        """
        print("Test completed!")
        print(f"Individual original images saved to: {os.path.join(self.params['test_output_dir'], 'originals')}")
        print(f"Individual reconstructed images saved to: {os.path.join(self.params['test_output_dir'], 'reconstructions')}")
        print(f"Side-by-side comparisons saved to: {os.path.join(self.params['test_output_dir'], 'side-by-side')}")

        # Generate random samples from the latent space
        if self.params['save_samples']:
            try:
                test_data = next(iter(self.trainer.datamodule.test_dataloader()))
                test_input, test_label = test_data
                test_label = test_label.to(self.curr_device)

                samples_dir = os.path.join(self.params['test_output_dir'], "samples")
                os.makedirs(samples_dir, exist_ok=True)

                # Generate samples
                with torch.no_grad():
                    samples = self.model.sample(64, self.curr_device, labels=test_label)

                # Save individual samples
                for i in range(samples.size(0)):
                    sample = samples[i].unsqueeze(0)  # Add batch dimension
                    vutils.save_image(sample.cpu().data,
                                     os.path.join(samples_dir, f"sample_{i}.png"),
                                     normalize=True)

                print(f"Random samples from latent space saved to: {samples_dir}")

            except Exception as e:
                print(f"Could not generate random samples: {e}")
        
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
