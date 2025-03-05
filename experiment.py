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
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.test_output_size = (256,256)
        self.logged_size_adjustment = False
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
                                              M_N = self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['kld_weight'],
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        self.sample_images()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        metrics = self.trainer.callback_metrics
        print("\n" + "-" * 50)
        print(f"Epoch {self.current_epoch}:")
        print(f"Training Total Loss: {metrics['loss']:.5f}")
        if metrics['Reconstruction_Loss'] != metrics['loss']:
            print(f"Training recon loss: {metrics['Reconstruction_Loss']:.5f}")
        if metrics['feature_loss'] != 0:
            print(f"Training feature loss: {metrics['feature_loss']:.5f}")
        print(f"Validation total loss: {metrics['val_loss']:.5f}")
        if metrics['val_Reconstruction_Loss'] != metrics['val_loss']:
            print(f"Validation recon loss: {metrics['val_Reconstruction_Loss']:.5f}")
        if metrics['val_feature_loss'] != 0:
            print(f"Validation feature loss: {metrics['val_feature_loss']:.5f}")
        print(f"Current LR: {self.trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']:.3g}")

        print("-" * 50 + "\n")

    def sample_images(self):
        """
        Generate and save sample reconstructions and random samples during training
        using the validation dataset, not the test dataset.
        """
        val_input, val_label = next(iter(self.trainer.datamodule.val_dataloader()))
        val_input = val_input.to(self.curr_device)
        val_label = val_label.to(self.curr_device)

        recons = self.model.generate(val_input, labels=val_label)
        if len(recons.shape) > 4:
            print(f"Detected non-standard reconstruction shape (expected for MIWAE): {recons.shape}. Using first sample.")
            recons = recons[0]

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

        # Initialize data storage if not already done
        if not hasattr(self, 'test_data'):
            self.test_data = []
            self.loss_stats = {
                'total_loss': {'min': float('inf'), 'max': float('-inf')},
                'recon_loss': {'min': float('inf'), 'max': float('-inf')},
                'feature_loss': {'min': float('inf'), 'max': float('-inf')}
            }
            print("Starting image collection for visualization...")

        # Get batch results for logging
        results = self.forward(real_img, labels=labels)
        test_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],
                                            optimizer_idx=0,
                                            batch_idx=batch_idx)

        # Log batch metrics
        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

        # Process each image and collect data
        for i in range(real_img.size(0)):
            img_idx = batch_idx * real_img.size(0) + i

            # Get individual image
            single_img = real_img[i:i+1]
            single_label = labels[i:i+1] if labels is not None else None

            # Forward pass for single image
            single_results = self.forward(single_img, labels=single_label)
            single_loss = self.model.loss_function(*single_results,
                                               M_N=self.params['kld_weight'],
                                               optimizer_idx=0,
                                               batch_idx=batch_idx)

            # Get reconstructed image
            recons = single_results[0]
            recons = self.ensure_4_dims(recons)

            # Scale losses by 1000 for readability
            total_loss = single_loss['loss'].item() * 1000
            recon_loss = single_loss['Reconstruction_Loss'].item() * 1000
            if 'feature_loss' in single_loss:
                feature_loss = single_loss['feature_loss'].item() * 1000
            else:
                feature_loss = None

            # Update global min/max values
            self.loss_stats['total_loss']['min'] = min(self.loss_stats['total_loss']['min'], total_loss)
            self.loss_stats['total_loss']['max'] = max(self.loss_stats['total_loss']['max'], total_loss)
            self.loss_stats['recon_loss']['min'] = min(self.loss_stats['recon_loss']['min'], recon_loss)
            self.loss_stats['recon_loss']['max'] = max(self.loss_stats['recon_loss']['max'], recon_loss)
            if feature_loss is not None:
                self.loss_stats['feature_loss']['min'] = min(self.loss_stats['feature_loss']['min'], feature_loss)
                self.loss_stats['feature_loss']['max'] = max(self.loss_stats['feature_loss']['max'], feature_loss)

            # Resize images
            original_resized = torch.nn.functional.interpolate(
                single_img, size=self.test_output_size, mode='bilinear', align_corners=False
            )
            reconstruction_resized = torch.nn.functional.interpolate(
                recons, size=self.test_output_size, mode='bilinear', align_corners=False
            )

            # Store data for processing
            self.test_data.append({
                'img_idx': img_idx,
                'original': original_resized.cpu(),
                'reconstruction': reconstruction_resized.cpu(),
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'feature_loss': feature_loss
            })

        return test_loss

    def on_test_end(self):
        """
        Function called at the end of test to save all images
        """
        print("Test completed! Processing images...")

        # Create directories for output
        if not self.params['side_by_side_only']:
            original_dir = os.path.join(self.params['test_output_dir'], "originals")
            recon_dir = os.path.join(self.params['test_output_dir'], "reconstructions")
            comparison_dir = os.path.join(self.params['test_output_dir'], "side-by-side")
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)
        else:
            comparison_dir = self.params['test_output_dir']
        os.makedirs(comparison_dir, exist_ok=True)

        # Print normalization ranges
        print(f"Loss ranges (scaled by 1000):")
        print(f"  Total Loss: {self.loss_stats['total_loss']['min']:.4f} to {self.loss_stats['total_loss']['max']:.4f}")
        print(f"  Recon Loss: {self.loss_stats['recon_loss']['min']:.4f} to {self.loss_stats['recon_loss']['max']:.4f}")
        if self.test_data[0]['feature_loss'] is not None:
            print(f"  Feature Loss: {self.loss_stats['feature_loss']['min']:.4f} to {self.loss_stats['feature_loss']['max']:.4f}")

        # Process all collected data with consistent normalization
        for data in self.test_data:
            img_idx = data['img_idx']
            original = data['original']
            reconstruction = data['reconstruction']
            total_loss = data['total_loss']
            recon_loss = data['recon_loss']
            feature_loss = data['feature_loss']

            # Save individual images if needed
            if not self.params['side_by_side_only']:
                vutils.save_image(original.data,
                                  os.path.join(original_dir, f"{img_idx}.png"),
                                  normalize=True)
                vutils.save_image(reconstruction.data,
                                  os.path.join(recon_dir, f"{img_idx}.png"),
                                  normalize=True)

            # Create side-by-side comparison
            comparison = torch.cat([original, reconstruction], dim=3)

            # Convert to PIL for annotation
            comparison_np = comparison.numpy()
            comparison_np = np.transpose(comparison_np[0], (1, 2, 0))
            comparison_np = (comparison_np - comparison_np.min()) / (comparison_np.max() - comparison_np.min()) * 255.0
            comparison_pil = Image.fromarray(comparison_np.astype(np.uint8))

            # Normalize values using global min/max
            total_norm_loss = self.normalize_loss(total_loss, 'total_loss')
            recon_norm_loss = self.normalize_loss(recon_loss, 'recon_loss')
            feature_norm_loss = self.normalize_loss(feature_loss, 'feature_loss') if feature_loss is not None else None

            # Create the final image
            if self.params['annotate_loss']:
                final_img = self.create_annotated_image(
                    comparison_pil, 
                    total_loss, total_norm_loss,
                    recon_loss, recon_norm_loss, 
                    feature_loss, feature_norm_loss
            )
            else:
                final_img = comparison_pil

            final_img.save(os.path.join(comparison_dir, f"{img_idx}.png"))

        print(f"Saved {len(self.test_data)} annotated images.")
        print(f"Side-by-side comparisons saved to: {comparison_dir}")
        if not self.params['side_by_side_only']:
            print(f"Individual original images saved to: {original_dir}")
            print(f"Individual reconstructed images saved to: {recon_dir}")

        # Generate random samples if requested
        if self.params.get('save_samples', False):
            self.generate_random_samples()

        # Clean up stored data
        delattr(self, 'test_data')
        delattr(self, 'loss_stats')

    def generate_random_samples(self):
        """
        Generate random samples from the latent space
        """
        try:
            test_data = next(iter(self.trainer.datamodule.test_dataloader()))
            test_input, test_label = test_data
            test_label = test_label.to(self.curr_device)

            samples_dir = os.path.join(self.params['test_output_dir'], "samples")
            os.makedirs(samples_dir, exist_ok=True)

            with torch.no_grad():
                samples = self.model.sample(64, self.curr_device, labels=test_label)

            samples = self.ensure_4_dims(samples)

            for i in range(samples.size(0)):
                sample = samples[i:i+1]
                sample_resized = torch.nn.functional.interpolate(
                    sample, size=self.test_output_size, mode='bilinear', align_corners=False
                )
                vutils.save_image(sample_resized.cpu().data,
                                  os.path.join(samples_dir, f"sample_{i}.png"),
                                  normalize=True)

            print(f"Random samples from latent space saved to: {samples_dir}")

        except Exception as e:
            print(f"Could not generate random samples: {e}")

    def create_annotated_image(self, comparison_img, total_loss, total_norm_loss, recon_loss, recon_norm_loss, feature_loss, feature_norm_loss):
        img_width, img_height = comparison_img.size
        header_height = 40

        new_img = Image.new('RGB', (img_width, img_height + header_height), color=(240, 240, 240))
        new_img.paste(comparison_img, (0, header_height))

        draw = ImageDraw.Draw(new_img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

        metrics = [
            {"name": "Total", "value": total_loss, "norm": total_norm_loss, "x": 10},
            {"name": "Recon", "value": recon_loss, "norm": recon_norm_loss, "x": img_width // 3},
            {"name": "Feature", "value": feature_loss, "norm": feature_norm_loss, "x": 2 * img_width // 3}
        ]

        for metric in metrics:
            if metric["value"] is None:
                continue
            x = metric["x"]
            color = self.get_color_from_score(metric["norm"])

            # Display normalized value (0-1)
            text = f"{metric['name']}: {metric['value']:.3f}"
            draw.text((x, 4), text, fill=color, font=font)

            meter_width = img_width // 4
            meter_height = 10
            meter_y = header_height - meter_height - 4

            # Background
            draw.rectangle(
                [(x, meter_y), (x + meter_width, meter_y + meter_height)],
                fill=(220, 220, 220), outline=(180, 180, 180)
            )

            # Filled part
            filled_width = int(meter_width * metric["norm"])
            if filled_width > 0:
                draw.rectangle(
                    [(x, meter_y), (x + filled_width, meter_y + meter_height)],
                    fill=color
                )

        return new_img

    def normalize_loss(self, loss_value, loss_type):
        min_val = self.loss_stats[loss_type]['min']
        max_val = self.loss_stats[loss_type]['max']

        if max_val == min_val:
            return 0.5

        return max(0.0, min(1.0, (loss_value - min_val) / (max_val - min_val)))

    def get_color_from_score(self, score):
        if score < 0.5:
            r = int(128 * (score * 2))
            g = 0
            b = 255
        else:
            r = int(128 + 127 * (score - 0.5) * 2)
            g = 0
            b = int(255 * (1 - (score - 0.5) * 2))

        return (r, g, b)

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
        if 'adaptive_lr' in self.params and self.params['adaptive_lr']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                threshold=0.00003,
                threshold_mode='abs',
                min_lr=1e-7,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            try:
                if self.params['scheduler_gamma'] is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                gamma=self.params['scheduler_gamma'])
                    scheds.append({"scheduler": scheduler, "interval": "epoch"})
                    # Check if another scheduler is required for the second optimizer
                    try:
                        if self.params['scheduler_gamma_2'] is not None:
                            scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                        gamma=self.params['scheduler_gamma_2'])
                            scheds.append({"scheduler": scheduler2, "interval": "epoch"})
                    except:
                        pass
                    return optims, scheds
            except:
                return optims

    def ensure_4_dims(self, t):
        if len(t.shape) != 4:
            if not self.logged_size_adjustment:
                print(f"\nDetected non-standard tensor shape (expected for MIWAE): {t.shape}.")
            # For MIWAE, shape could be [B, _, S, C, H, W] where S is number of samples
            # We want to extract [B, C, H, W] by taking the first sample (index 0 of the samples dimension)
            if len(t.shape) == 6:  # [B, _, S, C, H, W]
                t = t[:, 0, 0]  # Take first element of dimensions 1 and 2
            elif len(t.shape) == 5:  # [B, S, C, H, W]
                t = t[:, 0]  # Take first sample
            elif len(t.shape) == 3:
                t = t.unsqueeze(0)
            if not self.logged_size_adjustment:
                print(f"Adjusted tensor shape to: {t.shape}")
            self.logged_size_adjustment = True
        return t
