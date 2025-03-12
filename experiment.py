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
import torch.nn.functional as F
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
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
        self.reset_extreme_image_tracking()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def reset_extreme_image_tracking(self):
        """Reset tracking of extreme loss images for a new training epoch."""
        self.extreme_images = {
            'highest': {'loss': float('-inf'), 'img': None, 'idx': None},
            'lowest': {'loss': float('inf'), 'img': None, 'recon': None, 'idx': None}
        }

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        imgs, labels, indices = batch
        self.curr_device = imgs.device

        results = self.forward(imgs, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # Track per-image losses for difficulty sampling
        recon_img = results[0]
        input_img = results[1]
        per_img_loss = F.mse_loss(recon_img, input_img, reduction='none')    
        # Average over all dimensions except batch
        per_img_loss = per_img_loss.mean(dim=[1, 2, 3])  # Average over channels, height, width        
        self.trainer.datamodule.record_img_losses(indices, per_img_loss.detach().cpu())
        
        # Track images with extreme losses
        for i in range(len(per_img_loss)):
            loss_val = per_img_loss[i].item()
            
            # Check if this is the highest loss image so far
            if loss_val > self.extreme_images['highest']['loss']:
                self.extreme_images['highest'] = {
                    'loss': loss_val,
                    'img': imgs[i:i+1].detach().cpu(),
                    'recon': recon_img[i:i+1].detach().cpu(),
                    'idx': indices[i].item()
                }
            
            # Check if this is the lowest loss image so far
            if loss_val < self.extreme_images['lowest']['loss']:
                self.extreme_images['lowest'] = {
                    'loss': loss_val,
                    'img': imgs[i:i+1].detach().cpu(),
                    'recon': recon_img[i:i+1].detach().cpu(),
                    'idx': indices[i].item()
                }

        return train_loss['loss']

    def on_train_epoch_end(self):
        """
        Save extreme loss images and reset tracking at the end of each epoch.
        """
        # First, let the datamodule perform its end-of-epoch operations
        self.trainer.datamodule.on_epoch_end()
        
        # Create directory for saving comparisons
        comparisons_dir = os.path.join(self.logger.log_dir, "highest_and_lowest_loss_imgs")
        os.makedirs(comparisons_dir, exist_ok=True)
        
        # Save extreme images
        for key in ['highest', 'lowest']:
            data = self.extreme_images[key]
            if data['img'] is not None:
                # Resize to standard output size
                img_resized = F.interpolate(
                    data['img'], size=self.test_output_size, mode='bilinear', align_corners=False
                )
                recon_resized = F.interpolate(
                    data['recon'], size=self.test_output_size, mode='bilinear', align_corners=False
                )
                
                # Scale loss for readability
                loss_val = data['loss'] * 1000
                norm_loss = 0 if key == 'lowest' else 1
                comparison = self.create_side_by_side_image(img_resized, recon_resized, loss_val, norm_loss)
                comparison.save(os.path.join(
                    comparisons_dir, 
                    f"epoch_{self.current_epoch}_{key}_loss_{loss_val:.3f}_idx_{data['idx']}.png"
                ))
        
        self.reset_extreme_image_tracking()

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        imgs, labels, _ = batch
        self.curr_device = imgs.device

        results = self.forward(imgs, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['kld_weight'],
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self):
        self.sample_images()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        metrics = self.trainer.callback_metrics
        print("\n" + "-" * 50)
        print(f"Training total Loss: {metrics['loss']:.5f}")
        # If we only have reconstruction loss, no need to print stuff separately
        if not math.isclose(metrics['Reconstruction_Loss'], metrics['loss']):
            print(f"Training recon loss: {metrics['Reconstruction_Loss']:.5f}")
        if not math.isclose(metrics['feature_loss'], 0):
            print(f"Training feature loss: {metrics['feature_loss']:.5f}")
        print(f"Validation total loss: {metrics['val_loss']:.5f}")
        if not math.isclose(metrics['val_Reconstruction_Loss'], metrics['val_loss']):
            print(f"Validation recon loss: {metrics['val_Reconstruction_Loss']:.5f}")
        if not math.isclose(metrics['val_feature_loss'], 0):
            print(f"Validation feature loss: {metrics['val_feature_loss']:.5f}")
        print(f"Current LR: {self.trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']:.3g}")

        print("-" * 50 + "\n")

    def sample_images(self):
        """
        Generate and save sample reconstructions and random samples during training
        using the validation dataset, not the test dataset.
        """
        val_input, val_label, _ = next(iter(self.trainer.datamodule.val_dataloader()))
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
        imgs, labels, _ = batch
        self.curr_device = imgs.device

        if not hasattr(self, 'test_data'):
            self.test_data = []
            self.loss_stats = {
                'total_loss': {'min': float('inf'), 'max': float('-inf')},
                'recon_loss': {'min': float('inf'), 'max': float('-inf')},
                'feature_loss': {'min': float('inf'), 'max': float('-inf')}
            }
            print("Starting image collection for visualization...")

        results = self.forward(imgs, labels=labels)
        test_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],
                                            optimizer_idx=0,
                                            batch_idx=batch_idx)
        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True)

        # Process each image individually and collect data
        for i in range(imgs.size(0)):
            img_idx = batch_idx * imgs.size(0) + i

            single_img = imgs[i:i+1]
            single_label = labels[i:i+1] if labels is not None else None
            single_results = self.forward(single_img, labels=single_label)
            single_loss = self.model.loss_function(*single_results,
                                               M_N=self.params['kld_weight'],
                                               optimizer_idx=0,
                                               batch_idx=batch_idx)
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

            original_resized = F.interpolate(
                single_img, size=self.test_output_size, mode='bilinear', align_corners=False
            )
            reconstruction_resized = F.interpolate(
                recons, size=self.test_output_size, mode='bilinear', align_corners=False
            )

            self.test_data.append({
                'img_idx': img_idx,
                'original': original_resized.cpu(),
                'reconstruction': reconstruction_resized.cpu(),
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'feature_loss': feature_loss
            })

        return test_loss

    def create_side_by_side_image(self, original, reconstruction, total_loss, total_norm_loss):
        """
        Create a side-by-side comparison of original and reconstructed images with loss annotation.
        
        Args:
            original: Original image tensor
            reconstruction: Reconstructed image tensor
            total_loss: Loss value
            total_norm_loss: Normalized loss (0-1). If None, will be calculated using global min/max if available.
            
        Returns:
            PIL Image with side-by-side comparison and annotations
        """
        # Create side-by-side comparison
        comparison = torch.cat([original, reconstruction], dim=3)
        
        # Convert to PIL for annotation
        comparison_np = comparison.numpy()
        comparison_np = np.transpose(comparison_np[0], (1, 2, 0))
        comparison_np = (comparison_np - comparison_np.min()) / (comparison_np.max() - comparison_np.min()) * 255.0
        comparison_pil = Image.fromarray(comparison_np.astype(np.uint8))
        
        
        # Create annotated image if requested
        if self.params.get('annotate_loss', True):
            final_img = self.create_annotated_image(comparison_pil, total_loss, total_norm_loss)
        else:
            final_img = comparison_pil
        
        return final_img

    def on_test_end(self):
        """
        Function called at the end of test to save all images
        """
        if not self.params['side_by_side_only']:
            original_dir = os.path.join(self.params['test_output_dir'], "originals")
            recon_dir = os.path.join(self.params['test_output_dir'], "reconstructions")
            comparison_dir = os.path.join(self.params['test_output_dir'], "side-by-side")
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)
        else:
            comparison_dir = self.params['test_output_dir']
        os.makedirs(comparison_dir, exist_ok=True)

        print("Saving histogram...")
        self.save_loss_histogram()
        if self.params['histogram_only']:
            print("Skipping image saving as requested.")
            return
        print("Saving reconstructed images...")

        for data in self.test_data:
            img_idx = data['img_idx']
            original = data['original']
            reconstruction = data['reconstruction']
            total_loss = data['total_loss']
            
            # Save individual images if needed
            if not self.params['side_by_side_only']:
                vutils.save_image(original.data,
                                  os.path.join(original_dir, f"{img_idx}.png"),
                                  normalize=True)
                vutils.save_image(reconstruction.data,
                                  os.path.join(recon_dir, f"{img_idx}.png"),
                                  normalize=True)

            # Normalize the loss value
            total_norm_loss = self.normalize_loss(total_loss, 'total_loss')
            
            # Create side-by-side comparison using the helper function
            final_img = self.create_side_by_side_image(original, reconstruction, total_loss, total_norm_loss)
            
            # Save the comparison
            final_img.save(os.path.join(comparison_dir, f"{img_idx}.png"))

        print(f"Saved {len(self.test_data)} annotated images.")
        print(f"Side-by-side comparisons saved to: {comparison_dir}")
        if not self.params['side_by_side_only']:
            print(f"Individual original images saved to: {original_dir}")
            print(f"Individual reconstructed images saved to: {recon_dir}")

        if self.params.get('save_samples', False):
            self.generate_random_samples()

        # Clean up stored data
        delattr(self, 'test_data')
        delattr(self, 'loss_stats')
        
    def save_loss_histogram(self):
        """
        Generate and save a histogram of total reconstruction error across individual frames
        with vertical lines at 50th, 75th, 90th, and 95th percentiles
        """

        # Create directory for histogram
        histogram_dir = os.path.join(self.params['test_output_dir'], "histograms")
        os.makedirs(histogram_dir, exist_ok=True)

        # Extract total losses from test data
        total_losses = [data['total_loss'] for data in self.test_data]

        # Calculate percentiles
        percentiles = [50, 75, 90, 95]
        percentile_values = np.percentile(total_losses, percentiles)

        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(total_losses, bins=40, alpha=0.8, color='blue')
        plt.title('Histogram of loss across time steps')
        plt.xlabel('Loss (x1000)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Add vertical lines for percentiles
        colors = ['green', 'orange', 'purple', 'red']
        for i, (percentile, value) in enumerate(zip(percentiles, percentile_values)):
            plt.axvline(x=value, color=colors[i], linestyle='--', 
                       label=f'{percentile}th percentile: {value:.4f}')

        plt.legend()

        # Save the histogram
        last_sep = self.params['test_output_dir'].rfind('/')
        overall_output_dir = self.params['test_output_dir'][:last_sep]
        exp_name = self.params['test_output_dir'][last_sep+1:]
        histogram_path = os.path.join(overall_output_dir, f"{exp_name}_loss_histogram.png")
        plt.savefig(histogram_path, dpi=300)
        plt.close()

        # Calculate and print statistics
        print("\nTotal Loss Statistics:")
        print(f"  Mean: {np.mean(total_losses):.4f}")
        print(f"  Median: {np.median(total_losses):.4f}")
        print(f"  Std Dev: {np.std(total_losses):.4f}")
        print(f"  Min: {np.min(total_losses):.4f}")
        print(f"  Max: {np.max(total_losses):.4f}")
        print("\nPercentiles:")
        for percentile, value in zip(percentiles, percentile_values):
            print(f"  {percentile}th: {value:.4f}")

        print(f"Loss histogram saved to: {histogram_path}")
    
    def create_annotated_image(self, comparison_img, loss, norm_loss):
        img_width, img_height = comparison_img.size
        header_height = 28  # Adjusted for one line with larger text

        new_img = Image.new('RGB', (img_width, img_height + header_height), color=(240, 240, 240))
        new_img.paste(comparison_img, (0, header_height))

        draw = ImageDraw.Draw(new_img)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 15)
            except:
                font = ImageFont.load_default()

        color = self.get_color_from_score(norm_loss)

        text = f"Loss: {loss:.3f} ({int(norm_loss*100)}%)"

        left_margin = img_width // 6
        draw.text((left_margin, (header_height - 18) // 2), text, fill=color, font=font)

        # Meter bar on the right
        meter_start_x = img_width // 2 - 15
        meter_width = img_width // 3 + 15
        meter_height = 13
        meter_y = (header_height - meter_height) // 2

        # Background of meter
        draw.rectangle(
            [(meter_start_x, meter_y), (meter_start_x + meter_width, meter_y + meter_height)],
            fill=(220, 220, 220), outline=(180, 180, 180)
        )

        # Filled part of meter
        filled_width = int(meter_width * norm_loss)
        if filled_width > 0:
            draw.rectangle(
                [(meter_start_x, meter_y), (meter_start_x + filled_width, meter_y + meter_height)],
                fill=color
            )

        return new_img

    def normalize_loss(self, loss_value, loss_type):
        """
        Calculate the percentile of a loss value within the distribution
        """
        # Extract all values of this loss type from the test data
        if loss_type == 'total_loss':
            all_values = [data['total_loss'] for data in self.test_data]
        elif loss_type == 'recon_loss':
            all_values = [data['recon_loss'] for data in self.test_data]
        elif loss_type == 'feature_loss':
            all_values = [data['feature_loss'] for data in self.test_data if data['feature_loss'] is not None]

        # Calculate the percentile (0 to 1) of this value within the distribution
        percentile = sum(1 for x in all_values if x <= loss_value) / len(all_values)

        return percentile

    def get_color_from_score(self, percentile):
        """
        Map a percentile (0-1) to a color: blue (0) -> purple (0.5) -> red (1)
        """
        if percentile < 0.5:
            # Blue to Purple (0 to 0.5)
            normalized = percentile * 2  # Scale 0-0.5 to 0-1
            r = int(128 * normalized)
            g = 0
            b = int(255 - 127 * normalized)
        else:
            # Purple to Red (0.5 to 1)
            normalized = (percentile - 0.5) * 2  # Scale 0.5-1 to 0-1
            r = int(128 + 127 * normalized)
            g = 0
            b = int(128 - 128 * normalized)

        return (r, g, b)

    def generate_random_samples(self):
        """
        Generate random samples from the latent space
        """
        try:
            test_input, test_label, _ = next(iter(self.trainer.datamodule.test_dataloader()))
            test_label = test_label.to(self.curr_device)

            samples_dir = os.path.join(self.params['test_output_dir'], "samples")
            os.makedirs(samples_dir, exist_ok=True)

            with torch.no_grad():
                samples = self.model.sample(64, self.curr_device, labels=test_label)

            samples = self.ensure_4_dims(samples)

            for i in range(samples.size(0)):
                sample = samples[i:i+1]
                sample_resized = F.interpolate(
                    sample, size=self.test_output_size, mode='bilinear', align_corners=False
                )
                vutils.save_image(sample_resized.cpu().data,
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
