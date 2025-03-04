import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class PureAE(BaseVAE):
    """
    A basic Autoencoder model that inherits from BaseVAE to maintain compatibility
    with the existing framework. Unlike VAEs, there's no KL divergence or 
    reparameterization trick - it's a deterministic mapping.
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(PureAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024, 2048]
        self.last_dim = hidden_dims[-1]
        strides = [2, 2, 2, 2, 2] + [1] * (len(hidden_dims) - 5) # Five layers with stride 2, rest with stride 1
        output_padding = lambda stride: 1 if stride > 1 else 0
        # Build Encoder
        for i, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=strides[i], padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*4, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        strides.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=strides[i],
                                       padding=1,
                                       output_padding=output_padding(strides[i])),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent code.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) Latent code
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        # Direct mapping to latent space
        z = self.fc(result)
        
        # Return as list for compatibility with VAE interface
        # But there's no mu/logvar distinction
        return [z]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent code onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.last_dim, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)[0]  # Get first (only) element from encode output
        recons = self.decode(z)
        
        # Return list in format compatible with existing framework
        # [reconstructed_image, original_image, empty_tensor, empty_tensor]
        # to match VAE's [recons, input, mu, log_var]
        dummy_tensor = torch.zeros_like(z)  # Just a placeholder
        return [recons, input, dummy_tensor, dummy_tensor]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the AE loss function.
        :param args: [reconstructed_image, original_image, dummy1, dummy2]
        :param kwargs: Additional arguments
        :return: Dictionary containing loss values
        """
        recons = args[0]
        input = args[1]
        
        # Simple MSE reconstruction loss
        recons_loss = F.mse_loss(recons, input)
        
        # Return in format compatible with existing framework
        return {'loss': recons_loss, 'Reconstruction_Loss': recons_loss, 'KLD': torch.tensor(0.0).to(recons.device)}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from a uniform distribution in the latent space
        and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # Sample from uniform distribution in latent space (unlike VAE's normal distribution)
        z = torch.rand(num_samples, self.latent_dim, device=current_device) * 2 - 1
        
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
