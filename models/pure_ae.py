import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn
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
                 use_vgg: bool = False,
                 **kwargs) -> None:
        super(PureAE, self).__init__()

        self.latent_dim = latent_dim
        self.use_vgg = use_vgg

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        print("Hidden dims: ", hidden_dims)
        self.last_dim = hidden_dims[-1]
        strides = [2, 2, 2, 2, 2] + [1] * (len(hidden_dims) - 5) # Five layers with stride 2, rest with stride 1. Note: performance was poor when I tried >5 layers, not sure why. Could be something to experiment with.
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

        if self.use_vgg:
            self.feature_network = vgg19_bn(pretrained=True)
            # Freeze the pretrained feature network
            for param in self.feature_network.parameters():
                param.requires_grad = False
            self.feature_network.eval()
            
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

        if self.use_vgg:
            input_features = self.extract_features(input)
            recons_features = self.extract_features(recons)
        else: # Just use placeholders, we'll ignore them if we're not using VGG
            input_features = torch.zeros_like(z) 
            recons_features = torch.zeros_like(z)
        return [recons, input, recons_features, input_features]

    def extract_features(self,
                         input: Tensor,
                         feature_layers: List = None) -> List[Tensor]:
        """                                                                                                
        Extracts the features from the pretrained model    
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if not self.use_vgg:
            raise ValueError("Should not get here because we only call this when we're using VGG")
                
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features    

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
        recons_features = args[2]
        input_features = args[3]
        
        # Simple MSE pixel loss
        recons_loss = F.mse_loss(recons, input)

        # VGG loss
        feature_loss = torch.tensor(0.0).to(recons.device)
        if self.use_vgg:
            for (r, i) in zip(recons_features, input_features):
                feature_loss += F.mse_loss(r, i)
            loss = recons_loss + feature_loss
        else:
            loss = recons_loss
        
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': torch.tensor(0.0).to(recons.device), 'feature_loss': feature_loss}

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
        # We'll probably never use this, but it's here for compatibility
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
