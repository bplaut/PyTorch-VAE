import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn
from .types_ import *

class Autoencoder(BaseVAE):
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
                 center_focus_sigma: float = None,  # If provided, will use center-weighted loss
                 use_skip_connections: bool = False,
                 **kwargs) -> None:
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.use_vgg = use_vgg
        self.center_focus_sigma = center_focus_sigma
        self.center_weight_mask = None
        self.use_skip_connections = use_skip_connections

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        print("Hidden dims: ", hidden_dims)
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

        if self.use_vgg:
            self.feature_network = vgg19_bn(pretrained=True)
            # Freeze the pretrained feature network
            for param in self.feature_network.parameters():
                param.requires_grad = False
            self.feature_network.eval()
            
    def create_center_weight_mask(self, height, width, device):
        """
        Creates a Gaussian weighting mask that emphasizes the center of the image.
        
        Args:
            height: Height of the image
            width: Width of the image
            device: Device to place the tensor on
            
        Returns:
            Tensor of shape [1, 1, height, width] with highest weights at center
        """
        # Create coordinate grids
        y_coords = torch.arange(height, device=device).float()
        x_coords = torch.arange(width, device=device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Center coordinates
        y_center = (height - 1) / 2
        x_center = (width - 1) / 2
        
        # Compute Gaussian weights based on distance from center
        squared_dist = (y_grid - y_center)**2 + (x_grid - x_center)**2
        weights = torch.exp(-squared_dist / (2 * self.center_focus_sigma**2))
        
        # Normalize weights so the average weight is 1.0
        # This ensures the overall loss magnitude stays similar
        weights = weights * (height * width / weights.sum())
        
        # Add dimensions for batch and channel broadcasting
        return weights.unsqueeze(0).unsqueeze(0)
    
    def weighted_mse_loss(self, input, target, weight_mask):
        """
        Computes MSE loss with spatial weighting that prioritizes the center.
        
        Args:
            input: Predicted image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            weight_mask: Weight mask tensor [1, 1, H, W]
            
        Returns:
            Weighted MSE loss
        """
        # Element-wise squared error
        squared_diff = (input - target)**2
        
        # Expand mask to match input dimensions
        expanded_mask = weight_mask.expand(input.size(0), input.size(1), -1, -1)
        
        # Apply weights and calculate mean
        return (squared_diff * expanded_mask).mean()
    
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
        if not self.use_skip_connections:
            z = self.encode(input)[0]
            recons = self.decode(z)
        else:
            # Capture encoder outputs
            encoder_outputs = []
            x = input

            # Run through encoder, saving intermediate outputs
            for layer in self.encoder:
                x = layer(x)
                encoder_outputs.append(x)

            # Flatten and project to latent space
            z_flat = torch.flatten(x, start_dim=1)
            z = self.fc(z_flat)

            # Start decoding
            x = self.decoder_input(z)
            x = x.view(-1, self.last_dim, 2, 2)

            # Apply decoder with fixed skip connections
            # Assume encoder and decoder have same number of layers (reversed in decoder)
            for i, layer in enumerate(self.decoder):
                x = layer(x)

                # Connect to corresponding encoder layer (in reverse order)
                # Skip the earliest encoder layers if there are too many
                skip_idx = len(encoder_outputs) - i - 1
                if skip_idx >= 0 and x.shape[2:] == encoder_outputs[skip_idx].shape[2:]:
                    # Simple residual connection - just add the features
                    # (assumes channels already match or don't need to match perfectly)
                    x = x + encoder_outputs[skip_idx]

            # Final layer
            recons = self.final_layer(x)

        if self.use_vgg:
            input_features = self.extract_features(input)
            recons_features = self.extract_features(recons)
        else: # Just use placeholders, we'll ignore them if we're not using VGG
            input_features = torch.zeros_like(z) 
            recons_features = torch.zeros_like(z)
        return [recons, input, recons_features, input_features]

    def extract_features(self, input: Tensor, feature_layers: List = None) -> List[Tensor]:
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

    def loss_function(self, *args, **kwargs) -> dict:
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
        
        # Initialize or update center weight mask if using center-weighted loss
        if self.center_focus_sigma is not None:
            height, width = input.shape[2], input.shape[3]
            
            # Create mask if it doesn't exist, then save it for later so we don't have to recreate it
            if self.center_weight_mask is None:
                self.center_weight_mask = self.create_center_weight_mask(height, width, input.device)
                
            recons_loss = self.weighted_mse_loss(recons, input, self.center_weight_mask)
        else:
            # Standard MSE loss
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
