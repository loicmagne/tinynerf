import torch
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder

def test_vanilla_nerf():
    freqs_o = 2**torch.arange(0, 10) * torch.pi
    freqs_d = 2**torch.arange(0, 4) * torch.pi
    feature_mlp = VanillaFeatureMLP(freqs_o, [256 for k in range(8)])
    opacity_decoder = VanillaOpacityDecoder(256)
    color_decoder = VanillaColorDecoder(freqs_d, 256, [128])
    
    rays_o = torch.rand(100, 3)
    rays_d = torch.rand(100, 3)
    features = feature_mlp(rays_o)
    opacity = opacity_decoder(features)
    color = color_decoder(rays_d, features)

    assert opacity.size() == (100, 1)
    assert color.size() == (100, 3)