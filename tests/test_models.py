import torch
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder, PositionalEncoding
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder

def test_vanilla_nerf():
    feature_mlp = VanillaFeatureMLP(10, [256 for k in range(8)])
    opacity_decoder = VanillaOpacityDecoder(256)
    color_decoder = VanillaColorDecoder(10, 256, [128])
    
    rays_o = torch.rand(100, 3)
    rays_d = torch.rand(100, 3)
    features = feature_mlp(rays_o)
    opacity = opacity_decoder(features)
    color = color_decoder(features, rays_d)

    assert features.size() == (100, feature_mlp.feature_dim)
    assert opacity.size() == (100, 1)
    assert color.size() == (100, 3)
    
def test_positional_encoding():
    n_freqs = 10
    pos_enc = PositionalEncoding(n_freqs)

    t = torch.rand(100, 3)
    enc = pos_enc(t)
    assert enc.size() == (100, 3 * n_freqs * 2)
    assert enc.dtype == torch.float

    t = torch.rand(67, 32, 88, 3)
    enc = pos_enc(t)
    assert enc.size() == (67, 32, 88, 3 * n_freqs * 2)
    assert enc.dtype == torch.float
    
def test_kplanes():
    feature_field = KPlanesFeatureField(32)
    opacity_decoder = KPlanesExplicitOpacityDecoder(feature_dim=feature_field.feature_dim)
    color_decoder = KPlanesExplicitColorDecoder(feature_field.feature_dim, 4, 128)
    
    n_rays = 100
    rays_o = torch.rand(n_rays, 3)
    rays_d = torch.rand(n_rays, 3)
    features = feature_field(rays_o)
    opacity = opacity_decoder(features)
    color = color_decoder(features, rays_d)

    assert features.size() == (n_rays, feature_field.feature_dim)
    assert opacity.size() == (n_rays, 1)
    assert color.size() == (n_rays, 3)
    