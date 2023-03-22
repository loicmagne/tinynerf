import torch
from src.models import VanillaFeatureMLP, VanillaOpacityDecoder, VanillaColorDecoder, PositionalEncoding
from src.models import KPlanesFeatureField, KPlanesExplicitOpacityDecoder, KPlanesExplicitColorDecoder
from src.models import CobafaFeatureField

def test_vanilla_nerf():
    feature_mlp = VanillaFeatureMLP(10, 256, 8)
    opacity_decoder = VanillaOpacityDecoder(256)
    color_decoder = VanillaColorDecoder(10, 256, 128, 1)
    
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
    assert feature_field.loss_l1().item() >= 0.
    assert feature_field.loss_tv().item() >= 0.

def test_kplanes_hybrid():
    feature_field = KPlanesFeatureField(32)
    opacity_decoder = KPlanesExplicitOpacityDecoder(feature_dim=feature_field.feature_dim)
    color_decoder = VanillaColorDecoder(4, feature_field.feature_dim, 128, 3)
    
    n_rays = 100
    rays_o = torch.rand(n_rays, 3)
    rays_d = torch.rand(n_rays, 3)
    features = feature_field(rays_o)
    opacity = opacity_decoder(features)
    color = color_decoder(features, rays_d)

    assert features.size() == (n_rays, feature_field.feature_dim)
    assert opacity.size() == (n_rays, 1)
    assert color.size() == (n_rays, 3)
    assert feature_field.loss_l1().item() >= 0.
    assert feature_field.loss_tv().item() >= 0.

def test_cobafa():
    feature_field = CobafaFeatureField(
        basis_res=(torch.linspace(32., 128, 6)/1.).int().tolist(),
        coef_res=128,
        freqs=torch.linspace(2., 8., 6).tolist(),
        channels=[4,4,4,2,2,2],
        mlp_hidden_dim=128
    )
    opacity_decoder = VanillaOpacityDecoder(feature_field.feature_dim)
    color_decoder = VanillaColorDecoder(6, feature_field.feature_dim, 128, 2)

    n_rays = 100
    rays_o = torch.rand(n_rays, 3)
    rays_d = torch.rand(n_rays, 3)
    features = feature_field(rays_o)
    opacity = opacity_decoder(features)
    color = color_decoder(features, rays_d)

    assert features.size() == (n_rays, feature_field.feature_dim)
    assert opacity.size() == (n_rays, 1)
    assert color.size() == (n_rays, 3)