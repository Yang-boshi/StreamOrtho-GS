# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Feed-forward Gaussian head (Splatt3R), vendored for ZSO week 2 (S3).
#
# Source: `code/SLAM+GS/splatt3r/src/mast3r_src/mast3r/catmlp_dpt_head.py`
#         commit bda1dd07cf84b9189baabb1b8beac243959cfe5d, copied **whole**
#         (design doc S3: do not copy only `GaussianHead`, it depends on
#          `gaussian_postprocess`, the `reg_dense_*` activations and `Mlp`).
#
# Only the import block is adapted: the dust3r/croco paths are resolved from
# the local MASt3R-SLAM checkout instead of Splatt3R's `src/` layout.
# --------------------------------------------------------
import os
import sys

from stream_ortho import config as _zcfg

_REPO = os.path.abspath(str(_zcfg.MAST3R_SLAM_DIR))
for _p in (
    _REPO,
    os.path.join(_REPO, "thirdparty", "mast3r"),
    os.path.join(_REPO, "thirdparty", "mast3r", "dust3r"),
    os.path.join(_REPO, "thirdparty", "mast3r", "dust3r", "croco"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from einops import rearrange

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        else:
            res['desc_conf'] = res['conf'].clone()
    return res


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        return out
    
# @MODIFIED
def reg_dense_offsets(xyz, shift=6.0):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    offsets = xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))
    return offsets

# @MODIFIED
def reg_dense_scales(scales):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    scales = scales.exp()
    return scales

# @MODIFIED
def reg_dense_rotation(rotations, eps=1e-8):
    """
    Apply PixelSplat's rotation normalization
    """
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

# @MODIFIED
def reg_dense_sh(sh):
    """
    Apply PixelSplat's spherical harmonic postprocessing
    """
    sh = rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)
    return sh

# @MODIFIED
def reg_dense_opacities(opacities):
    """
    Apply PixelSplat's opacity postprocessing
    """
    return opacities.sigmoid()

# @MODIFIED
def gaussian_postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None, use_offsets=False, sh_degree=1):
    
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    assert conf_mode is not None, "Confidence mode must be provided for Gaussian head"
    assert desc_dim is not None, "Descriptor dimension must be provided for Gaussian head"
    assert two_confs, "Two confidences must be provided for Gaussian head"
    
    pts3d, conf, desc, desc_conf, offset, scales, rotations, sh, opacities = torch.split(fmap, [3, 1, desc_dim, 1, 3, 3, 4, 3 * sh_degree, 1], dim=-1)
    
    pts3d = reg_dense_depth(pts3d, mode=depth_mode)
    conf = reg_dense_conf(conf.squeeze(-1), mode=conf_mode)
    desc = reg_desc(desc, mode=desc_mode)
    desc_conf = reg_dense_conf(desc_conf.squeeze(-1), mode=desc_conf_mode)
    offset = reg_dense_offsets(offset)
    scales = reg_dense_scales(scales)
    rotations = reg_dense_rotation(rotations)
    sh = reg_dense_sh(sh)
    opacities = reg_dense_opacities(opacities)

    res = {
        'pts3d': pts3d,
        'conf': conf,
        'desc': desc,
        'desc_conf': desc_conf,
        'scales': scales,
        'rotations': rotations,
        'sh': sh,
        'opacities': opacities
    }

    if use_offsets:
        res['means'] = pts3d.detach() + offset
    else:
        res['means'] = pts3d.detach()

    return res


# @MODIFIED
class GaussianHead(PixelwiseTaskWithDPT):
    """Version of the above, modified to also output Gaussian parameters"""

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", use_offsets=False, sh_degree=1, **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

        # Gaussian Num Channels =
            # 3D mean offsets (3) +
            # Scales (3) +
            # Rotations (4) +
            # Spherical Harmonics (3 * sh_degree) +
            # Opacity (1)
        gaussian_num_channels = 3 + 3 + 4 + 3 * sh_degree + 1
        self.gaussian_dpt = PixelwiseTaskWithDPT(
            num_channels=gaussian_num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
            dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type
        )
        
        final_conv_layer = self.gaussian_dpt.dpt.head[-1]
        splits_and_inits = [
            (3, 0.001, 0.001),  # 3D mean offsets
            (3, 0.00003, -7.0),  # Scales
            (4, 1.0, 0.0),  # Rotations
            (3 * sh_degree, 1.0, 0.0),  # Spherical Harmonics
            (1, 1.0, -2.0)  # Opacity
        ]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            torch.nn.init.xavier_uniform_(
                final_conv_layer.weight[start_channels:start_channels+out_channel, :, :, :],
                s
            )
            torch.nn.init.constant_(
                final_conv_layer.bias[start_channels:start_channels+out_channel],
                b
            )
            start_channels += out_channel

        self.use_offsets = use_offsets
        self.sh_degree = sh_degree


    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # extract gaussian_features
        gaussian_features = self.gaussian_dpt.dpt(decout, image_size=(img_shape[0], img_shape[1]))
        # gaussian_features = self.gaussian_local_features(cat_output)  # B,S,D
        # gaussian_features = gaussian_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        # gaussian_features = F.pixel_shuffle(gaussian_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features, gaussian_features], dim=1)
        if self.postprocess:
            out = gaussian_postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode,
                                   use_offsets=self.use_offsets,
                                   sh_degree=self.sh_degree)
        return out


def mast3r_head_factory(head_type, output_mode, net, has_conf=False, use_offsets=False, sh_degree=1):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')
    # @MODIFIED
    elif head_type == 'gaussian_head' and output_mode.startswith('pts3d+gaussian+desc'):
        local_feat_dim = int(output_mode[19:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return GaussianHead(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression',
                                               use_offsets=use_offsets,
                                               sh_degree=sh_degree)
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")


# ---------------------------------------------------------------------------
# ZSO addition: loader (the class itself is untouched)
# ---------------------------------------------------------------------------

def build_gaussian_head(net, device="cuda", use_offsets=False, sh_degree=1,
                        local_feat_dim=24, has_conf=True):
    """Build a standalone ``GaussianHead`` matching the released Splatt3R head.

    The numbers below are **not** guesses; they mirror Splatt3R's
    ``mast3r_head_factory(head_type='gaussian_head', output_mode='pts3d+gaussian+desc24')``:

    * ``local_feat_dim = 24`` (from ``desc24``); ``two_confs=True`` adds the
      second confidence -> 25 channels -> ``head_local_features.fc2`` has
      25*16**2 = 6400 outputs (verified against the checkpoint).
    * ``feature_dim = 256``, ``last_dim = 128`` (= feature_dim//2).
    * ``num_channels = 4`` (3 xyz + 1 conf) -> ``dpt.head.4`` is (4,128,1,1).
    * the gaussian DPT outputs 14 channels = offset(3)+scale(3)+quat(4)+sh(3)+opacity(1).
    * ``use_offsets=False`` because the released checkpoint was trained with
      ``use_offsets: False`` (see its ``hyper_parameters.config``): the trained
      ``means`` are exactly the head's own ``pts3d``.

    The design doc's snippet (``local_feat_dim=16, last_dim=32``) does **not**
    match the released weights and would fail to load.
    """
    l2 = net.dec_depth
    head = GaussianHead(
        net=net,
        has_conf=has_conf,
        local_feat_dim=local_feat_dim,
        num_channels=3 + int(has_conf),
        feature_dim=256,
        last_dim=128,
        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        dim_tokens=[net.enc_embed_dim, net.dec_embed_dim, net.dec_embed_dim, net.dec_embed_dim],
        postprocess=postprocess,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type="regression",
        use_offsets=use_offsets,
        sh_degree=sh_degree,
    )
    return head.to(device)


def load_gaussian_head(net, ckpt_path, device="cuda", head_index=1,
                       use_offsets=False, sh_degree=1, verbose=True):
    """Load Splatt3R's ``encoder.downstream_head{1,2}.*`` weights into a head."""
    head = build_gaussian_head(net, device=device, use_offsets=use_offsets,
                               sh_degree=sh_degree)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    prefix = f"encoder.downstream_head{head_index}."
    head_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not head_sd:
        raise RuntimeError(f"checkpoint 中找不到 {prefix}* 的权重: {ckpt_path}")
    missing, unexpected = head.load_state_dict(head_sd, strict=False)
    if verbose:
        print(f"[gaussian_head] loaded {len(head_sd)} tensors from {prefix}*")
        print(f"[gaussian_head] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("   missing:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("   unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    if missing or unexpected:
        raise RuntimeError(
            f"GaussianHead 权重与结构不匹配 (missing={len(missing)}, unexpected={len(unexpected)})"
        )
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head
