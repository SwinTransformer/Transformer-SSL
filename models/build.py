# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from functools import partial

from .swin_transformer import SwinTransformer
from .moby import MoBY


def build_model(config):
    # ToDo: use a more logical way to build models
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_before_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
        )

    elif model_type == 'moby':
        swin = partial(
            SwinTransformer,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
        )
        encoder = swin(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.ONLINE_DROP_PATH_RATE,
        )
        encoder_k = SwinTransformer(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.TARGET_DROP_PATH_RATE,
        )
        model = MoBY(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.MOBY.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.MOBY.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.MOBY.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.MOBY.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.MOBY.PRED_NUM_LAYERS,
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
