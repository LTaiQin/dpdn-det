import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
import clip
from .dpdn_prompt_bank import DPDNPromptBank


# from .DynamicRefinement import DynamicRefinementModule


class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            num_classes: int,
            zs_weight_path: str,
            zs_weight_dim: int = 512,
            use_bias: float = 0.0,
            norm_weight: bool = True,
            norm_temperature: float = 50.0,
            cfg,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)

        raw_class_names = getattr(cfg.MODEL, "OPEN_VOCAB_CLASS_NAMES", [])
        if len(raw_class_names) == 0:
            raise ValueError(
                "cfg.MODEL.OPEN_VOCAB_CLASS_NAMES is empty. "
                "Please ensure dataset registration populates open-vocabulary classes.")

        foreground_class_names = []
        seen = set()
        for name in raw_class_names:
            normalized = name.strip()
            key = normalized.lower()
            if normalized == '' or key in seen:
                continue
            seen.add(key)
            foreground_class_names.append(normalized)

        if len(foreground_class_names) != cfg.MODEL.ROI_HEADS.NUM_CLASSES:
            raise ValueError(
                f"ROI_HEADS.NUM_CLASSES ({cfg.MODEL.ROI_HEADS.NUM_CLASSES}) must "
                f"match number of open-vocabulary classes ({len(foreground_class_names)}).")

        dpdn_cfg = getattr(cfg.MODEL.ROI_BOX_HEAD, "DPDN", None)
        use_dpdn = bool(getattr(dpdn_cfg, "ENABLED", True)) if dpdn_cfg is not None else True
        self.use_dpdn = use_dpdn
        self.dpdn_kl_weight = float(getattr(dpdn_cfg, "KL_WEIGHT", 1.0)) if dpdn_cfg is not None else 1.0
        if use_dpdn:
            num_templates = int(getattr(dpdn_cfg, "NUM_TEMPLATES", 5)) if dpdn_cfg is not None else 5
            gating_hidden = int(getattr(dpdn_cfg, "GATING_HIDDEN", 128)) if dpdn_cfg is not None else 128
            self.dpdn = DPDNPromptBank(
                clip_model=clip_model,
                class_names=foreground_class_names,
                device=device,
                include_background=True,
                num_templates=num_templates,
                gating_hidden=gating_hidden,
            ).to(device)
        else:
            self.register_buffer("zs_weight", torch.tensor(
                np.load(zs_weight_path), dtype=torch.float32
            ).permute(1, 0).contiguous(), persistent=True)  # [D, C]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            'cfg': cfg,
        }

    def forward(self, x, classifier=None):
        """
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        """

        x = self.linear(x)
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)

        if self.use_dpdn:
            score = self.dpdn(x, return_kl=self.training)
        else:
            zs_weight = self.zs_weight
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))],
                dim=1
            )
            if self.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=0)
            score = torch.mm(x, zs_weight)
        if self.use_bias:
            score = score + self.cls_bias

        return score

    def pop_dpdn_kl_loss(self):
        if not self.use_dpdn:
            return None
        kl = self.dpdn.pop_kl_loss()
        if kl is None:
            return None
        return kl * self.dpdn_kl_weight


class WeightTransferZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            num_classes: int,
            zs_weight_path: str,
            zs_weight_dim: int = 512,
            use_bias: float = 0.0,
            norm_weight: bool = True,
            norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        # this layer now acts as frozen distilled linear layer
        self.linear = nn.Linear(input_size, zs_weight_dim)
        for param in self.linear.parameters():
            param.requires_grad = False

        # FC weight transfer layers
        self.fc1 = nn.Linear(input_size, zs_weight_dim)
        self.fc2 = nn.Linear(zs_weight_dim, input_size)
        self.relu = nn.LeakyReLU(0.1)
        # FC residual layers
        self.fc3 = nn.Linear(input_size, 1024)
        self.fc4 = nn.Linear(1024, zs_weight_dim)
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None, des_feats=None):
        """
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        """
        # Compute the weights through transfer function
        t = self.fc1(self.linear.weight)
        t_act = self.relu(t)
        transfer_weights = self.fc2(t_act)
        # Pass though linear layer after weight transfer
        res_x = self.fc3(x)
        res_x = self.relu(res_x)
        res_x = self.fc4(res_x)
        x = res_x + F.linear(x, weight=transfer_weights)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
