import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Optional, Tuple

class DPDNPromptBank(nn.Module):
    """
    Dynamic Prompt Distribution Network (DPDN) – Probabilistic / Variational Version.

    Matches the architecture diagram:
    - Density/Scatter Plot: Implemented via Variational Inference (predicting mu, sigma).
    - Gamma/Beta: Implemented as Affine Transformations on the template weights.
    """

    def __init__(
        self,
        *,
        clip_model: nn.Module,
        class_names: List[str],
        device: str = "cuda",
        include_background: bool = True,
        num_templates: int = 5,
        gating_hidden: int = 128,
        latent_dim: int = 64  # 新增：潜变量维度
    ):
        super().__init__()
        self.device = device
        self.clip_model = clip_model
        self.class_names = list(class_names)
        self.include_background = bool(include_background)
        self.num_templates = int(num_templates)

        # --- 1. Template Preparation ---
        template_pool = [
            "a photo of a {}", "a close-up of {}", "a detailed view of {}",
            "a high resolution image of {}", "a professional photograph of {}",
            "an image focusing on the texture of {}", "a visual highlighting the color of {}",
            "a shot emphasizing the shape of {}",
        ]
        self.templates = template_pool[: max(1, self.num_templates)]
        self.feature_dim = clip_model.visual.output_dim
        self.all_class_names = self.class_names + (["background"] if self.include_background else [])

        # --- 2. Probabilistic Gating Network (The "Density Head") ---
        # 对应图中的 Density / Scatter Plot 部分
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, gating_hidden),
            nn.ReLU(inplace=True)
        )
        
        # 预测分布的参数：均值 mu 和 对数方差 log_var
        self.fc_mu = nn.Linear(gating_hidden, latent_dim)
        self.fc_var = nn.Linear(gating_hidden, latent_dim) 

        # --- 3. Dynamic Template Decoder (Gamma/Beta) ---
        # 对应图中的 Dynamic Templates -> Gamma/Beta 部分
        # 从潜变量 z 解码出混合权重
        self.weight_generator = nn.Sequential(
            nn.Linear(latent_dim, gating_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gating_hidden, len(self.templates))
        )
        
        # 可选：如果想显式模拟 Gamma/Beta 仿射变换，可以使用下面的结构
        self.fc_gamma = nn.Linear(latent_dim, len(self.templates))
        self.fc_beta = nn.Linear(latent_dim, len(self.templates))

        # Register buffers
        self.register_buffer("template_weight", self._build_template_weight(), persistent=True)
        self.register_buffer("prototype_matrix", self._build_default_prototypes(), persistent=True)
        self._use_fixed = False
        self._last_kl_loss: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _build_template_weight(self) -> torch.Tensor:
        # Same as before: Build [T, D, C] tensor
        weights = []
        # 注意：如果你决定要在图中去掉 BG Proto 的独立框，就用这种方式混合
        # 如果你想保留独立框，这里就不要 append "background"
        process_classes = self.all_class_names 
        
        for template in self.templates:
            prompts = [template.format(name) for name in process_classes]
            tokens = clip.tokenize(prompts).to(self.device)
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            weights.append(emb.t().contiguous()) 
        return torch.stack(weights, dim=0) # [T, D, C]

    @torch.no_grad()
    def _build_default_prototypes(self) -> torch.Tensor:
        proto = self.template_weight.mean(dim=0).t().contiguous()
        return F.normalize(proto, p=2, dim=1)

    def set_dynamic(self, enabled: bool = True):
        self._use_fixed = not bool(enabled)

    def set_fixed_prototypes(self, prototypes: torch.Tensor):
        """
        Set fixed prototypes for inference-time classifier reset.

        Expected shape: [num_classes, feature_dim].
        """
        with torch.no_grad():
            proto = prototypes.to(self.prototype_matrix.device, dtype=self.prototype_matrix.dtype)
            proto = F.normalize(proto, p=2, dim=1)
            self.prototype_matrix.data.copy_(proto)
        self._use_fixed = True

    def pop_kl_loss(self) -> Optional[torch.Tensor]:
        kl = self._last_kl_loss
        self._last_kl_loss = None
        return kl

    def forward(self, visual_features: torch.Tensor, return_kl: bool = False) -> torch.Tensor:
        """
        Forward logic matching the diagram structure:
        RoI -> Density (mu, sigma) -> Sample z -> Gamma/Beta -> Aggregation
        """
        x = visual_features
        x_norm = F.normalize(visual_features, dim=-1)

        if self._use_fixed:
            proto = F.normalize(self.prototype_matrix, dim=-1)
            self._last_kl_loss = None
            return x @ proto.t()

        # --- Step 1: Density Modeling (对应图中的坐标轴和 rho, sigma) ---
        h = self.shared_encoder(x_norm)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        # 计算标准差 sigma (用于图中的 sigma 符号)
        std = torch.exp(0.5 * log_var) 

        # --- Step 2: Sampling (对应图中的散点图/随机性) ---
        # Reparameterization trick: z = mu + sigma * epsilon
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # 推理时通常直接用均值，或者也采样（看你需求）

        # --- Step 3: Gamma / Beta Generation (对应图中的 gamma, beta) ---
        # 我们假设 Gamma 控制权重的锐度(Scale)，Beta 控制权重的偏移(Shift)
        gamma = torch.sigmoid(self.fc_gamma(z)) # range (0, 1)
        beta = self.fc_beta(z)                  # range (-inf, inf)

        # --- Step 4: Aggregation (对应图中的 Sum 符号) ---
        # 原始权重 logits
        base_logits = self.weight_generator(z) # [B, T]
        
        # 应用 Gamma 和 Beta 进行仿射变换 (Affine Transformation)
        # 这完美解释了图中的箭头经过 gamma/beta 指向 Aggregation
        modulated_logits = base_logits * gamma + beta
        
        gate_weights = torch.softmax(modulated_logits, dim=-1) # [B, T]

        # 最终聚合: Sum(Weight_t * Template_t)
        # logits_per_t: [B, T, C] (Precomputed via einsum logic)
        logits_per_t = torch.einsum("bd,tdc->btc", x, self.template_weight) 
        final_logits = torch.einsum("bt,btc->bc", gate_weights, logits_per_t)

        if return_kl:
            # KL divergence between N(mu, var) and N(0, 1)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
            self._last_kl_loss = kl_loss
        else:
            self._last_kl_loss = None

        return final_logits
