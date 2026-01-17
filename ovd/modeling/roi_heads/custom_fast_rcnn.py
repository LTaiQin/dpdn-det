import math
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from ..utils import load_class_freq, get_fed_loss_inds
from .zero_shot_classifier import ZeroShotClassifier, WeightTransferZeroShotClassifier
from torch.nn.functional import normalize

__all__ = ["CustomFastRCNNOutputLayers"]


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            cls_score=None,
            use_sigmoid_ce=False,
            use_fed_loss=False,
            ignore_zero_cats=False,
            fed_loss_num_cat=50,
            image_label_loss='',
            use_zeroshot_cls=False,
            pms_loss_weight=0.1,
            prior_prob=0.01,
            cat_freq_path='',
            fed_loss_freq_weight=0.5,
            distil_l1_loss_weight=0.5,
            irm_loss_weight=0.0,
            rkd_temperature=100,
            num_distil_prop=5,
            weight_transfer=False,
            mil_topk=5,
            mil_tau=1.0,
            mil_label_smooth=0.0,
            mil_use_prop_score=True,
            **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )
        self.use_zeroshot_cls = use_zeroshot_cls
        self.use_sigmoid_ce = use_sigmoid_ce
        self.image_label_loss = image_label_loss
        self.pms_loss_weight = pms_loss_weight
        self.use_fed_loss = use_fed_loss
        self.ignore_zero_cats = ignore_zero_cats
        self.fed_loss_num_cat = fed_loss_num_cat
        self.distil_l1_loss_weight = distil_l1_loss_weight
        self.irm_loss_weight = irm_loss_weight
        self.rkd_temperature = rkd_temperature
        self.num_distil_prop = num_distil_prop
        self.weight_transfer = weight_transfer
        self.mil_topk = int(mil_topk)
        self.mil_tau = float(mil_tau)
        self.mil_label_smooth = float(mil_label_smooth)
        self.mil_use_prop_score = bool(mil_use_prop_score)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

        if self.use_fed_loss or self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        if self.use_zeroshot_cls:
            del self.cls_score
            del self.bbox_pred
            assert cls_score is not None
            self.cls_score = cls_score
            self.bbox_pred = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, 4)
            )
            weight_init.c2_xavier_fill(self.bbox_pred[0])
            nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
            nn.init.constant_(self.bbox_pred[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'use_zeroshot_cls': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS,
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'image_label_loss': cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS,
            'pms_loss_weight': cfg.MODEL.ROI_BOX_HEAD.PMS_LOSS_WEIGHT,
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
            'fed_loss_num_cat': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
            'distil_l1_loss_weight': cfg.MODEL.DISTIL_L1_LOSS_WEIGHT,
            'irm_loss_weight': cfg.MODEL.IRM_LOSS_WEIGHT,
            'num_distil_prop': cfg.MODEL.NUM_DISTIL_PROP,
            'weight_transfer': cfg.MODEL.ROI_BOX_HEAD.WEIGHT_TRANSFER,
            'mil_topk': cfg.MODEL.ROI_BOX_HEAD.MIL_TOPK,
            'mil_tau': cfg.MODEL.ROI_BOX_HEAD.MIL_TAU,
            'mil_label_smooth': cfg.MODEL.ROI_BOX_HEAD.MIL_LABEL_SMOOTH,
            'mil_use_prop_score': cfg.MODEL.ROI_BOX_HEAD.MIL_USE_PROP_SCORE,
        })
        use_bias = cfg.MODEL.ROI_BOX_HEAD.USE_BIAS
        if ret['use_zeroshot_cls']:
            if ret['weight_transfer']:
                ret['cls_score'] = WeightTransferZeroShotClassifier(cfg, input_shape, use_bias=use_bias)
            else:
                ret['cls_score'] = ZeroShotClassifier(cfg, input_shape, use_bias=use_bias)
        return ret

    def losses(self, predictions, proposals, distil_features, use_advanced_loss=True):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        num_classes = self.num_classes

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        cls_loss = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        dpdn_kl_loss = None
        if hasattr(self.cls_score, "pop_dpdn_kl_loss"):
            dpdn_kl_loss = self.cls_score.pop_dpdn_kl_loss()
        # print("cls_loss:", cls_loss)
        # region-based knowledge distillation loss
        if distil_features is not None:
            image_features, clip_features = distil_features
            # Point-wise embedding matching loss (L1)
            distil_l1_loss = self.distil_l1_loss(image_features, clip_features)
            if self.irm_loss_weight > 0:
                # Inter-embedding relationship matching loss (IRM)
                irm_loss = self.irm_loss(image_features, clip_features)
                losses = {
                    "cls_loss": cls_loss,
                    "box_reg_loss": self.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                        num_classes=num_classes),
                    "distil_l1_loss": distil_l1_loss,
                    "irm_loss": irm_loss,
                }
            else:
                losses = {
                    "cls_loss": cls_loss,
                    "box_reg_loss": self.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                        num_classes=num_classes),
                    "distil_l1_loss": distil_l1_loss,
                }
        else:
            losses = {
                "cls_loss": cls_loss,
                "box_reg_loss": self.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                    num_classes=num_classes)
            }
        if dpdn_kl_loss is not None:
            losses["dpdn_kl_loss"] = dpdn_kl_loss
        return losses

    # Point-wise embedding matching loss (L1)
    def distil_l1_loss(self, image_features, clip_features):
        weight = self.distil_l1_loss_weight * self.rkd_temperature
        loss = F.l1_loss(image_features, clip_features, reduction='mean')
        loss = loss * weight
        return loss

    # Inter-embedding relationship matching loss (IRM)
    def irm_loss(self, image_features, clip_features):
        weight = self.irm_loss_weight * self.rkd_temperature
        g_img = normalize(torch.matmul(image_features, torch.t(image_features)), 1)
        g_clip = normalize(torch.matmul(clip_features, torch.t(clip_features)), 1)
        irm_loss = torch.norm(g_img - g_clip) ** 2
        return irm_loss * weight

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :C]

        weight = 1

        if self.use_fed_loss and (self.freq_weight is not None):
            appeared = get_fed_loss_inds(gt_classes, num_sample_cats=self.fed_loss_num_cat, C=C,
                                         weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()
        if self.ignore_zero_cats and (self.freq_weight is not None):
            w = (self.freq_weight.view(-1) > 1e-4).float()
            weight = weight * w.view(1, C).expand(B, C)

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def box_reg_loss(
            self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, num_classes=-1):
        """
        Allow custom background index
        """
        num_classes = num_classes if num_classes > 0 else self.num_classes
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < num_classes))[0]
        # class-agnostic regression
        fg_pred_deltas = pred_deltas[fg_inds]
        # smooth_l1 loss
        gt_pred_deltas = self.box2box_transform.get_deltas(proposal_boxes[fg_inds], gt_boxes[fg_inds], )
        box_reg_loss = smooth_l1_loss(fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum")
        return box_reg_loss / max(gt_classes.numel(), 1.0)

    def inference(self, predictions, proposals):
        """
        enable use proposal boxes
        """
        predictions = (predictions[0], predictions[1])
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = scores.sigmoid()
        return probs.split(num_inst_per_image, dim=0)

    def image_label_losses(self, predictions, proposals, distil_features, image_labels):
        """
        Inputs:
            scores: N x (C + 1)
            image_labels B x 1
        """
        num_inst_per_image = [len(p) for p in proposals]
        scores = predictions[0]
        scores = scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)
        prop_scores = [None for _ in num_inst_per_image]
        B = len(scores)
        img_box_count = 0
        select_size_count = 0
        select_x_count = 0
        select_y_count = 0
        max_score_count = 0
        storage = get_event_storage()
        loss = scores[0].new_zeros([1])[0]
        for idx, (score, labels, prop_score, p) in enumerate(zip(
                scores, image_labels, prop_scores, proposals)):
            if score.shape[0] == 0:
                loss += score.new_zeros([1])[0]
                continue
            for i_l, label in enumerate(labels):
                label = int(label)
                if hasattr(p, "target_classes"):
                    candidate_inds = (p.target_classes == label).nonzero().flatten()
                    if candidate_inds.numel() == 0:
                        continue
                else:
                    # Paper-style MIL: treat all RPN proposals as the candidate bag.
                    candidate_inds = torch.arange(score.shape[0], device=score.device)
                if self.image_label_loss == 'pseudo_max_score':
                    loss_i, ind = self._psuedo_maxscore_loss(score, label, candidate_inds, p)
                elif self.image_label_loss == 'mil_topk':
                    loss_i, ind = self._mil_topk_loss(score, label, candidate_inds, p)
                else:
                    assert 0, self.image_label_loss
                loss += loss_i / len(labels)
                if type(ind) == type([]):
                    img_box_count = sum(ind) / len(ind)
                    if self.debug:
                        for ind_i in ind:
                            p.selected[ind_i] = label
                else:
                    img_box_count = ind
                    select_size_count = p[ind].proposal_boxes.area() / (p.image_size[0] * p.image_size[1])
                    max_score_count = score[ind, label].sigmoid()
                    select_x_count = (p.proposal_boxes.tensor[ind, 0] +
                                      p.proposal_boxes.tensor[ind, 2]) / 2 / p.image_size[1]
                    select_y_count = (p.proposal_boxes.tensor[ind, 1] +
                                      p.proposal_boxes.tensor[ind, 3]) / 2 / p.image_size[0]

        loss = loss / B
        storage.put_scalar('stats_l_image', loss.item())
        if comm.is_main_process():
            storage.put_scalar('pool_stats', img_box_count)
            storage.put_scalar('stats_select_size', select_size_count)
            storage.put_scalar('stats_select_x', select_x_count)
            storage.put_scalar('stats_select_y', select_y_count)
            storage.put_scalar('stats_max_label_score', max_score_count)

        # region-based knowledge (RKD) distillation loss
        dpdn_kl_loss = None
        if hasattr(self.cls_score, "pop_dpdn_kl_loss"):
            dpdn_kl_loss = self.cls_score.pop_dpdn_kl_loss()
        if distil_features is not None:
            image_features, clip_features = distil_features
            # Point-wise embedding matching loss (L1)
            distil_l1_loss = self.distil_l1_loss(image_features, clip_features)
            if self.irm_loss_weight > 0:
                # Inter-embedding relationship loss (IRM)
                irm_loss = self.irm_loss(image_features, clip_features)
                losses = {
                    'pms_loss': loss * self.pms_loss_weight,
                    'cls_loss': score.new_zeros([1])[0],
                    'box_reg_loss': score.new_zeros([1])[0],
                    'distil_l1_loss': distil_l1_loss,
                    'irm_loss': irm_loss,
                }
            else:
                losses = {
                    'pms_loss': loss * self.pms_loss_weight,
                    'cls_loss': score.new_zeros([1])[0],
                    'box_reg_loss': score.new_zeros([1])[0],
                    'distil_l1_loss': distil_l1_loss,
                }
        else:
            losses = {
                'pms_loss': loss * self.pms_loss_weight,
                'cls_loss': score.new_zeros([1])[0],
                'box_reg_loss': score.new_zeros([1])[0]
            }
        if dpdn_kl_loss is not None:
            losses["dpdn_kl_loss"] = dpdn_kl_loss
        return losses

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        scores = []
        cls_scores = self.cls_score(x)
        scores.append(cls_scores)
        scores = torch.cat(scores, dim=1)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    # pseudo-max score loss (pms loss)
    def _psuedo_maxscore_loss(self, score, label, candidate_inds, p):
        target = score.new_zeros(score.shape[1])
        target[label] = 1.0
        label_logits = score[candidate_inds, label]
        best_pos = torch.argmax(label_logits)
        ind = int(candidate_inds[best_pos].item())
        loss = F.binary_cross_entropy_with_logits(score[ind], target, reduction='sum')
        return loss, ind

    def _mil_topk_loss(self, score, label, candidate_inds, p):
        target = score.new_zeros(score.shape[1])
        if self.mil_label_smooth > 0:
            eps = self.mil_label_smooth
            target = target + eps / float(score.shape[1])
            target[label] = 1.0 - eps + eps / float(score.shape[1])
        else:
            target[label] = 1.0

        label_logits = score[candidate_inds, label]
        topk = min(max(self.mil_topk, 1), int(label_logits.numel()))
        top_vals, top_local = torch.topk(label_logits, k=topk, dim=0)
        top_inds = candidate_inds[top_local]

        weights = torch.softmax(top_vals / max(self.mil_tau, 1e-6), dim=0)
        if self.mil_use_prop_score and hasattr(p, "objectness_logits") and p.objectness_logits.numel() == len(p):
            prop_w = p.objectness_logits[top_inds].sigmoid().clamp(min=0)
            weights = weights * prop_w
            weights = weights / (weights.sum() + 1e-12)

        loss = score.new_zeros([1])[0]
        for w, ind in zip(weights, top_inds):
            loss = loss + w * F.binary_cross_entropy_with_logits(score[int(ind.item())], target, reduction='sum')
        return loss, top_inds.cpu().tolist()
