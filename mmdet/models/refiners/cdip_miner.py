import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import bbox2roi

try:
    from ssdcv.core import get_valid_proposals
except ImportError:
    from mmdet.core import get_valid_proposals


@HEADS.register_module()
class CDIPMiner(nn.Module):
    """Category-Detail Information Joint Proposal Miner.

    This module integrates category and detail information for accurate
    proposal mining under noisy label conditions, corresponding to Eq. (1)-(8)
    in the paper.
    """

    def __init__(self,
                 in_channels,
                 roi_extractor,
                 detail_branch_channels=256,
                 gabor_orientations=8,
                 num_classes=80,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CDIPMiner, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.detail_branch_channels = detail_branch_channels
        self.gabor_orientations = gabor_orientations
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ===== Detail information perception branch =====
        self.detail_roi_extractor = build_roi_extractor(roi_extractor)

        # Multi-scale convolution branches
        self.conv_branch1 = ConvModule(
            in_channels, detail_branch_channels, kernel_size=(7, 1), padding=(3, 0))
        self.conv_branch2 = ConvModule(
            in_channels, detail_branch_channels, kernel_size=(5, 1), padding=(2, 0))
        self.conv_branch3 = ConvModule(
            in_channels, detail_branch_channels, kernel_size=(1, 1), padding=0)

        # Fusion layer
        self.fusion_conv = ConvModule(
            detail_branch_channels * 3, detail_branch_channels, kernel_size=1)

        # Detail feature FC layer
        self.detail_fc = nn.Linear(detail_branch_channels * 7 * 7, 1024)
        self.detail_cls = nn.Linear(1024, 1)

        # ===== Category information perception branch =====
        self.category_fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        self.cls_branch = nn.Linear(1024, num_classes)
        self.ins_branch = nn.Linear(1024, 1)

        # ===== Activation functions =====
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def gabor_filter(self, x):
        """Apply Gabor filter to extract edge and texture features."""
        import cv2
        import numpy as np

        batch_size = x.size(0)
        filtered = []

        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            kernels = []
            for theta_idx in range(self.gabor_orientations):
                theta_val = theta_idx * np.pi / self.gabor_orientations
                kernel = cv2.getGaborKernel(
                    (21, 21), 4.0, theta_val, 10.0, 0.5, 0,
                    ktype=cv2.CV_32F
                )
                kernels.append(kernel)

            filtered_img = np.zeros_like(img)
            for kernel in kernels:
                filtered_i = cv2.filter2D(img, cv2.CV_32F, kernel)
                filtered_img = np.maximum(filtered_img, np.abs(filtered_i))

            filtered.append(torch.from_numpy(filtered_img.transpose(2, 0, 1)).float().to(x.device))

        return torch.stack(filtered)

    def forward_detail_branch(self, x, proposals, img_metas):
        """Forward detail information perception branch."""
        if self.training:
            x_gabor = [self.gabor_filter(x_i) for x_i in x]
        else:
            x_gabor = x

        roi_feats = self.detail_roi_extractor(
            x_gabor[:self.detail_roi_extractor.num_inputs],
            proposals
        )

        feat1 = self.conv_branch1(roi_feats)
        feat2 = self.conv_branch2(roi_feats)
        feat3 = self.conv_branch3(roi_feats)

        fused = torch.cat([feat1, feat2, feat3], dim=1)
        fused = self.fusion_conv(fused)

        fused = fused.flatten(1)
        detail_feat = self.relu(self.detail_fc(fused))
        detail_score = self.sigmoid(self.detail_cls(detail_feat))

        return detail_score, detail_feat

    def forward_category_branch(self, roi_feats):
        """Forward category information perception branch."""
        x = self.category_fc(roi_feats)
        cls_score = self.cls_branch(x)
        ins_score = self.sigmoid(self.ins_branch(x))

        cls_score = self.softmax(cls_score)
        ins_score = self.sigmoid(ins_score)

        return cls_score, ins_score, x

    def compute_mil_loss(self, cls_scores, ins_scores, detail_scores, gt_labels):
        """Compute MIL loss with Hadamard product (Eq. 7-8)."""
        losses = []

        for i in range(len(cls_scores)):
            if len(cls_scores[i]) == 0:
                continue

            # S = S_c ⊙ S_n ⊙ S_d
            S = cls_scores[i] * ins_scores[i] * detail_scores[i]
            S_bag = S.sum(dim=0)
            S_bag = S_bag / (S_bag.sum() + 1e-8)

            if len(gt_labels[i]) > 0:
                target = gt_labels[i][0]
                loss = F.cross_entropy(S_bag.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=cls_scores[0].device)
        return sum(losses) / len(losses)

    def forward_train(self, x, proposal_list, gt_bboxes, gt_labels, ann_weight, img_metas):
        """Forward pass for training."""
        valid_proposals = get_valid_proposals(proposal_list, gt_bboxes, gt_labels, ann_weight)

        # Detail branch
        detail_scores = []
        detail_feats = []
        for proposals in valid_proposals:
            detail_score, detail_feat = self.forward_detail_branch(x, proposals, img_metas)
            detail_scores.append(detail_score)
            detail_feats.append(detail_feat)

        # Category branch
        cls_scores = []
        ins_scores = []
        for proposals in valid_proposals:
            rois = bbox2roi(proposals)
            roi_feats = self.detail_roi_extractor(
                x[:self.detail_roi_extractor.num_inputs],
                rois
            )
            roi_feats = roi_feats.flatten(1)
            cls_score, ins_score, _ = self.forward_category_branch(roi_feats)
            cls_scores.append(cls_score)
            ins_scores.append(ins_score)

        # Compute losses
        losses = {}

        # MIL loss
        losses['loss_cdip_mil'] = self.compute_mil_loss(
            cls_scores, ins_scores, detail_scores, gt_labels
        )

        # Classification loss
        cls_loss = 0
        for i, score in enumerate(cls_scores):
            if len(score) > 0 and len(gt_labels[i]) > 0:
                cls_loss += F.cross_entropy(score, gt_labels[i][0].unsqueeze(0))
        losses['loss_cdip_cls'] = cls_loss / len(cls_scores) if len(cls_scores) > 0 else 0

        # Instance loss
        ins_loss = 0
        for i, score in enumerate(ins_scores):
            if len(score) > 0 and len(gt_labels[i]) > 0:
                target = torch.ones_like(score) * gt_labels[i][0]
                ins_loss += F.binary_cross_entropy(score.squeeze(), target.float())
        losses['loss_cdip_ins'] = ins_loss / len(ins_scores) if len(ins_scores) > 0 else 0

        # Detail loss
        detail_loss = 0
        for i, score in enumerate(detail_scores):
            if len(score) > 0:
                detail_loss += -torch.log(score.mean() + 1e-8)
        losses['loss_cdip_detail'] = detail_loss / len(detail_scores) if len(detail_scores) > 0 else 0

        return losses

    def forward_test(self, x, proposal_list, img_metas):
        """Forward pass for testing."""
        return proposal_list