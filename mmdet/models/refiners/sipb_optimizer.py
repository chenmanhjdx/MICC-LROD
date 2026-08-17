import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.builder import HEADS, build_roi_extractor
from mmdet.core import bbox2roi


@HEADS.register_module()
class SIPBOptimizer(nn.Module):
    """Spatial Information-Assisted Proposal Bag Optimizer.

    This module integrates spatial information from a wide receptive field
    to optimize proposal bags, corresponding to Eq. (9)-(10) in the paper.
    """

    def __init__(self,
                 in_channels,
                 roi_extractor,
                 relaxation_factors=[2.0, 4.0],
                 se_ratio=16,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SIPBOptimizer, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.relaxation_factors = relaxation_factors
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.roi_extractor = build_roi_extractor(roi_extractor)

        # Multi-scale convolution
        self.scale_conv = ConvModule(
            in_channels * 3, in_channels, kernel_size=3, padding=1)

        # SE module
        self.se_ratio = se_ratio
        self.se_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // se_ratio, in_channels),
            nn.Sigmoid()
        )

        # Regression layer
        self.reg_fc = nn.Linear(in_channels * 7 * 7, 4)

        # Shared FC layers
        self.shared_fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.shared_fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU(inplace=True)

    def scale_relaxation(self, proposals, img_shape):
        """Apply scale relaxation to generate multi-scale proposals."""
        relaxed_proposals = []
        for factor in self.relaxation_factors:
            relaxed = proposals.clone()
            widths = proposals[:, 2] - proposals[:, 0]
            heights = proposals[:, 3] - proposals[:, 1]
            cx = (proposals[:, 0] + proposals[:, 2]) / 2
            cy = (proposals[:, 1] + proposals[:, 3]) / 2

            new_w = widths * factor
            new_h = heights * factor

            relaxed[:, 0] = cx - new_w / 2
            relaxed[:, 2] = cx + new_w / 2
            relaxed[:, 1] = cy - new_h / 2
            relaxed[:, 3] = cy + new_h / 2

            relaxed[:, 0] = relaxed[:, 0].clamp(0, img_shape[1])
            relaxed[:, 1] = relaxed[:, 1].clamp(0, img_shape[0])
            relaxed[:, 2] = relaxed[:, 2].clamp(0, img_shape[1])
            relaxed[:, 3] = relaxed[:, 3].clamp(0, img_shape[0])

            relaxed_proposals.append(relaxed)

        return relaxed_proposals

    def forward(self, x, proposal_list, img_metas):
        """Forward pass for proposal bag optimization."""
        optimized_proposals = []

        for i, proposals in enumerate(proposal_list):
            if len(proposals) == 0:
                optimized_proposals.append(proposals)
                continue

            img_shape = img_metas[i]['img_shape']

            # Generate multi-scale proposals
            relaxed_proposals = self.scale_relaxation(proposals, img_shape)
            all_proposals = [proposals] + relaxed_proposals

            # Extract features for each scale
            scale_feats = []
            for props in all_proposals:
                rois = bbox2roi([props])
                feats = self.roi_extractor(
                    x[:self.roi_extractor.num_inputs],
                    rois
                )
                scale_feats.append(feats)

            # Concatenate and fuse
            concat_feats = torch.cat(scale_feats, dim=1)
            fused_feats = self.scale_conv(concat_feats)

            # SE module
            global_feats = fused_feats.mean(dim=[2, 3])
            se_weights = self.se_fc(global_feats)
            se_feats = fused_feats * se_weights.view(-1, fused_feats.size(1), 1, 1)

            # Regression
            se_feats = se_feats.flatten(1)
            shared_feats = self.relu(self.shared_fc1(se_feats))
            shared_feats = self.relu(self.shared_fc2(shared_feats))

            delta = self.reg_fc(shared_feats)
            refined_proposals = proposals + delta.view_as(proposals)

            # Clip to image boundaries
            refined_proposals[:, 0] = refined_proposals[:, 0].clamp(0, img_shape[1])
            refined_proposals[:, 1] = refined_proposals[:, 1].clamp(0, img_shape[0])
            refined_proposals[:, 2] = refined_proposals[:, 2].clamp(0, img_shape[1])
            refined_proposals[:, 3] = refined_proposals[:, 3].clamp(0, img_shape[0])

            optimized_proposals.append(refined_proposals)

        return optimized_proposals

    def forward_train(self, x, proposal_list, gt_bboxes, img_metas):
        """Training forward pass with loss computation."""
        optimized_proposals = self.forward(x, proposal_list, img_metas)

        loss_reg = 0
        num_valid = 0
        for i, (optimized, gt) in enumerate(zip(optimized_proposals, gt_bboxes)):
            if len(optimized) > 0 and len(gt) > 0:
                loss_reg += F.l1_loss(optimized, gt[:len(optimized)])
                num_valid += 1

        losses = {'loss_sipb_reg': loss_reg / num_valid if num_valid > 0 else 0}
        return losses, optimized_proposals