import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.models.refiners import CDIPMiner, SIPBOptimizer


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 # New parameters for CDIP Miner and SIPB Optimizer
                 cdip_miner=None,
                 sipb_optimizer=None,
                 use_alternating_cascade=True):
        super(StandardRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.use_alternating_cascade = use_alternating_cascade

        # Initialize CDIP Miner
        if cdip_miner is not None:
            self.cdip_miner = build_head(cdip_miner)
        else:
            self.cdip_miner = None

        # Initialize SIPB Optimizer
        if sipb_optimizer is not None:
            self.sipb_optimizer = build_head(sipb_optimizer)
        else:
            self.sipb_optimizer = None

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      ann_weight,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Modified forward_train with CDIP Miner and SIPB Optimizer integration.
        Corresponds to the alternating cascading arrangement described in Section III-D.
        """
        # ===== Phase 0: Basic proposal bag (manual neighborhood sampling) =====
        # proposal_list is assumed to be generated from RPN or manual sampling

        # ===== Phase I: SIPB I → CDIP I =====
        if self.sipb_optimizer is not None and self.cdip_miner is not None:
            # Step 1: SIPB I optimizes proposal bag (spatial context)
            proposal_list_phase1 = self.sipb_optimizer.forward(
                x, proposal_list, img_metas)

            # Step 2: CDIP I performs proposal mining
            cdip_losses_phase1 = self.cdip_miner.forward_train(
                x, proposal_list_phase1, gt_bboxes, gt_labels, ann_weight, img_metas)
        else:
            proposal_list_phase1 = proposal_list
            cdip_losses_phase1 = {}

        # ===== Phase II: SIPB II → CDIP II =====
        if self.sipb_optimizer is not None and self.cdip_miner is not None:
            # Step 1: SIPB II further optimizes (same architecture as SIPB I)
            proposal_list_phase2 = self.sipb_optimizer.forward(
                x, proposal_list_phase1, img_metas)

            # Step 2: CDIP II final proposal mining
            cdip_losses_phase2 = self.cdip_miner.forward_train(
                x, proposal_list_phase2, gt_bboxes, gt_labels, ann_weight, img_metas)

            # Compute SIPB regression loss
            sipb_losses, proposal_list_phase2 = self.sipb_optimizer.forward_train(
                x, proposal_list_phase1, gt_bboxes, img_metas)
        else:
            proposal_list_phase2 = proposal_list_phase1
            cdip_losses_phase2 = {}
            sipb_losses = {}

        # ===== Merge CDIP and SIPB losses =====
        cdip_losses = {**cdip_losses_phase1, **cdip_losses_phase2}
        sipb_losses = {**sipb_losses}

        # ===== Standard training flow with optimized proposal bags =====
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list_phase2[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list_phase2[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()

        # BBox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels, ann_weight,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # Add CDIP and SIPB losses
        losses.update(cdip_losses)
        losses.update(sipb_losses)

        # Mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    # All other methods (init_assigner_sampler, _bbox_forward, etc.) remain unchanged