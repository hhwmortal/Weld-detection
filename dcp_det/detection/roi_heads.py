from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops
from ..modules.dcp_evc_fpn import LocalContrastPriorExtractor





def calc_diou(bboxes1, bboxes2):
    """Return DIoU for aligned pairs of boxes in xyxy format."""
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter = (br - tl).clamp(min=0)
    inter_area = inter[:, 0] * inter[:, 1]

    w1, h1 = bboxes1[:, 2] - bboxes1[:, 0], bboxes1[:, 3] - bboxes1[:, 1]
    w2, h2 = bboxes2[:, 2] - bboxes2[:, 0], bboxes2[:, 3] - bboxes2[:, 1]
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    center1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2
    center2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2
    d2 = (center1 - center2).pow(2).sum(dim=1)

    enclose_tl = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_br = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    c2 = (enclose_br - enclose_tl).pow(2).sum(dim=1).clamp(min=1e-6)

    diou = iou - (d2 / c2)
    return diou


def mean_contrast_in_matched_gt(contrast_maps, matched_gt_boxes, labels, image_shapes):
    """Return one detached DCP mean for each positive sampled RoI."""
    q_values = []
    for image_index, (gt_boxes, labels_in_image, image_shape) in enumerate(
            zip(matched_gt_boxes, labels, image_shapes)):
        positive = labels_in_image > 0
        boxes = gt_boxes[positive].detach()
        if boxes.numel() == 0:
            continue

        valid_h, valid_w = image_shape
        contrast = contrast_maps[image_index, 0, :valid_h, :valid_w].detach()
        integral = F.pad(contrast, (1, 0, 1, 0)).cumsum(0).cumsum(1)

        x1 = boxes[:, 0].floor().long().clamp(min=0, max=max(valid_w - 1, 0))
        y1 = boxes[:, 1].floor().long().clamp(min=0, max=max(valid_h - 1, 0))
        x2 = boxes[:, 2].ceil().long().clamp(min=1, max=valid_w)
        y2 = boxes[:, 3].ceil().long().clamp(min=1, max=valid_h)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        region_sum = (
            integral[y2, x2]
            - integral[y1, x2]
            - integral[y2, x1]
            + integral[y1, x1]
        )
        region_area = ((x2 - x1) * (y2 - y1)).to(region_sum.dtype)
        q_values.append((region_sum / region_area.clamp(min=1.0)).clamp(0.0, 1.0))

    if not q_values:
        return contrast_maps.new_empty((0,))
    return torch.cat(q_values, dim=0)


def fastrcnn_loss(
    class_logits,
    box_regression,
    labels,
    regression_targets,
    proposals,
    box_coder,
    matched_gt_boxes,
    contrast_maps,
    image_shapes,
    dcp_lambda=1.0,
    dcp_eps=1e-6,
):
    """Compute classification loss and the final DCP-weighted DIoU loss."""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]

    N = class_logits.shape[0]
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    pred_offsets = box_regression[sampled_pos_inds_subset, labels_pos]
    gt_offsets = regression_targets[sampled_pos_inds_subset]

    if pred_offsets.numel() > 0:
        proposals_tensor = torch.cat(proposals, dim=0)
        pos_proposals = proposals_tensor[sampled_pos_inds_subset]

        pred_boxes = box_coder.decode(pred_offsets, [pos_proposals])
        pred_boxes = pred_boxes.reshape(-1, 4)

        gt_boxes = box_coder.decode(gt_offsets, [pos_proposals])
        gt_boxes = gt_boxes.reshape(-1, 4)

        diou = calc_diou(pred_boxes, gt_boxes)
        diou_loss_per_positive = 1.0 - diou
        q_values = mean_contrast_in_matched_gt(
            contrast_maps,
            matched_gt_boxes,
            labels.split([len(item) for item in proposals]),
            image_shapes,
        )
        if q_values.numel() != diou_loss_per_positive.numel():
            raise RuntimeError(
                "DCP positive count mismatch: q={} diou={}".format(
                    q_values.numel(), diou_loss_per_positive.numel()
                )
            )
        weights = (1.0 + float(dcp_lambda) * (1.0 - q_values)).detach()
        weighted_numerator = (weights * diou_loss_per_positive).sum()
        box_loss = weighted_numerator / (weights.sum() + float(dcp_eps))
        dcp_statistics = {
            "q": q_values.detach(),
            "weights": weights.detach(),
            "unweighted_diou": diou_loss_per_positive.detach(),
            "labels": labels_pos.detach(),
            "weighted_numerator": weighted_numerator.detach(),
            "weight_sum": weights.sum().detach(),
        }

    else:
        box_loss = torch.tensor(0.0, device=class_logits.device)
        dcp_statistics = None

    return classification_loss, box_loss, dcp_statistics


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img,   # default: 100
                 dcp_lambda=1.0,
                 dcp_eps=1e-6,
                 prior_kernel_size=31,
                 image_mean=None,
                 image_std=None):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100
        self.dcp_lambda = float(dcp_lambda)
        self.dcp_eps = float(dcp_eps)
        self.dcp_prior_extractor = LocalContrastPriorExtractor(
            kernel_size=prior_kernel_size,
            prior_input="normalized_padded",
            image_mean=image_mean,
            image_std=image_std,
        )
        self.reset_dcp_statistics()

    def reset_dcp_statistics(self):
        self._dcp_statistics = {
            "count": 0,
            "q_sum": 0.0,
            "q_sq_sum": 0.0,
            "q_min": float("inf"),
            "q_max": float("-inf"),
            "weight_sum": 0.0,
            "weight_min": float("inf"),
            "weight_max": float("-inf"),
            "diou_sum": 0.0,
            "weighted_numerator": 0.0,
            "class_q": {},
        }
        self.last_dcp_batch_statistics = None

    def _update_dcp_statistics(self, batch_statistics):
        if batch_statistics is None or batch_statistics["q"].numel() == 0:
            return
        q_values = batch_statistics["q"]
        weights = batch_statistics["weights"]
        diou_values = batch_statistics["unweighted_diou"]
        labels = batch_statistics["labels"]
        count = int(q_values.numel())

        stats = self._dcp_statistics
        stats["count"] += count
        stats["q_sum"] += float(q_values.sum().item())
        stats["q_sq_sum"] += float((q_values * q_values).sum().item())
        stats["q_min"] = min(stats["q_min"], float(q_values.min().item()))
        stats["q_max"] = max(stats["q_max"], float(q_values.max().item()))
        stats["weight_sum"] += float(weights.sum().item())
        stats["weight_min"] = min(stats["weight_min"], float(weights.min().item()))
        stats["weight_max"] = max(stats["weight_max"], float(weights.max().item()))
        stats["diou_sum"] += float(diou_values.sum().item())
        stats["weighted_numerator"] += float(batch_statistics["weighted_numerator"].item())

        for class_id in labels.unique().tolist():
            class_values = q_values[labels == int(class_id)]
            class_stats = stats["class_q"].setdefault(
                int(class_id), {"count": 0, "sum": 0.0, "sq_sum": 0.0})
            class_stats["count"] += int(class_values.numel())
            class_stats["sum"] += float(class_values.sum().item())
            class_stats["sq_sum"] += float((class_values * class_values).sum().item())

        self.last_dcp_batch_statistics = {
            "number_positive_rois": count,
            "q_min": float(q_values.min().item()),
            "q_mean": float(q_values.mean().item()),
            "q_max": float(q_values.max().item()),
            "weight_min": float(weights.min().item()),
            "weight_mean": float(weights.mean().item()),
            "weight_max": float(weights.max().item()),
            "mean_unweighted_diou": float(diou_values.mean().item()),
            "weighted_dcp_diou": float(
                batch_statistics["weighted_numerator"].item()
                / (batch_statistics["weight_sum"].item() + self.dcp_eps)),
        }

    def get_dcp_statistics(self):
        stats = self._dcp_statistics
        count = stats["count"]
        if count == 0:
            return None
        mean_q = stats["q_sum"] / count
        variance_q = max(stats["q_sq_sum"] / count - mean_q * mean_q, 0.0)
        class_q = {}
        for class_id, item in sorted(stats["class_q"].items()):
            class_mean = item["sum"] / item["count"]
            class_variance = max(
                item["sq_sum"] / item["count"] - class_mean * class_mean, 0.0)
            class_q[str(class_id)] = {
                "count": item["count"],
                "mean_q": class_mean,
                "std_q": class_variance ** 0.5,
            }
        return {
            "number_positive_rois": count,
            "mean_q": mean_q,
            "std_q": variance_q ** 0.5,
            "min_q": stats["q_min"],
            "max_q": stats["q_max"],
            "mean_weight": stats["weight_sum"] / count,
            "min_weight": stats["weight_min"],
            "max_weight": stats["weight_max"],
            "mean_unweighted_diou": stats["diou_sum"] / count,
            "mean_weighted_dcp_diou": (
                stats["weighted_numerator"] / (stats["weight_sum"] + self.dcp_eps)),
            "class_q": class_q,
        }

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应标签
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        list元素个数为batch_size
        Args:
            proposals: rpn预测的boxes
            targets:

        Returns:

        """

        # 检查target数据是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取标注好的boxes以及labels信息
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets, matched_gt_boxes

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        （1）根据proposal以及预测的回归参数计算出最终bbox坐标
        （2）对预测类别结果进行softmax处理
        （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        （4）移除所有背景信息
        （5）移除低概率目标
        （6）移除小尺寸目标
        （7）执行nms处理，并按scores进行排序
        （8）根据scores排序返回前topk个目标
        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal
            image_shapes: 打包成batch前每张图像的宽高

        Returns:

        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None,   # type: Optional[List[Dict[str, Tensor]]]
                transformed_images=None
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            proposals, labels, regression_targets, matched_gt_boxes = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_gt_boxes = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        # box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # 通过roi_pooling后的两层全连接层
        # box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)

        # 接着分别预测目标类别和边界框回归参数
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            if transformed_images is None:
                raise ValueError("DCP-DIoU requires the transformed padded image batch")
            with torch.no_grad():
                contrast_maps = self.dcp_prior_extractor(transformed_images)
            loss_classifier, loss_box_reg, dcp_statistics = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets,
                proposals, self.box_coder,
                matched_gt_boxes=matched_gt_boxes,
                contrast_maps=contrast_maps,
                image_shapes=image_shapes,
                dcp_lambda=self.dcp_lambda,
                dcp_eps=self.dcp_eps)
            self._update_dcp_statistics(dcp_statistics)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses
