# --------------------------------------------------------
# SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection (CVPR22-ORAL)
# Written by Wuyang Li
# Based on https://github.com/CityU-AIM-Group/SCAN/blob/main/fcos_core/modeling/rpn/fcos/condgraph.py
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from .loss import make_prototype_evaluator
from fcos_core.layers import  BCEFocalLoss, MultiHeadAttention, Affinity
import sklearn.cluster as cluster
from fcos_core.modeling.discriminator.layer import GradientReversal
import logging
from .Generate import *
import cv2
import os
import torchvision.transforms as transforms
import numpy as np
import clip
from PIL import Image
import tqdm
from torchvision.transforms import functional as Fvis

class OrthogonalProjectionLoss(nn.Module):
	def __init__(self, gamma=0.5):
		super(OrthogonalProjectionLoss, self).__init__()
		self.gamma = gamma

	def forward(self, features, labels=None):
		device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

		#  features are normalized
		features = F.normalize(features, p=2, dim=1)

		labels = labels[:, None]  # extend dim

		mask = torch.eq(labels, labels.t()).bool().to(device)
		eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

		mask_pos = mask.masked_fill(eye, 0).float()
		mask_neg = (~mask).float()
		dot_prod = torch.matmul(features, features.t())

		pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
		neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

		loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

		return loss

class GRAPHHead(torch.nn.Module):
    # Project the sampled visual features to the graph embeddings:
    # visual features: [0,+INF) -> graph embedding: (-INF, +INF)
    def __init__(self, cfg, in_channels, out_channel, mode='in'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(GRAPHHead, self).__init__()
        if mode == 'in':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_IN
        elif mode == 'out':
            num_convs = cfg.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT
        else:
            num_convs = cfg.MODEL.FCOS.NUM_CONVS
            print('undefined num_conv in middle head')

        middle_tower = []
        for i in range(num_convs):
            middle_tower.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            if mode == 'in':
                if cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'GN':
                    middle_tower.append(nn.GroupNorm(32, in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'IN':
                    middle_tower.append(nn.InstanceNorm2d(in_channels))
                elif cfg.MODEL.MIDDLE_HEAD.IN_NORM == 'BN':
                    middle_tower.append(nn.BatchNorm2d(in_channels))
            if i != (num_convs - 1):
                middle_tower.append(nn.ReLU())

        self.add_module('middle_tower', nn.Sequential(*middle_tower))

        for modules in [self.middle_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        middle_tower = []
        for l, feature in enumerate(x):
            middle_tower.append(self.middle_tower(feature))
        return middle_tower

class GModule(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(GModule, self).__init__()

        init_item = []
        self.cfg = cfg.clone()
        self.logger = logging.getLogger("fcos_core.trainer")
        self.logger.info('node dis setting: ' + str(cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE))

        self.fpn_strides                = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_classes                = cfg.MODEL.FCOS.NUM_CLASSES
        
        # One-to-one (o2o) matching or many-to-many (m2m) matching?
        self.matching_cfg               = cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_CFG # 'o2o' and 'm2m'
        self.with_cluster_update        = cfg.MODEL.MIDDLE_HEAD.GM.WITH_CLUSTER_UPDATE # add spectral clustering to update seeds
        self.with_semantic_completion   = cfg.MODEL.MIDDLE_HEAD.GM.WITH_SEMANTIC_COMPLETION # generate hallucination nodes
        
        # add quadratic matching constraints.
        #TODO qudratic matching is not very stable in end-to-end training
        self.with_quadratic_matching    = cfg.MODEL.MIDDLE_HEAD.GM.WITH_QUADRATIC_MATCHING

        # Several weights hyper-parameters
        self.weight_matching            = cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_WEIGHT
        self.weight_nodes               = cfg.MODEL.MIDDLE_HEAD.GM.NODE_LOSS_WEIGHT
        self.weight_dis                 = cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_WEIGHT
        self.lambda_dis                 = cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_LAMBDA

        # Detailed settings
        self.with_domain_interaction    = cfg.MODEL.MIDDLE_HEAD.GM.WITH_DOMAIN_INTERACTION
        self.with_complete_graph        = cfg.MODEL.MIDDLE_HEAD.GM.WITH_COMPLETE_GRAPH
        self.with_node_dis              = cfg.MODEL.MIDDLE_HEAD.GM.WITH_NODE_DIS
        self.with_global_graph          = cfg.MODEL.MIDDLE_HEAD.GM.WITH_GLOBAL_GRAPH

        # Test 3 positions to put the node alignment discriminator. (the former is better)
        self.node_dis_place             = cfg.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE

        # future work
        self.with_cond_cls              = cfg.MODEL.MIDDLE_HEAD.GM.WITH_COND_CLS # use conditional kernel for node classification? (didn't use)
        self.with_score_weight          = cfg.MODEL.MIDDLE_HEAD.GM.WITH_SCORE_WEIGHT # use scores for node loss (didn't use)

        # Node sampling
        self.graph_generator            = make_prototype_evaluator(self.cfg)

        # Pre-processing for the vision-to-graph transformation
        self.head_in_cfg =  cfg.MODEL.MIDDLE_HEAD.IN_NORM
        if self.head_in_cfg != 'LN':
            self.head_in = GRAPHHead(cfg, in_channels, in_channels, mode='in')
        else:
            self.head_in_ln = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256, elementwise_affine=False),
            )
            init_item.append('head_in_ln')

        # node classification layers
        self.node_cls_middle = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        ) 
        init_item.append('node_cls_middle')
        
        # Graph-guided Memory Bank
        self.seed_project_left = nn.Linear(256, 256) # projection layer for the node completion
        self.register_buffer('sr_seed', torch.randn(self.num_classes, 256)) # seed = bank
        self.register_buffer('tg_seed', torch.randn(self.num_classes, 256))

        # We directly utilize the singe-head attention for the graph aggreagtion and cross-graph interaction, 
        # which will be improved in our future work
        self.cross_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2') # Cross Graph Interaction
        self.intra_domain_graph = MultiHeadAttention(256, 1, dropout=0.1, version='v2') # Intra-domain graph aggregation

        # Semantic-aware Node Affinity
        self.node_affinity = Affinity(d=256)
        self.InstNorm_layer = nn.InstanceNorm2d(1)
        # CLIP
        # CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.template_list = [
            "This is a {}",
            "There is a {}",
            "a photo of a {} in the scene",
            "a photo of a small {} in the scene",
            "a photo of a medium {} in the scene",
            "a photo of a large {} in the scene",
            "a photo of a {}",
            "a photo of a small {}",
            "a photo of a medium {}",
            "a photo of a large {}",
            "This is a photo of a {}",
            "This is a photo of a small {}",
            "This is a photo of a medium {}",
            "This is a photo of a large {}",
            "There is a {} in the scene",
            "There is the {} in the scene",
            "There is one {} in the scene",
            "This is a {} in the scene",
            "This is the {} in the scene",
            "This is one {} in the scene",
            "This is one small {} in the scene",
            "This is one medium {} in the scene",
            "This is one large {} in the scene",
            "There is a small {} in the scene",
            "There is a medium {} in the scene",
            "There is a large {} in the scene",
            "There is a {} in the photo",
            "There is the {} in the photo",
            "There is one {} in the photo",
            "There is a small {} in the photo",
            "There is the small {} in the photo",
            "There is one small {} in the photo",
            "There is a medium {} in the photo",
            "There is the medium {} in the photo",
            "There is one medium {} in the photo",
            "There is a large {} in the photo",
            "There is the large {} in the photo",
            "There is one large {} in the photo",
            "There is a {} in the picture",
            "There is the {} in the picture",
            "There is one {} in the picture",
            "There is a small {} in the picture",
            "There is the small {} in the picture",
            "There is one small {} in the picture",
            "There is a medium {} in the picture",
            "There is the medium {} in the picture",
            "There is one medium {} in the picture",
            "There is a large {} in the picture",
            "There is the large {} in the picture",
            "There is one large {} in the picture",
            "This is a {} in the photo",
            "This is the {} in the photo",
            "This is one {} in the photo",
            "This is a small {} in the photo",
            "This is the small {} in the photo",
            "This is one small {} in the photo",
            "This is a medium {} in the photo",
            "This is the medium {} in the photo",
            "This is one medium {} in the photo",
            "This is a large {} in the photo",
            "This is the large {} in the photo",
            "This is one large {} in the photo",
            "This is a {} in the picture",
            "This is the {} in the picture",
            "This is one {} in the picture",
            "This is a small {} in the picture",
            "This is the small {} in the picture",
            "This is one small {} in the picture",
            "This is a medium {} in the picture",
            "This is the medium {} in the picture",
            "This is one medium {} in the picture",
            "This is a large {} in the picture",
            "This is the large {} in the picture",
            "This is one large {} in the picture",
        ]

        self.COCO_CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
        # [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "car"}, {"id": 3, "name": "truck"}, {"id": 4, "name": "bus"}, {"id": 5, "name": "motorcycle"}, {"id": 6, "name": "bicycle"}]
        # {"id": 1, "name": "pedestrian"}, {"id": 2, "name": "rider"}, {"id": 3, "name": "car"}, {"id": 4, "name": "truck"}, {"id": 5, "name": "bus"}, {"id": 6, "name": "train"}, {"id": 7, "name": "motorcycle"}, {"id": 8, "name": "bicycle"}, {"id": 9, "name": "traffic light"}, {"id": 10, "name": "traffic sign"}]
        self.projection = nn.Linear(256, 512)
        self.temperature = 0.01
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.clip_model.eval()
        self.text_features_for_classes_list = []
        for child in self.clip_model.children():
            for param in child.parameters():
                param.requires_grad = False
        for template in tqdm.tqdm(self.template_list):
            text_features_for_classes = torch.cat(
                [self.clip_model.encode_text(clip.tokenize(template.format(c)).to(self.device)).detach() for c in
                 self.COCO_CLASSES])
            self.text_features_for_classes_list.append(F.normalize(text_features_for_classes, dim=-1))
        self.text_features_for_classes_list = torch.stack(self.text_features_for_classes_list).mean(dim=0)
        self.text_features_for_classes_list = self.text_features_for_classes_list.float()
        self.text_features_for_classes_list = F.normalize(self.text_features_for_classes_list, dim=-1)
        # loss
        self.supconloss = OrthogonalProjectionLoss()  # numbranch随类别数变化
        self.supconloss_weight = 1e-3
        # Structure-aware Matching Loss
        # Different matching loss choices
        if cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'L1':
            self.matching_loss = nn.L1Loss(reduction='sum')
        elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'MSE':
            self.matching_loss = nn.MSELoss(reduction='sum')
        elif cfg.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG == 'FL':
            self.matching_loss = BCEFocalLoss()
        self.quadratic_loss = torch.nn.L1Loss(reduction='mean')

        if self.with_node_dis: 
            self.grad_reverse = GradientReversal(self.lambda_dis)
            self.node_dis_2 = nn.Sequential(
                nn.Linear(256,256),
                nn.LayerNorm(256,elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.LayerNorm(256,elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256,elementwise_affine=False),
                nn.ReLU(),
                nn.Linear(256,1)
            )
            init_item.append('node_dis')
            self.loss_fn = nn.BCEWithLogitsLoss()
        self._init_weight(init_item)

        # Generation way
        self.SeNode_generator = Generator2(256, 256).cuda()


    def _init_weight(self, init_item=None):
        nn.init.normal_(self.seed_project_left.weight, std=0.01)
        nn.init.constant_(self.seed_project_left.bias, 0)
        if 'node_dis' in init_item:
            for i in self.node_dis_2:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_dis initialized')
        if 'node_cls_middle' in init_item:
            for i in self.node_cls_middle:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('node_cls_middle initialized')
        if 'head_in_ln' in init_item:
            for i in self.head_in_ln:
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, std=0.01)
                    nn.init.constant_(i.bias, 0)
            self.logger.info('head_in_ln initialized')

    def forward(self, images, features, targets=None, score_maps=None,centerness = None):
        '''
        We have equal number of source/target feature maps
        features: [sr_feats, tg_feats]
        targets: [sr_targets, None]

        '''
        if targets:
            features, feat_loss = self._forward_train(images, features, targets, score_maps,centerness)
            return features, feat_loss

        else:
            features = self._forward_inference(images, features)
            return features, None

    def _forward_train(self, images, features, targets=None, score_maps=None,centerness = None):
            features_s, features_t = features
            middle_head_loss = {}
            if centerness:
                centerness_s, centerness_t = centerness
            # STEP1: sample pixels and generate semantic incomplete graph nodes
            # node_1 and node_2 mean the source/target raw nodes
            # label_1 and label_2 mean the GT and pseudo labels
            nodes_1, labels_1, weights_1, loss_cluster_s, gt_reg, gt_cls = self.graph_generator(
                self.compute_locations(features_s), features_s, targets, centerness_s
            )#mark
            nodes_2, labels_2, weights_2 ,loss_cluster_t= self.graph_generator(
                None, features_t, score_maps,centerness_t
            )
            if nodes_2 is not None:
                # print((torch.cat(gt_cls)-1)tolist.())
                # src CLIP op
                # visualize imgae w GT
                images_s, images_t = images  # unnormal
                to_pil = transforms.ToPILImage()
                mean = [-i for i in self.cfg.INPUT.PIXEL_MEAN]
                std = [1.0, 1.0, 1.0]
                image_list = []
                # denormalize
                for img in images_s.tensors:
                    img = Fvis.normalize(img.squeeze(0), mean=mean,
                                         std=std)
                    img = img[[2, 1, 0]] * 255
                    img_np = to_pil(img.cpu().detach())
                    image_list.append(img_np)
                reg_list = []
                cls_list = []
                for reg, cls in zip(gt_reg, gt_cls):
                    reg_np = reg.cpu().numpy().astype(int)
                    reg_list.append(reg_np)
                    gt_np = cls.cpu().numpy().astype(int)
                    cls_list.append(gt_np)
                clip_crop_feat_src_batch = []
                clip_crop_label_src_batch = []
                for img_v, reg_v, cls_v in zip(image_list, reg_list, cls_list):
                    # img_tmp = img_v
                    reg_tmp = reg_v
                    cls_tmp = cls_v
                    # print(cls_v)
                    preprocessed = []
                    for i in range(len(reg_tmp)):
                        x1, y1, x2, y2 = reg_tmp[i][0], reg_tmp[i][1], reg_tmp[i][2], reg_tmp[i][3]
                        # lbl = str(cls_v[i])
                        # cv2.rectangle(img_v, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # 在目标框上方打上标签
                        # cv2.putText(img_v, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        # print(x1, y1, x2, y2)
                        cropped_image = img_v.crop((x1, y1, x2, y2))
                        w, h = cropped_image.size
                        # print(cropped_image.size)
                        # print('*'*10)
                        if w > 10 and h > 10:
                            clip_crop_label_src_batch.append(cls_tmp[i])
                            croped = self.preprocess(cropped_image)
                            preprocessed.append(croped)
                    if len(preprocessed) > 0:
                        preprocessed = torch.stack(preprocessed).to(self.device)
                        clip_crop_feat_src_batch.append(
                            torch.nn.functional.normalize(self.clip_model.encode_image(preprocessed), p=2, dim=1))
                clip_crop_feat_src_batch = torch.cat(clip_crop_feat_src_batch)
                clip_crop_label_src_batch = torch.tensor(clip_crop_label_src_batch).cuda()
                # Align
                # src region feat
                clip_regionfeat = self.projection(torch.cat([nodes_1, nodes_2]))
                clip_regionfeat = torch.nn.functional.normalize(clip_regionfeat, p=2, dim=1)
                clip_label = torch.cat([labels_1, labels_2])
                fg_idx = clip_label != 0
                clip_regionfeat = clip_regionfeat[fg_idx]
                clip_label = clip_label[fg_idx]
                # print(clip_crop_label_src_batch)
                # clip_crop_label_src_batch = clip_crop_label_src_batch
                # print(clip_crop_label_src_batch)
                # print('*'*10)
                # labels_exist = set(clip_crop_label_src_batch.tolist()).intersection(set(clip_label.tolist()))
                # # print(labels_exist, set(clip_crop_label_src_batch.tolist()), set(clip_label.tolist()))
                # if len(labels_exist) != 0:
                #     text_cls_loss = []
                #     text_cnt = 0
                #     for c in labels_exist:
                #         regionfeat_idx = clip_label == c
                #         clip_src_idx = clip_crop_label_src_batch == c
                #         cls_score_text = clip_regionfeat[regionfeat_idx] @ text_features.T
                #         text_cnt += len(cls_score_text)
                #         text_cls_loss.append(
                #             F.cross_entropy(cls_score_text / self.temperature, clip_label[regionfeat_idx],
                #                             reduction='none'))

                # feature loss
                tgt_src_jointfeat = torch.cat([clip_crop_feat_src_batch, clip_regionfeat], dim=0)
                tgt_src_jointlabel = torch.cat([clip_crop_label_src_batch, clip_label], dim=0)
                #
                supconloss = self.supconloss(tgt_src_jointfeat, tgt_src_jointlabel)
                middle_head_loss.update({'compact_loss': self.supconloss_weight * supconloss})

            #loss_cluster_s.to("cuda")
            #print("all_cluster_loss", all_cluster_loss.size())
            #print("loss_cluster_s",loss_cluster_s)
            #print("loss_cluster_t", loss_cluster_t)
            middle_head_loss.update({'cluster_loss': 0.001*(loss_cluster_s+loss_cluster_t)})
            #print("(loss_cluster_s+loss_cluster_t)",(loss_cluster_s+loss_cluster_t))
            #print("torch.mean(torch.Tensor(loss_cluster_s+loss_cluster_t))",torch.mean(torch.Tensor(all_cluster_loss)))
            # to avoid the failure of extreme cases with limited bs
            if nodes_1.size(0) < 6 or len(nodes_1.size()) == 1:
                return features, middle_head_loss

            #  conduct node alignment to prevent overfit
            if self.with_node_dis and nodes_2 is not None and self.node_dis_place =='feat' :
                nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                tg_rev = torch.cat([target_1, target_2], dim=0)
                nodes_rev = self.node_dis_2(nodes_rev)
                node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                middle_head_loss.update({'dis_loss': node_dis_loss})

            # STEP2: vision-to-graph transformation 
            # LN is conducted on the node embedding
            # GN/BN are conducted on the whole image feature
            if  self.head_in_cfg != 'LN':
                features_s = self.head_in(features_s)
                features_t = self.head_in(features_t)
                nodes_1, labels_1, weights_1 = self.graph_generator(
                    self.compute_locations(features_s), features_s, targets
                )
                nodes_2, labels_2, weights_2 = self.graph_generator(
                    None, features_t, score_maps
                )
            else:
                nodes_1 = self.head_in_ln(nodes_1)
                nodes_2 = self.head_in_ln(nodes_2) if nodes_2 is not None else None

            # TODO: Matching can only work for adaptation when both source and target nodes exist. 
            # Otherwise, we split the source nodes half-to-half to train SIGMA

            if nodes_2 is not None: # Both domains have graph nodes
                #print("nodes_1,nodes_2",nodes_1.size(),nodes_2.size())
                # STEP3: Conduct Domain-guided Node Completion (DNC)
                (nodes_1, nodes_2), (labels_1, labels_2), (weights_1, weights_2) = \
                    self._forward_preprocessing_source_target((nodes_1, nodes_2), (labels_1, labels_2),(weights_1,weights_2))
                #print("After nodes_1,nodes_2", nodes_1.size(), nodes_2.size())
                #print(sadjfl)
                # STEP4: Single-layer GCN
                if self.with_complete_graph:
                    nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
                    nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

                # STEP5: Update Graph-guided Memory Bank (GMB) with enhanced node embedding
                generate_loss = self.update_seed(nodes_1, labels_1, nodes_2, labels_2)
                middle_head_loss.update({'generate_loss ': 0.01*generate_loss})

                if self.with_node_dis and self.node_dis_place =='intra':
                    nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                    target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                    target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                    tg_rev = torch.cat([target_1, target_2], dim=0)
                    nodes_rev = self.node_dis_2(nodes_rev)
                    node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                    middle_head_loss.update({'dis_loss': node_dis_loss})

                # STEP6: Conduct Cross Graph Interaction (CGI)
                if self.with_domain_interaction:
                    nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)

                if self.with_node_dis and self.node_dis_place =='inter':
                    nodes_rev = self.grad_reverse(torch.cat([nodes_1, nodes_2], dim=0))
                    target_1 = torch.full([nodes_1.size(0), 1], 1.0, dtype=torch.float, device=nodes_1.device)
                    target_2 = torch.full([nodes_2.size(0), 1], 0.0, dtype=torch.float, device=nodes_2.device)
                    tg_rev = torch.cat([target_1, target_2], dim=0)
                    nodes_rev = self.node_dis_2(nodes_rev)
                    node_dis_loss = self.weight_dis * self.loss_fn(nodes_rev.view(-1), tg_rev.view(-1))
                    middle_head_loss.update({'dis_loss': node_dis_loss})

                # STEP7:  node loss
                node_loss = self._forward_node_loss(
                    torch.cat([nodes_1, nodes_2], dim=0),
                    torch.cat([labels_1, labels_2], dim=0),
                    torch.cat([weights_1, weights_2], dim=0)
                                                    )

            else: # Use all source nodes for training if no target nodes in the early training stage
                (nodes_1, nodes_2),(labels_1, labels_2) = \
                    self._forward_preprocessing_source(nodes_1, labels_1)

                nodes_1, edges_1 = self._forward_intra_domain_graph(nodes_1)
                nodes_2, edges_2 = self._forward_intra_domain_graph(nodes_2)

                generate_loss = self.update_seed(nodes_1, labels_1, nodes_1, labels_1)
                middle_head_loss.update({'generate_loss ': 0.01*generate_loss})

                nodes_1, nodes_2 = self._forward_cross_domain_graph(nodes_1, nodes_2)
                node_loss = self._forward_node_loss(
                    torch.cat([nodes_1, nodes_2],dim=0),
                    torch.cat([labels_1, labels_2],dim=0)
                )
            middle_head_loss.update({'node_loss': self.weight_nodes * node_loss})

            # STEP8: Generate Semantic-aware Node Affinity and Structure-aware Matching loss
            if self.matching_cfg != 'none':
                matching_loss_affinity, affinity = self._forward_aff(nodes_1, nodes_2, labels_1, labels_2)
                middle_head_loss.update({'mat_loss_aff': self.weight_matching * matching_loss_affinity })

                if self.with_quadratic_matching:
                    matching_loss_quadratic = self._forward_qu(edges_1.detach(), edges_2.detach(), affinity)
                    middle_head_loss.update({'mat_loss_qu':  matching_loss_quadratic})

            return features, middle_head_loss

    def _forward_preprocessing_source_target(self, nodes, labels, weights):

        '''
        nodes: sampled raw source/target nodes
        labels: the ground-truth/pseudo-label of sampled source/target nodes
        weights: the confidence of sampled source/target nodes ([0.0,1.0] scores for target nodes and 1.0 for source nodes )

        We permute graph nodes according to the class from 1 to K and complete the missing class.

        '''

        sr_nodes, tg_nodes = nodes
        sr_nodes_label, tg_nodes_label = labels
        sr_loss_weight, tg_loss_weight = weights

        labels_exist = torch.cat([sr_nodes_label, tg_nodes_label]).unique()

        sr_nodes_category_first = []
        tg_nodes_category_first = []

        sr_labels_category_first = []
        tg_labels_category_first = []

        sr_weight_category_first = []
        tg_weight_category_first = []
        #print("self.with_semantic_completion",self.with_semantic_completion)
        noise_dim=256
        for c in labels_exist:

            sr_indx = sr_nodes_label == c
            tg_indx = tg_nodes_label == c

            sr_nodes_c = sr_nodes[sr_indx]
            tg_nodes_c = tg_nodes[tg_indx]

            sr_weight_c = sr_loss_weight[sr_indx]
            tg_weight_c = tg_loss_weight[tg_indx]

            if sr_indx.any() and tg_indx.any(): # If the category appear in both domains, we directly collect them!

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                labels_sr = sr_nodes_c.new_ones(len(sr_nodes_c)) * c
                labels_tg = tg_nodes_c.new_ones(len(tg_nodes_c)) * c

                sr_labels_category_first.append(labels_sr)
                tg_labels_category_first.append(labels_tg)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(tg_weight_c)

            elif tg_indx.any():  # If there're no source nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(tg_nodes_c)
                sr_nodes_c = self.sr_seed[c].unsqueeze(0).expand(num_nodes, 256)


                if self.with_semantic_completion:
                    rng_state = torch.get_rng_state()#全局随机种子保护

                    temp_seed = sr_nodes_c[0][0]#通讯种子
                    torch.manual_seed(temp_seed)
                    noise = torch.randn(tg_nodes_c.size(0), 256).cuda()#噪声生成
                    sr_nodes_c = noise + sr_nodes_c

                    torch.set_rng_state(rng_state)#全局随机种子保护

                    sr_nodes_c = self.SeNode_generator(sr_nodes_c)




                    # if len(tg_nodes_c) < 5:
                    #     noise = torch.randn(tg_nodes_c.size(0), noise_dim).cuda()  # 生成与 tg_nodes_c 大小匹配的随机噪声
                    #     sr_nodes_c = self.SeNode_generator(noise) + sr_nodes_c  # 用生成器生成新的节点表示并加到现有的 sr_nodes_c 上
                    # else:
                    #     noise = torch.randn(sr_nodes_c.size(0), noise_dim).cuda()  # 生成与 sr_nodes_c 大小匹配的随机噪声
                    #     mean = sr_nodes_c
                    #     std = tg_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())
                    #     generated_nodes = self.SeNode_generator(noise)  # 用生成器生成新的节点表示
                    #     sr_nodes_c = mean + generated_nodes * std  # 生成的节点表示按目标节点集合的标准差进行缩放并加到均值上
                else:
                    sr_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda()

                sr_nodes_c = self.seed_project_left(sr_nodes_c)
                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)
                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)
                sr_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).cuda())
                tg_weight_category_first.append(tg_weight_c)

            elif sr_indx.any():  # If there're no target nodes in this category, we complete it with hallucination nodes!

                num_nodes = len(sr_nodes_c)

                sr_nodes_category_first.append(sr_nodes_c)
                tg_nodes_c = self.tg_seed[c].unsqueeze(0).expand(num_nodes, 256)

                if self.with_semantic_completion:
                    # tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda() + tg_nodes_c if len(sr_nodes_c)<5 \
                    #     else torch.normal(mean=tg_nodes_c,
                    #                           std=sr_nodes_c.std(0).unsqueeze(0).expand(sr_nodes_c.size())).cuda()

                    #print("sr_nodes_c00", sr_nodes_c.size())

                    rng_state = torch.get_rng_state()#全局随机种子保护

                    temp_seed = tg_nodes_c[0][0]#通讯种子
                    torch.manual_seed(temp_seed)
                    noise = torch.randn(sr_nodes_c.size(0), 256).cuda()#噪声生成
                    tg_nodes_c = noise + tg_nodes_c
                    #print("tg_nodes_c1", tg_nodes_c.size())
                    torch.set_rng_state(rng_state)#全局随机种子保护

                    tg_nodes_c = self.SeNode_generator(tg_nodes_c)
                    #print("tg_nodes_c2",tg_nodes_c.size())

                    #print("tg over")
                    # if len(sr_nodes_c) < 5:
                    #     noise = torch.randn(sr_nodes_c.size(0), noise_dim).cuda()  # 生成与 sr_nodes_c 大小匹配的随机噪声
                    #     print("noise1", noise.size(), "sr_nodes_c", sr_nodes_c.size())
                    #     print("noise2", noise.size(), "sr_nodes_c", sr_nodes_c.size())
                    #     generated_nodes = self.SeNode_generator(noise)  # 用生成器生成新的节点表示
                    #     tg_nodes_c = generated_nodes + tg_nodes_c  # 将生成的节点表示加到现有的 tg_nodes_c 上
                    #     print("noise2", noise.size(), "sr_nodes_c", sr_nodes_c.size())
                    # else:
                    #     noise = torch.randn(tg_nodes_c.size(0), noise_dim).cuda()  # 生成与 tg_nodes_c 大小匹配的随机噪声
                    #     print("noise1", noise.size(), "sr_nodes_c", sr_nodes_c.size())
                    #     mean = tg_nodes_c
                    #     std = sr_nodes_c.std(0).unsqueeze(0).expand(tg_nodes_c.size())
                    #     generated_nodes = self.SeNode_generator(noise)  # 用生成器生成新的节点表示
                    #     tg_nodes_c = mean + generated_nodes * std  # 生成的节点表示按目标节点集合的标准差进行缩放并加到均值上
                    #     print("noise2", noise.size(), "sr_nodes_c", sr_nodes_c.size())

                else:
                    tg_nodes_c = torch.normal(0, 0.01, size=tg_nodes_c.size()).cuda()

                tg_nodes_c = self.seed_project_left(tg_nodes_c)
                tg_nodes_category_first.append(tg_nodes_c)

                sr_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)
                tg_labels_category_first.append(torch.ones(num_nodes, dtype=torch.float).cuda() * c)

                sr_weight_category_first.append(sr_weight_c)
                tg_weight_category_first.append(torch.ones(num_nodes, dtype=torch.long).cuda())

        nodes_sr = torch.cat(sr_nodes_category_first, dim=0)
        nodes_tg = torch.cat(tg_nodes_category_first, dim=0)

        weight_sr = torch.cat(sr_weight_category_first, dim=0)
        weight_tg = torch.cat(tg_weight_category_first, dim=0)

        label_sr = torch.cat(sr_labels_category_first, dim=0)
        label_tg = torch.cat(tg_labels_category_first, dim=0)

        return (nodes_sr, nodes_tg), (label_sr, label_tg), (weight_sr, weight_tg)

    def _forward_preprocessing_source(self, sr_nodes, sr_nodes_label):
        labels_exist = sr_nodes_label.unique()

        nodes_1_cls_first = []
        nodes_2_cls_first = []
        labels_1_cls_first = []
        labels_2_cls_first = []

        for c in labels_exist:
            sr_nodes_c = sr_nodes[sr_nodes_label == c]
            nodes_1_cls_first.append(torch.cat([sr_nodes_c[::2, :]]))
            nodes_2_cls_first.append(torch.cat([sr_nodes_c[1::2, :]]))

            labels_side1 = sr_nodes_c.new_ones(len(nodes_1_cls_first[-1])) * c
            labels_side2 = sr_nodes_c.new_ones(len(nodes_2_cls_first[-1])) * c

            labels_1_cls_first.append(labels_side1)
            labels_2_cls_first.append(labels_side2)

        nodes_1 = torch.cat(nodes_1_cls_first, dim=0)
        nodes_2 = torch.cat(nodes_2_cls_first, dim=0)

        labels_1 = torch.cat(labels_1_cls_first, dim=0)
        labels_2 = torch.cat(labels_2_cls_first, dim=0)

        return (nodes_1, nodes_2), (labels_1, labels_2)

    def _forward_intra_domain_graph(self, nodes):
        nodes, edges = self.intra_domain_graph(nodes, nodes, nodes)
        return nodes, edges

    def _forward_cross_domain_graph(self, nodes_1, nodes_2):

        if self.with_global_graph:
            n_1 = len(nodes_1)
            n_2 = len(nodes_2)
            global_nodes = torch.cat([nodes_1, nodes_2], dim=0)
            global_nodes = self.cross_domain_graph(global_nodes, global_nodes, global_nodes)[0]

            nodes1_enahnced = global_nodes[:n_1]
            nodes2_enahnced = global_nodes[n_1:]
        else:
            nodes2_enahnced = self.cross_domain_graph(nodes_1, nodes_1, nodes_2)[0]
            nodes1_enahnced = self.cross_domain_graph(nodes_2, nodes_2, nodes_1)[0]

        return nodes1_enahnced, nodes2_enahnced

    def _forward_node_loss(self, nodes, labels, weights=None):

        labels= labels.long()
        assert len(nodes) == len(labels)

        if weights is None:  # Source domain
            if self.with_cond_cls:
                tg_embeds = self.node_cls_middle(self.tg_seed)
                logits = self.dynamic_fc(nodes, tg_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels,
                                        reduction='mean')
        else:  # Target domain
            if self.with_cond_cls:
                sr_embeds = self.node_cls_middle(self.sr_seed)
                logits = self.dynamic_fc(nodes, sr_embeds)
            else:
                logits = self.node_cls_middle(nodes)

            node_loss = F.cross_entropy(logits, labels.long(),
                                        reduction='none')
            node_loss = (node_loss * weights).float().mean() if self.with_score_weight else node_loss.float().mean()

        return node_loss

    def update_seed(self, sr_nodes, sr_labels, tg_nodes=None, tg_labels=None):

        with torch.no_grad():
            for cls in sr_labels.unique().long():
                bs = sr_nodes[sr_labels == cls].detach()
                # lvl_sum
                bs_cls_score = F.softmax(self.node_cls_middle(bs), dim=-1)[:, cls].unsqueeze(0).T
                bs_cls = torch.sum(bs * bs_cls_score, dim=0)
                momentum_lvl = torch.nn.functional.cosine_similarity(bs_cls.unsqueeze(0),
                                                                     self.sr_seed[cls].unsqueeze(0))
                self.sr_seed[cls] = self.sr_seed[cls] * momentum_lvl + bs_cls * (1.0 - momentum_lvl)
                ####====================================================####

        kl_all=[]
        for cls in sr_labels.unique().long():
            origin_node = sr_nodes[sr_labels == cls].detach()
            nodes_nums = len(origin_node)
            if nodes_nums<10:
                #print("sr", nodes_nums, "pass")
                continue
            generate_node = self.sr_seed[cls].unsqueeze(0).expand(nodes_nums, 256)
            #生成種子
            rng_state = torch.get_rng_state()  # 全局随机种子保护
            temp_seed = self.sr_seed[cls][0]  # 通讯种子
            torch.manual_seed(temp_seed)
            noise = torch.randn(nodes_nums, 256).cuda()  # 噪声生成
            generate_node = noise + generate_node
            torch.set_rng_state(rng_state)  # 全局随机种子保护
            #
            generate_node = self.SeNode_generator(generate_node)

            # kl散度计算
            prob1 = F.softmax(origin_node, dim=1)
            prob2 = F.softmax(generate_node, dim=1)
            kl_loss = F.kl_div(prob1.log(), prob2, reduction='batchmean')
            #print("KL Divergence:", kl_div)
            kl_all.append(kl_loss.unsqueeze(0))



        with torch.no_grad():
            if tg_nodes is not None:
                for cls in tg_labels.unique().long():
                    bs = tg_nodes[tg_labels == cls].detach()
                    bs_cls_score = F.softmax(self.node_cls_middle(bs), dim=-1)[:, cls].unsqueeze(0).T
                    bs_cls = torch.sum(bs * bs_cls_score, dim=0)
                    momentum_lvl = torch.nn.functional.cosine_similarity(bs_cls.unsqueeze(0),
                                                                         self.tg_seed[cls].unsqueeze(0))
                    self.tg_seed[cls] = self.tg_seed[cls] * momentum_lvl + bs_cls * (1.0 - momentum_lvl)
                    ####====================================================####

        for cls in tg_labels.unique().long():
            origin_node = tg_nodes[tg_labels == cls].detach()
            nodes_nums = len(origin_node)
            if nodes_nums<10:
                #print("tg",nodes_nums,"pass")
                continue
            generate_node = self.tg_seed[cls].unsqueeze(0).expand(nodes_nums, 256)
            #生成種子
            rng_state = torch.get_rng_state()  # 全局随机种子保护
            temp_seed = self.tg_seed[cls][0]  # 通讯种子
            torch.manual_seed(temp_seed)
            noise = torch.randn(nodes_nums, 256).cuda()  # 噪声生成
            generate_node = noise + generate_node
            torch.set_rng_state(rng_state)  # 全局随机种子保护
            #
            generate_node = self.SeNode_generator(generate_node)

            # kl散度计算
            prob1 = F.softmax(origin_node, dim=1)
            prob2 = F.softmax(generate_node, dim=1)
            kl_loss = F.kl_div(prob1.log(), prob2, reduction='batchmean')
            #print("KL Divergence:", kl_div)
            kl_all.append(kl_loss.unsqueeze(0))


        #print("generate_loss", kl_all)
        generate_loss = torch.cat(kl_all,dim=0)
        generate_loss = torch.mean(generate_loss)
        #print("generate_loss",generate_loss)

        return generate_loss





        # ######################################################################## #
        # k = 20 # conduct clustering when we have enough graph nodes
        # for cls in sr_labels.unique().long():
        #     bs = sr_nodes[sr_labels == cls].detach()
        #
        #     if len(bs) > k and self.with_cluster_update:
        #         #TODO Use Pytorch-based GPU version
        #         sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
        #                                         assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
        #         seed_cls = self.sr_seed[cls]
        #         indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
        #         indx = (indx == indx[0])[1:]
        #         bs = bs[indx].mean(0)
        #     else:
        #         bs = bs.mean(0)
        #
        #     momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.sr_seed[cls].unsqueeze(0))
        #     self.sr_seed[cls] = self.sr_seed[cls] * momentum + bs * (1.0 - momentum)
        #
        # if tg_nodes is not None:
        #     for cls in tg_labels.unique().long():
        #         bs = tg_nodes[tg_labels == cls].detach()
        #         if len(bs) > k and self.with_cluster_update:
        #             seed_cls = self.tg_seed[cls]
        #             sp = cluster.SpectralClustering(2, affinity='nearest_neighbors', n_jobs=-1,
        #                                             assign_labels='kmeans', random_state=1234, n_neighbors=len(bs) // 2)
        #             indx = sp.fit_predict(torch.cat([seed_cls[None, :], bs]).cpu().numpy())
        #             indx = (indx == indx[0])[1:]
        #             bs = bs[indx].mean(0)
        #         else:
        #             bs = bs.mean(0)
        #         momentum = torch.nn.functional.cosine_similarity(bs.unsqueeze(0), self.tg_seed[cls].unsqueeze(0))
        #         self.tg_seed[cls] = self.tg_seed[cls] * momentum + bs * (1.0 - momentum)

    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):
        if self.matching_cfg == 'o2o':
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())

            M = self.InstNorm_layer(M[None, None, :, :])
            M = self.sinkhorn_rpm(M[:, 0, :, :], n_iters=20).squeeze().exp()

            TP_mask = (matching_target == 1).float()
            indx = (M * TP_mask).max(-1)[1]
            TP_samples = M[range(M.size(0)), indx].view(-1, 1)
            TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

            FP_samples = M[matching_target == 0].view(-1, 1)
            FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()

            # TP_loss = self.matching_loss(TP_sample, TP_target.float())
            #TODO Find a better reduction strategy
            TP_loss = self.matching_loss(TP_samples, TP_target.float())/ len(TP_samples)
            FP_loss = self.matching_loss(FP_samples, FP_target.float())/ torch.sum(FP_samples).detach()
            # print('FP: ', FP_loss, 'TP: ', TP_loss)
            matching_loss = TP_loss + FP_loss

        elif self.matching_cfg == 'm2m': # Refer to the Appendix
            M = self.node_affinity(nodes_1, nodes_2)
            matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
            matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()
        else:
            M = None
            matching_loss = 0
        return matching_loss, M

    def _forward_inference(self, images, features):
        return features

    def _forward_qu(self, edge_1, edge_2, affinity):
        R =  torch.mm(edge_1, affinity) - torch.mm(affinity, edge_2)
        loss = self.quadratic_loss(R, R.new_zeros(R.size()))
        return loss

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def sinkhorn_rpm(self, log_alpha, n_iters=5, slack=True, eps=-1):
        ''' Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)

        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        '''
        prev_alpha = None
        if slack:
            zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
            log_alpha_padded = zero_pad(log_alpha[:, None, :, :])
            log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

            for i in range(n_iters):
                # Row normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                    dim=1)
                # Column normalization
                log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                    dim=2)
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()
            log_alpha = log_alpha_padded[:, :-1, :-1]
        else:
            for i in range(n_iters):
                # Row normalization (i.e. each row sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))
                # Column normalization (i.e. each column sum to 1)
                log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))
                if eps > 0:
                    if prev_alpha is not None:
                        abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                        if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                            break
                    prev_alpha = torch.exp(log_alpha).clone()              
        return log_alpha

    def dynamic_fc(self, features, kernel_par):
        weight = kernel_par
        return torch.nn.functional.linear(features, weight, bias=None)

    def dynamic_conv(self, features, kernel_par):
        weight = kernel_par.view(self.num_classes, -1, 1, 1)
        return torch.nn.functional.conv2d(features, weight)

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long(), :].cuda()
def build_graph_matching_head(cfg, in_channels):
    return GModule(cfg, in_channels)
