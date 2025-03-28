#CCCS
INF = 100000000

import random
from sklearn import metrics
import scipy.sparse as sp
import warnings
import math
try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    mat = mat.cpu().detach().numpy()
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust

class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        locations = locations.cuda()

        object_sizes_of_interest= object_sizes_of_interest.cuda()
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):

            targets_per_im = targets[im_i]

            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox.cuda()
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)


            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = ((top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))* \
                     ((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]))
        return torch.sqrt(centerness)

    def replace_targets(self,locations, box_cls, box_regression, centerness, targets):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        reg_targets_flatten = []
        labels, reg_targets = self.prepare_targets(locations, targets)
        tmp = []
        for l in range(len(labels)):
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            tmp.append(reg_targets[l].size(0))
        reg_targets_flatten = torch.cat(reg_targets_flatten,dim=0)
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
        centerness_targets_list = []
        k = 0
        for i in tmp:
            centerness_targets_list.append(centerness_targets[k:k+i])
            k += i
        box_cls_gt = []
        box_reg_gt = []
        box_ctr_gt = []
        for l in range(len(labels)):
            n, c, h, w = box_cls[l].size()
            if c >len(labels):
                c=c-1
            lb = F.one_hot(labels[l].reshape(-1), 9)[:,1:].float()
            box_cls_gt.append(lb.reshape(n,h,w,c).permute(0,3,1,2).cuda())
            box_reg_gt.append(reg_targets[l].reshape(-1).reshape(n,h,w,4).permute(0,3,1,2).cuda())
            box_ctr_gt.append(centerness_targets_list[l].reshape(-1).reshape(n,h,w,1).permute(0,3,1,2).float().cuda())
        return box_cls_gt, box_reg_gt, box_ctr_gt


    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []

        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss

def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator

class PrototypeComputation(object):
    """
    This class conducts the node sampling.
    """

    def __init__(self, cfg):
        self.num_class = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.num_class_fgbg = cfg.MODEL.FCOS.NUM_CLASSES
        self.cfg =cfg.clone()
        self.class_threshold = 0.5 #cfg.SOLVER.MIDDLE_HEAD.PLABEL_TH
        self.num_nodes_per_class = cfg.MODEL.MIDDLE_HEAD.GM.NUM_NODES_PER_LVL_SR
        self.num_nodes_per_lvl = cfg.MODEL.MIDDLE_HEAD.GM.NUM_NODES_PER_LVL_TG
        self.bg_ratio = cfg.MODEL.MIDDLE_HEAD.GM.BG_RATIO
    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

        return labels_level_first
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0


            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = ( (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])) * \
                     ((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]))
        return torch.sqrt(centerness)


    def __call__(self, locations, features, targets, centerness = None):

        if locations:  # Sampling in the source domain

            N, C, _, _ = features[0].size()
            gt_reg = []
            gt_cls = []
            for img_idx, _ in enumerate(targets):
                targets_per_im = targets[img_idx]
                assert targets_per_im.mode == "xyxy"
                bboxes = targets_per_im.bbox.cuda()
                labels_per_im = targets_per_im.get_field("labels")
                gt_reg.append(bboxes)
                gt_cls.append(labels_per_im)

            labels = self.prepare_targets(locations, targets)

            pos_points_all = []
            pos_labels_all = []
            neg_points_all = []
            centerness_all = []
            for l in range(len(labels)):
                pos_indx = labels[l].reshape(-1) > 0
                neg_indx = labels[l].reshape(-1) == 0

                # Sparse sampling to save GPU memory

                pos_nodes_in = features[l].permute(0, 2, 3, 1).reshape(-1, C)[pos_indx]
                centerness_in = centerness[l].permute(0, 2, 3, 1).reshape(-1, 1)[pos_indx]

                pos_labels_in = labels[l][pos_indx]


                pos_points_all.append(pos_nodes_in[::])
                pos_labels_all.append(pos_labels_in[::])
                centerness_all.append(centerness_in[::])


                # Sampling Background Nodes
                if self.cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG:
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                    neg_points_all.append(neg_points_temp[::])

            pos_points = torch.cat(pos_points_all, dim=0)
            pos_labels = torch.cat(pos_labels_all, dim=0)
            neg_points = torch.cat(neg_points_all, dim=0)
            pos_cen = torch.cat(centerness_all, dim=0)

            if len(pos_points) > 200:
                A, orig_dist = clust_rank(pos_points)
                u, num_clust = get_clust(A, orig_dist)
                cluster_label = np.unique(u)
                cluster_indx = []
                index_mat = np.array(range(len(pos_points)))  # temp mat for index
                cluster_cen_var=[]
                for i in cluster_label:
                    clabel_indx = index_mat[u == i]  # get bool mat
                    # find max centerness
                    find_c_mat = pos_cen[clabel_indx]
                    find_num = index_mat[clabel_indx][torch.argmax(find_c_mat)]
                    cluster_indx.append(find_num)
                    cluster_cen_var.append(torch.var(find_c_mat).unsqueeze(0))

                cluster_indx = random.choices(cluster_indx,k=200)
                pos_points = pos_points[cluster_indx]
                pos_labels = pos_labels[cluster_indx]


                loss_cluster = torch.cat(cluster_cen_var, dim=0)
                loss_cluster = torch.mean(loss_cluster)



            else:

                loss_cluster=torch.tensor([0.0],device="cuda:0",requires_grad=True)


            try:
                neg_points = neg_points[:int(len(pos_points)/10)]
            except:
                pass


            if self.cfg.MODEL.MIDDLE_HEAD.PROTO_WITH_BG:
                neg_labels = pos_labels.new_zeros((neg_points.size(0)))
                pos_points = torch.cat([neg_points, pos_points], dim=0)
                pos_labels = torch.cat([neg_labels, pos_labels])
            return pos_points, pos_labels, pos_labels.new_ones(pos_labels.shape).long(), loss_cluster, gt_reg, gt_cls

        else: # Sampling in the target domain
            act_maps_lvl_first = targets
            N, C, _, _ = features[0].size()
            N, Cls, _, _ = targets[0].size()
            neg_points =[]
            pos_plabels = []
            pos_points = []
            pos_weight = []
            centerness_all = []
            for l, feature in enumerate(features):
                act_maps = act_maps_lvl_first[l].permute(0, 2, 3, 1).reshape(-1, self.num_class)
                conf_pos_indx = (act_maps > self.class_threshold).sum(dim=-1).bool()
                neg_indx = ~((act_maps > 0.05).sum(dim=-1).bool())



                # Balanced sampling BG pixels
                if conf_pos_indx.any():
                    pos_points.append(features[l].permute(0, 2, 3, 1).reshape(-1, C)[conf_pos_indx])
                    centerness_all.append(centerness[l].permute(0, 2, 3, 1).reshape(-1, 1)[conf_pos_indx])

                    scores, indx = act_maps[conf_pos_indx,:].max(-1)

                    pos_plabels.append(indx + 1)
                    pos_weight.append(scores.detach())
                    neg_points_temp = features[l].permute(0, 2, 3, 1).reshape(-1, C)[neg_indx]
                    num_pos = len(scores)
                    neg_indx_new = list(np.floor(np.linspace(0, (neg_indx.sum()- 2).item(), (num_pos//self.bg_ratio))).astype(int))
                    neg_points.append(neg_points_temp[neg_indx_new])

            if len(pos_points) > 0:
                pos_points = torch.cat(pos_points,dim=0)
                pos_plabels = torch.cat(pos_plabels,dim=0)
                neg_points = torch.cat(neg_points, dim=0)
                pos_weight = torch.cat(pos_weight, dim=0)
                pos_cen = torch.cat(centerness_all, dim=0)


                A, orig_dist = clust_rank(pos_points)
                u, num_clust = get_clust(A, orig_dist)
                cluster_label = np.unique(u)
                cluster_indx = []
                cluster_cen_var = []
                index_mat = np.array(range(len(pos_points)))  # temp mat for index

                for i in cluster_label:
                    clabel_indx = index_mat[u == i]  # get bool mat

                    find_c_mat = pos_cen[clabel_indx]
                    find_num = index_mat[clabel_indx][torch.argmax(find_c_mat)]
                    cluster_indx.append(find_num)

                    cluster_cen_var.append(torch.var(find_c_mat).unsqueeze(0))

                loss_cluster = torch.cat(cluster_cen_var, dim=0)
                loss_cluster = torch.mean(loss_cluster)

                if np.isnan(loss_cluster.item()):
                    #print("get it nan")
                    loss_cluster = torch.tensor([0.0],device="cuda:0", requires_grad=True)

                #print("target points 1",len(pos_points))

                if len(pos_points) > 200:
                    cluster_indx = random.choices(cluster_indx, k=200)
                #print("i change num")
                pos_points = pos_points[cluster_indx]
                pos_plabels = pos_plabels[cluster_indx]
                pos_weight = pos_weight[cluster_indx]
                #
                # print("pos_points", pos_points.size())
                # print("pos_plabels", pos_plabels.size())
            try:
                neg_points = neg_points[:int(len(pos_points)/10)]
            except:
                pass
            # print(end-start)
            #print("target points 2", len(pos_points))

            if len(pos_points) > 0:
                pass
            else:
                loss_cluster = torch.tensor([0.0],device="cuda:0",requires_grad=True)


            if len(pos_points)>0:
                neg_plabels = pos_plabels.new_zeros((neg_points.size(0)))
                neg_weight = pos_weight.new_ones(neg_points.size(0)) * 0.5
                points = torch.cat([neg_points, pos_points], dim=0)
                plabels = torch.cat([neg_plabels, pos_plabels])

                loss_weight = torch.cat([neg_weight, pos_weight])
                return points, plabels, loss_weight.long(),loss_cluster
            else:
                return None, None, None,loss_cluster

def make_prototype_evaluator(cfg):
    prototype_evaluator = PrototypeComputation(cfg)
    return prototype_evaluator
