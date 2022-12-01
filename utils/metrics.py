import os, logging
from ops.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from ops.utils import points_in_boxes_gpu
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Metric:
    def __init__(self, cfg, run_path):
        self.cfg = cfg
        self.run_path = run_path

    def add_samples(self, data_dict):
        raise NotImplementedError

    def save_detections(self, filename):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


class MetricObjDet(Metric):
    def __init__(self, cfg, run_path, name='none', bev=False):
        super(MetricObjDet, self).__init__(cfg, run_path)
        self.n_coop = cfg['n_coop'] if 'n_coop' in list(cfg.keys()) else 0
        self.samples = []
        self.pred_boxes = {}
        self.gt_boxes = {}
        self.confidences = {}
        self.v_ids = {}
        self.bev = bev
        self.iou_fn = boxes_iou_bev if self.bev else boxes_iou3d_gpu
        file_test = run_path / 'thr{}_ncoop{}_{}.pth'.format(self.cfg['score_threshold'], self.n_coop, name)
        self.has_test_detections = False
        if file_test.exists():
            data = torch.load(file_test)
            self.samples = data['samples']
            self.pred_boxes = data['pred_boxes']
            self.gt_boxes = data['gt_boxes']
            self.confidences = data['confidences']
            self.v_ids = data['ids']
            self.has_test_detections = True

    def add_sample(self, name, pred_boxes, gt_boxes, confidences, ids=None):
        if len(pred_boxes)>0:
            assert pred_boxes.device.type==self.device
        if len(gt_boxes) > 0:
            assert gt_boxes.device.type==self.device
        self.samples.append(name)
        self.pred_boxes[name] = pred_boxes
        self.gt_boxes[name] = gt_boxes
        self.confidences[name] = confidences
        if ids is not None:
            self.v_ids[name] = ids

    @torch.no_grad()
    def add_samples(self, names, preds, gts, confs, ids=None):
        for i in range(len(names)):
            self.add_sample(names[i], preds[i].float(), gts[i].float(), confs[i], ids[i])

    def save_detections(self, filename):
        dict_detections = {
            'samples': self.samples,
            'pred_boxes': self.pred_boxes,
            'gt_boxes': self.gt_boxes,
            'confidences': self.confidences,
            'ids': self.v_ids
        }
        torch.save(dict_detections, str(self.run_path / filename
                                        .format(self.cfg['score_threshold'], self.n_coop)))
        self.has_test_detections = True

    def cal_precision_recall(self, IoU_thr=0.5):
        list_sample = []
        list_confidence = []
        list_tp = []
        N_gt = 0

        for sample in self.samples:
            if len(self.pred_boxes[sample])>0 and len(self.gt_boxes[sample])>0:
                ious = self.iou_fn(self.pred_boxes[sample], self.gt_boxes[sample])
                n, m = ious.shape
                list_sample.extend([sample] * n)
                list_confidence.extend(self.confidences[sample])
                N_gt += len(self.gt_boxes[sample])
                max_iou_pred_to_gts = ious.max(dim=1)
                max_iou_gt_to_preds = ious.max(dim=0)
                tp = max_iou_pred_to_gts[0] > IoU_thr
                is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                                ==torch.tensor([i for i in range(len(tp))], device=tp.device)
                tp[torch.logical_not(is_best_match)] = False
                list_tp.extend(tp)
            elif len(self.pred_boxes[sample])==0:
                N_gt += len(self.gt_boxes[sample])
            elif len(self.gt_boxes[sample])==0:
                tp = torch.zeros(len(self.pred_boxes[sample]), device=self.pred_boxes[sample].device)
                list_tp.extend(tp.bool())
        order_inds = torch.tensor(list_confidence).argsort(descending=True)
        tp_all = torch.tensor(list_tp)[order_inds]
        list_accTP = tp_all.cumsum(dim=0)
        # list_accFP = torch.logical_not(tp_all).cumsum(dim=0)
        list_precision = list_accTP.float() / torch.arange(1, len(list_sample) + 1)
        list_recall = list_accTP.float() / N_gt
        # plt.plot(list_recall.numpy(), list_precision.numpy(), 'k.')
        # plt.savefig(str(model.run_path / 'auc_thr{}_ncoop{}.png'
        #                 .format(model.cfg['score_threshold'], model.n_coop)))
        # plt.close()

        return list_precision, list_recall

    def cal_ap_all_point(self, IoU_thr=0.5):
        '''
        source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L292
        '''

        prec, rec = self.cal_precision_recall(IoU_thr=IoU_thr)
        mrec = []
        mrec.append(0)
        [mrec.append(e.item()) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e.item()) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    def cal_ap_11_point(self, IoU_thr=0.5):
        '''
        source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L315
        '''
        # 11-point interpolated average precision
        prec, rec = self.cal_precision_recall(IoU_thr=IoU_thr)
        mrec = []
        # mrec.append(0)
        [mrec.append(e.item()) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e.item()) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]


class MetricSemSeg(Metric):
    def __init__(self, cfg, run_path, name='test'):
        super(MetricSemSeg, self).__init__(cfg, run_path)
        self.filename = os.path.join(run_path, name)
        self.n_cls = cfg['n_cls']
        # model.result = {
        #     'tp': [],
        #     'tn': [],
        #     'fp': [],
        #     'fn': [],
        #     'N': 0
        # }
        self.result = {
            'area_intersect': torch.zeros(self.n_cls),
            'area_label': torch.zeros(self.n_cls),
            'area_pred': torch.zeros(self.n_cls),
            'area_union': torch.zeros(self.n_cls)
        }

    def add_samples(self, data_dict):
        preds = torch.argmax(data_dict['pred_cls'], dim=1).view(-1, 1)
        tgts = data_dict['tgt_cls'].view(-1, 1)
        # mask = (tgts != 0)
        # preds = preds[mask]
        # tgts = tgts[mask]
        classes = torch.arange(self.n_cls, dtype=preds.dtype, device=preds.device).view(1, -1)
        intersect = preds[preds == tgts]
        area_intersect = (intersect.view(-1, 1) == (classes)).sum(0)
        area_pred = (preds.view(-1, 1) == (classes)).sum(0)
        area_label = (tgts.view(-1, 1) == (classes)).sum(0)
        area_union = area_label + area_label - area_intersect
        self.result['area_intersect'] = self.result['area_intersect'] + area_intersect.cpu()
        self.result['area_label'] = self.result['area_label'] + area_label.cpu()
        self.result['area_pred'] = self.result['area_pred'] + area_pred.cpu()
        self.result['area_union'] = self.result['area_union'] + area_union.cpu()
        # pred_pos = preds.int() == classes
        # pred_neg = torch.logical_not(pred_pos)
        # tgt_pos = tgts.int() == classes
        # tgt_neg = torch.logical_not(tgt_pos)
        # tp = torch.logical_and(pred_pos, tgt_pos).sum(0)
        # tn = torch.logical_and(pred_neg, tgt_neg).sum(0)
        # fp = torch.logical_and(pred_pos, tgt_neg).sum(0)
        # fn = torch.logical_and(pred_neg, tgt_pos).sum(0)
        # acc_ = tp.sum() / len(tgts)
        # model.result['tp'].append(tp)
        # model.result['tn'].append(tn)
        # model.result['fp'].append(fp)
        # model.result['fn'].append(fn)
        # model.result['N'] += len(tgts)

    def cal_ious_and_accs(self):
        area_intersect = self.result['area_intersect'].sum(0)
        area_label = self.result['area_label'].sum(0)
        area_union = self.result['area_union'].sum(0)
        all_acc = area_intersect.sum() / area_label.sum()
        acc = area_intersect / area_label
        iou = area_intersect / area_union

        result = {
            'all_acc': all_acc,
            'acc': acc,
            'iou': iou
        }
        for k, v in result.items():
            print(k, v)
        return result

    def save_detections(self, filename):
        torch.save(self.result, filename)


class MetricStaticIou(Metric):
    def __init__(self, cfg, run_path, logger, name='test'):
        super(MetricStaticIou, self).__init__(cfg, run_path)
        self.filename = os.path.join(run_path, name)
        self.logger = logger
        self.cfg = cfg
        self.thrs = torch.arange(0.1, 1.1, 0.1)
        self.result = {
            'conf': [],
            'unc': [],
            'gt_bev': [],
            'iou_all': [],
        }

    def add_samples(self, out_dict):
        obs_mask = out_dict['road_obs_mask']
        self.result['conf'].append(out_dict['road_confidence'][obs_mask])
        self.result['unc'].append(out_dict['road_uncertainty'][obs_mask])
        self.result['gt_bev'].append(out_dict['road_bev'][obs_mask].bool())
        pos_mask = torch.argmax(out_dict['road_confidence'], dim=-1).bool()
        mi = torch.logical_and(pos_mask, out_dict['road_bev']).sum()
        mu = torch.logical_or(pos_mask, out_dict['road_bev']).sum()
        self.result['iou_all'].append(mi / mu)

    def summary(self):
        conf = torch.cat(self.result['conf'], dim=0)
        unc = torch.cat(self.result['unc'], dim=0)
        ious_all = torch.stack(self.result['iou_all'], dim=0).mean().item()
        gt_bev = torch.cat(self.result['gt_bev'], dim=0)

        order_inds = torch.tensor(conf[:, 1]).argsort(descending=False)
        tp_all = torch.logical_and(torch.argmax(conf).bool(), gt_bev)
        tp_all = torch.tensor(tp_all)[order_inds]
        list_accTP = tp_all.cumsum(dim=0)
        list_precision = list_accTP.float() / torch.arange(1, len(unc) + 1).to(unc.device)
        list_recall = list_accTP.float() / gt_bev.sum()
        plt.plot(list_recall.numpy(), list_precision.numpy(), 'k.')
        plt.savefig(os.path.join(self.filename, 'auc.png'))
        plt.close()

        ious = []
        recalls = []
        fig = plt.figure(figsize=(6, 6))
        for unc_thr in self.thrs:
            valid = torch.logical_and(unc < unc_thr, unc >= unc_thr - 0.1)
            pos = torch.argmax(conf[valid], dim=-1) == 1
            tp = torch.logical_and(pos, gt_bev[valid]).sum()
            fp = torch.logical_and(pos, torch.logical_not(gt_bev[valid])).sum()
            fn = torch.logical_and(torch.logical_not(pos), gt_bev[valid]).sum()
            union = torch.logical_or(pos, gt_bev[valid]).sum()
            tp_ratio = (tp / union).item()
            fp_ratio = (fp / union).item()
            fn_ratio = (fn / union).item()
            iou = tp / union
            recall = tp / gt_bev.bool().sum()
            ious.append(iou * 100)
            recalls.append(recall * 100)

            plt.bar(unc_thr-0.05, [tp_ratio, fp_ratio, fn_ratio], width=0.08,
                    bottom=[0, tp_ratio, tp_ratio+fp_ratio],
                    color=['green', 'orange', 'blue'])
        plt.savefig(os.path.join(self.filename, 'tp_fp_fn.png'))
        plt.close()

        ss = \
            "==================================================================================\n" \
            "  Iou Overall: " f"{ious_all * 100:4.1f} \n"\
            "          thr: " + "  ".join([f"{thr:4.1f} " for thr in self.thrs]) + "\n"\
            "   static iou: " + "  ".join([f"{iou:4.1f} " for iou in ious]) + "\n"\
            "static recall: " + "  ".join([f"{rc:4.1f} " for rc in recalls]) + "\n"\

        print(ss)
        self.logger.write(ss)


class MetricDynamicIou(Metric):
    def __init__(self, cfg, run_path, logger, name='test'):
        super(MetricDynamicIou, self).__init__(cfg, run_path)
        self.filename = os.path.join(run_path, name)
        self.logger = logger
        voxel_size = cfg['voxel_size']
        self.vs = voxel_size[0] if \
            isinstance(voxel_size, list) \
            else voxel_size
        self.r = cfg['r']
        self.grid_size = int(self.r / self.vs)
        self.thrs = torch.arange(0.1, 1.1, 0.1)
        self.result = {
            'iou_global': [],
            'ious_glob_gt_all': [],
            'jiou': [],
            'iou': [],
            'jiou_boxwise': [],
            'iou_boxwise': [],
        }

    def add_samples(self, out_dict):
        self.remove_ego_box(out_dict)
        self.add_global_ious(out_dict)
        self.add_box_ious(out_dict)

    def remove_ego_box(self, out_dict):
        for k in ['gt', 'pred']:
            mask = torch.norm(out_dict[f'{k}_boxes'][:, 1:3], dim=-1) > 1
            out_dict[f'{k}_boxes'] = out_dict[f'{k}_boxes'][mask]
            out_dict[f'{k}_box_samples'] = out_dict[f'{k}_box_samples'][mask]
            out_dict[f'{k}_box_unc'] = out_dict[f'{k}_box_unc'][mask]
            out_dict[f'{k}_box_conf'] = out_dict[f'{k}_box_conf'][mask]

    def add_box_ious(self, out_dict):
        conf = out_dict['box_bev_conf'][..., 1]
        pred_box_conf = self.get_box_unc_from_sam_unc(out_dict['pred_box_conf'][..., 1])
        # gt_box_unc = out_dict['gt_box_unc']
        pred_box_sam = out_dict['pred_box_samples']
        gt_box_sam = out_dict['gt_box_samples']
        pred_boxes = out_dict['pred_boxes']
        gt_boxes = out_dict['gt_boxes']
        bs = conf.shape[0]

        aligned_gt_boxes = []
        aligned_gt_sam = []
        aligned_gt_box_conf = []
        aligned_mask = []
        aligned_ious = []
        for b in range(bs):
            boxes1 = pred_boxes[pred_boxes[:, 0] == b]
            gt_msk = gt_boxes[:, 0] == b
            boxes2 = gt_boxes[gt_msk]
            if len(boxes1) > 0 and len(boxes2) > 0:
                box_ious = boxes_iou_bev(boxes1[:, 1:], boxes2[:, 1:])
                max_ious, aligned_gt_idx = box_ious.max(dim=1)
                aligned_gt_boxes.append(gt_boxes[gt_msk][aligned_gt_idx])
                aligned_gt_sam.append(gt_box_sam[gt_msk][aligned_gt_idx])
                aligned_gt_box_conf.append(out_dict['gt_box_conf'][gt_msk][aligned_gt_idx])

            elif len(boxes1)==0:
                continue
            else:
                max_ious = torch.zeros_like(boxes1[:, 0])
                aligned_gt_boxes.append(torch.zeros_like(boxes1))
                aligned_gt_sam.append(torch.zeros_like(pred_box_sam[pred_boxes[:, 0] == b]))
                aligned_gt_box_conf.append(torch.zeros_like(out_dict['pred_box_conf'][pred_boxes[:,
                                                                                  0] == b]))
            aligned_mask.append(max_ious > 0)
            aligned_ious.append(max_ious)

        aligned_gt_boxes = torch.cat(aligned_gt_boxes, dim=0)
        aligned_gt_sam = torch.cat(aligned_gt_sam, dim=0)
        aligned_gt_box_conf = torch.cat(aligned_gt_box_conf, dim=0)
        aligned_mask = torch.cat(aligned_mask, dim=0)
        aligned_ious = torch.cat(aligned_ious, dim=0)

        jious_boxwise = []
        ious_boxwise = []
        for thr in self.thrs:
            pred_msk = pred_box_conf > (1 - thr)
            pred_sam = pred_box_sam[pred_msk]
            gt_msk = torch.logical_and(pred_msk, aligned_mask)
            gt_sam = aligned_gt_sam[gt_msk]
            cur_pred_box = pred_boxes[pred_msk]
            cur_gt_box = aligned_gt_boxes[gt_msk]

            jiou, iou = self.jiou(conf, cur_pred_box, cur_gt_box)
            jious_boxwise.append(jiou)
            ious_boxwise.append(iou)
        jiou_oa, iou_oa = self.jiou(conf, pred_boxes[pred_box_conf > 0.0], gt_boxes)

        self.result['jiou'].append(jiou_oa)
        self.result['iou'].append(iou_oa)
        self.result['jiou_boxwise'].append(torch.stack(jious_boxwise, dim=0))
        self.result['iou_boxwise'].append(torch.stack(ious_boxwise, dim=0))

        mpred = pred_boxes[:, 0] == 0
        malin = aligned_mask[mpred]
        vis_pred_boxes = pred_boxes[mpred][malin]
        vis_gt_boxes = aligned_gt_boxes[mpred][malin]
        nbox = len(vis_pred_boxes)
        if nbox<4:
            return
        cols = 3
        rows = int(np.ceil(nbox / cols))
        fig = plt.figure(figsize=(7, rows * 1.4))
        axes = fig.subplots(rows, cols)

        vis_cfs = pred_box_conf[mpred][malin]
        sort_idx = torch.argsort(vis_cfs, descending=True)
        vis_cfs = vis_cfs[sort_idx]
        vis_ious = aligned_ious[mpred][malin][sort_idx]
        vis_gt_boxes = vis_gt_boxes[sort_idx]
        vis_pred_boxes = vis_pred_boxes[sort_idx]
        vis_pred_sam = pred_box_sam[mpred][malin][sort_idx]
        vis_gt_sam = aligned_gt_sam[mpred][malin][sort_idx]
        vis_pred_conf = out_dict['pred_box_conf'][mpred][malin][sort_idx]
        vis_gt_conf = aligned_gt_box_conf[mpred][malin][sort_idx]
        from utils.vislib import draw_box_plt
        for i in range(cols * rows):
            ax = axes[i // cols, i % cols]
            if i >= nbox:
                ax.axis('off')
                continue
            iou = vis_ious[i]
            jiou, _ = self.jiou(conf, vis_pred_boxes[i:i + 1],
                                    vis_gt_boxes[i:i + 1])
            cf = vis_cfs[i]
            # ax.set_title(
            #     f'{u:.2f}:[{jiou.item():.2f},{iou.item():.2f}]',
            #     fontsize=10
            # )
            ax.text(0.01, -0.1,
                    f'{cf:.2f}',
                    fontsize=12,
                    linespacing=0.8,
                    backgroundcolor=[1-cf.item(), 1, 1-cf.item()],
                    # bbox=dict(facecolor=[1, u.item(), u.item()]),
                    transform=ax.transAxes)
            ax.text(0.3, -0.1,
                    f':[{jiou.item():.2f},{iou.item():.2f}]',
                    fontsize=12,
                    linespacing=0.8,
                    transform=ax.transAxes)
            ax.axis('equal')
            ax.axis('off')
            sam = vis_pred_sam[i].cpu().numpy()
            cur_conf = vis_pred_conf[i].cpu().numpy()
            ax.scatter(sam[..., 0], sam[..., 1], c=cur_conf[..., 1], cmap='cool',
                       s=10, vmin=0, vmax=1)
            ax = draw_box_plt(
                vis_pred_boxes[i:i + 1, 1:], ax, color='r'
            )
            sam = vis_gt_sam[i].cpu().numpy()
            cur_conf = vis_gt_conf[i].cpu().numpy()
            ax.scatter(sam[..., 0], sam[..., 1], c=cur_conf[..., 1], cmap='cool',
                       s=10, vmin=0, vmax=1)
            ax = draw_box_plt(
                vis_gt_boxes[i:i + 1, 1:], ax, color='g'
            )
        plt.savefig(os.path.join(self.filename, 'img',
                                 '_'.join(out_dict['frame_id'][0]) + '_3.png'),
                                 bbox_inches='tight')
        plt.close()
        # print('d')

    def get_box_unc_from_sam_unc(self, box_sam_unc):
        """
        Summarize one unc value for a box based on its sample unc.
        The final unc. is the min. from the mean unc of the 4 triangles of the box.
        Box unc = min(triangle(box_sam_unc).mean())
        :param box-sam_unc: (B, n, n)
        :return: box_unc: (B)
        """
        tmp = box_sam_unc.cpu().numpy()

        triangle = torch.ones_like(box_sam_unc)
        triur = torch.triu(triangle).bool()
        trill = triur.permute(0, 2, 1)
        triul = torch.flip(triur, dims=(1,))
        trilr = torch.flip(trill, dims=(1,))

        bs = triangle.shape[0]
        unc_ur = box_sam_unc[triur].view(bs, -1).mean(dim=-1)
        unc_ul = box_sam_unc[triul].view(bs, -1).mean(dim=-1)
        unc_lr = box_sam_unc[trilr].view(bs, -1).mean(dim=-1)
        unc_ll = box_sam_unc[trill].view(bs, -1).mean(dim=-1)

        box_unc = torch.min(torch.stack([unc_ur, unc_ul, unc_ll, unc_lr],
                                        dim=-1), dim=-1)[0]

        return box_unc

    def get_indices(self, box_samples, boxes):
        s = box_samples.shape[1]
        xy = torch.floor((box_samples + self.r) / self.vs).long()
        mask = torch.logical_and(xy >= 0, xy < self.grid_size * 2).all(dim=-1)
        xy_indices = xy[mask].T
        bi_pred = torch.tile(boxes[:, 0].view(-1, 1, 1), (1, s, s))
        batch_indices = bi_pred[mask].long()
        indices = torch.cat([batch_indices.view(1, -1), xy_indices], dim=0)
        return indices

    def add_global_ious(self, out_dict):
        conf = out_dict['box_bev_conf']
        unc = out_dict['box_bev_unc']
        pred_boxes = out_dict['pred_boxes']
        gt_boxes = out_dict['gt_boxes']

        ious_valid = []
        # ious_obs = []
        ious_all = []
        for thr in self.thrs:
            unc_mask = unc < thr
            ious_valid.append(self.iou_global(conf, gt_boxes, unc_mask, unc_mask))
            # ious_obs.append(iou)
            ious_all.append(self.iou_global(conf, gt_boxes, unc_mask))
        self.result['iou_global'].append(torch.stack(ious_valid, dim=0))
        # self.result['iou_global_gt_obs'].append(torch.stack(ious_obs, dim=0))
        self.result['ious_glob_gt_all'].append(torch.stack(ious_all, dim=0))

    def iou_global(self, conf, gt_boxes, unc_mask, gt_mask=None):
        gt_mask_out = torch.zeros_like(conf[..., 0])
        if gt_mask is None:
            gt_mask = gt_mask_out + 1

        indices = torch.stack(torch.where(gt_mask), dim=1)
        ixy = indices.float()
        ixy[:, 1:] = (ixy[:, 1:] + 0.5) * self.vs - self.r
        ixyz = F.pad(ixy, (0, 1), 'constant', 0.0)
        boxes = gt_boxes.clone()
        boxes[:, 3] = 0
        boxes_decomposed, box_idx_of_pts = points_in_boxes_gpu(
            ixyz, boxes, batch_size=conf.shape[0]
        )
        inds = indices[box_idx_of_pts >= 0].T
        gt_mask_out[inds[0], inds[1], inds[2]] = 1
        pos_mask = torch.logical_and(torch.argmax(conf, dim=-1).bool(),
                                     unc_mask)
        mi = torch.logical_and(gt_mask_out.bool(), pos_mask)
        mu = torch.logical_or(gt_mask_out.bool(), pos_mask)
        iou = mi.sum() / mu.sum()
        return iou


    def jiou(self, conf, pred_boxes, gt_boxes):
        if len(pred_boxes)==0 or len(gt_boxes)==0:
            return torch.tensor(0, device=conf.device), torch.tensor(0, device=conf.device)
        indices = torch.stack(torch.where(conf > 0), dim=1)
        ixy = indices.float()
        ixy[:, 1:] = (ixy[:, 1:] + 0.5) * self.vs - self.r
        ixyz = F.pad(ixy, (0, 1), 'constant', 0.0)
        # pred
        boxes = pred_boxes.clone()
        boxes[:, 3] = 0
        boxes_decomposed, box_idx_of_pts = points_in_boxes_gpu(
            ixyz, boxes, batch_size=conf.shape[0]
        )
        pred_mask = box_idx_of_pts >= 0

        #gt
        boxes = gt_boxes.clone()
        boxes[:, 3] = 0
        boxes_decomposed, box_idx_of_pts = points_in_boxes_gpu(
            ixyz, boxes, batch_size=conf.shape[0]
        )
        gt_mask = box_idx_of_pts >= 0

        # plt.imshow(pred_mask[0].cpu().numpy())
        # plt.show()
        # plt.close()

        mi = torch.logical_and(pred_mask, gt_mask)
        mu = torch.logical_or(pred_mask, gt_mask)

        indices_int = indices[mi].T
        indices_uni = indices[mu].T
        intersection = conf[indices_int[0], indices_int[1], indices_int[2]]
        union = conf[indices_uni[0], indices_uni[1], indices_uni[2]]

        jiou = intersection.sum() / union.sum()
        iou = mi.sum() / mu.sum()
        return jiou, iou

    def summary(self):
        ious_glob = torch.stack(self.result['iou_global'], dim=0).mean(dim=0) * 100
        ious_glob_gt_all = torch.stack(self.result['ious_glob_gt_all'], dim=0).mean(dim=0) * 100

        jiou_oa = torch.stack(self.result['jiou'], dim=0).mean() * 100
        iou_oa = torch.stack(self.result['iou'], dim=0).mean() * 100

        jious_boxwise = torch.stack(self.result['jiou_boxwise'], dim=0)
        jious_boxwise = (jious_boxwise.sum(dim=0) / jious_boxwise.bool().sum(dim=0)) * 100
        ious_boxwise= torch.stack(self.result['iou_boxwise'], dim=0)
        ious_boxwise = (ious_boxwise.sum(dim=0) / ious_boxwise.bool().sum(dim=0)) * 100

        ss = \
            "=================================Dynamic=====================================\n" \
            "      JIoU Overall: " + f"{jiou_oa.item():4.1f} \n" \
            "       IoU Overall: " + f"{iou_oa.item():4.1f} \n" \
            "               thr: " + " ".join([f"{thr:4.1f} " for thr in self.thrs]) + "\n" \
            "        iou global: " + " ".join([f"{iou:4.1f} " for iou in ious_glob]) + "\n" \
            " iou global gt all: " + " ".join([f"{iou:4.1f} " for iou in ious_glob_gt_all]) + "\n" \
            "      jiou_boxwise: " + " ".join([f"{jiou:4.1f} " for jiou in jious_boxwise]) + "\n" \
            "      iou_boxwise : " + " ".join([f"{iou:4.1f} " for iou in ious_boxwise]) + "\n"

        print(ss)
        self.logger.write(ss)



