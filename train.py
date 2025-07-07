import argparse
import copy
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp
import pprint
import time
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from csc_pa.dataset.builder import get_loader
from csc_pa.models.model_helper import ModelBuilder
from csc_pa.utils.dist_helper import setup_distributed
from csc_pa.utils.loss_helper import (
    get_unsupervised_filter_loss,
    get_criterion,
)
from csc_pa.utils.lr_helper import get_optimizer, get_scheduler
from csc_pa.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="./experiments/BUSI/semi_config_aug_1_2.yaml")
parser.add_argument("--save_dir", type=str, default="./CSC-PA/BUSI/1-fold/1_2/")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--port", default=None, type=int)

import math
import random
import torch.nn as nn


def manual_convolution(image, kernel, stride):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))
    for i in range(0, image_height - kernel_height + 1, stride):
        for j in range(0, image_width - kernel_width + 1, stride):
            window = image[i:i + kernel_height, j:j + kernel_width]
            output[i // stride, j // stride] = np.sum(window * kernel)
    return output

def get_masked_set(pred_u_large, cube_size=57, ratio=0.7, masked_ratio=0.7):
    logits_u_aug, label_u_aug = torch.max(torch.softmax(pred_u_large, dim=1), dim=1)
    b, h, w = label_u_aug.shape
    all_idx = {i for i in range(int(h * w / cube_size / cube_size))}
    entropy_pixel = - (logits_u_aug * torch.log(logits_u_aug + 1e-10))
    entropy_pixel_masked = (entropy_pixel * label_u_aug).detach().cpu().numpy().astype(np.float32)
    label_u_aug_np = label_u_aug.detach().cpu().numpy()
    kernel = np.ones((cube_size, cube_size))
    masked_set = []
    for i in range(b):
        patch_val = manual_convolution(entropy_pixel_masked[i], kernel, cube_size) / (manual_convolution(label_u_aug_np[i], kernel, cube_size) + 1e-10)
        num = math.ceil(np.count_nonzero(patch_val) * ratio)
        idx = np.argpartition(-patch_val.ravel(), num)[:num]
        idx_masked_f = np.argpartition(-patch_val.ravel(), num)[num:]
        idx_masked_b = list(all_idx - set(idx) - set(idx_masked_f))
        num_samples_f = math.ceil(len(idx_masked_f) * masked_ratio)
        num_samples_b = math.ceil(len(idx_masked_b) * masked_ratio)
        samples_f = random.sample(list(idx_masked_f), num_samples_f)
        samples_b = random.sample(list(idx_masked_b), num_samples_b)
        idx_masked = samples_f + samples_b
        masked_set.append(idx_masked)
    del patch_val
    return masked_set

def extract_patches(image, patch_size):
    num_channels, image_height, image_width = image.shape
    num_patches_vertical = image_height // patch_size
    num_patches_horizontal = image_width // patch_size
    patches = np.zeros((num_patches_vertical * num_patches_horizontal, num_channels, patch_size, patch_size))

    patch_index = 0
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            start_i = i * patch_size
            start_j = j * patch_size
            patch = image[:, start_i:start_i + patch_size, start_j:start_j + patch_size]
            patches[patch_index] = patch
            patch_index += 1
    return patches

def reconstruct_image(patches, image_shape):
    num_channels, image_height, image_width = image_shape
    patch_size = patches.shape[2]
    num_patches_vertical = image_height // patch_size
    num_patches_horizontal = image_width // patch_size
    image = np.zeros((num_channels, image_height, image_width))

    patch_index = 0
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            start_i = i * patch_size
            start_j = j * patch_size
            patch = patches[patch_index]
            image[:, start_i:start_i + patch_size, start_j:start_j + patch_size] = patch
            patch_index += 1
    return image

def get_masked(image_u, masked_set, cube_size=57):
    b, _, w, h = image_u.shape
    image_masked = []
    for i in range(len(masked_set)):
        img_u = image_u[i].cpu().numpy()
        img_u_patch = extract_patches(img_u, cube_size)
        for j in range(len(masked_set[i])):
            img_u_patch[masked_set[i][j]] = 0
        image_masked.append(reconstruct_image(img_u_patch, img_u.shape))
    del img_u_patch
    return torch.Tensor(np.array(image_masked)).cuda()


def get_label_u(predict1, predict2, percent):
    batch_size, num_class, h, w = predict1.shape
    pred_w = F.softmax(predict1, dim=1)
    logits_u_aug_w, label_u_w = torch.max(pred_w, dim=1)

    pred_s = F.softmax(predict2, dim=1)
    logits_u_aug_s, label_u_s = torch.max(pred_s, dim=1)

    with torch.no_grad():
        # drop pixels with high entropy
        prob_w = torch.softmax(predict1, dim=1)
        entropy_w = -torch.sum(prob_w * torch.log(prob_w + 1e-10), dim=1)

        thresh = np.percentile(
            entropy_w[label_u_w != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask_w = entropy_w.ge(thresh).bool() * (label_u_w != 255).bool()
        label_u_w[thresh_mask_w] = 255

        # drop pixels with high entropy
        prob_s = torch.softmax(predict2, dim=1)
        entropy_s = -torch.sum(prob_s * torch.log(prob_s + 1e-10), dim=1)

        thresh = np.percentile(
            entropy_s[label_u_s != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask_s = entropy_s.ge(thresh).bool() * (label_u_s != 255).bool()
        label_u_s[thresh_mask_s] = 255
    label_mask = label_u_w.eq(label_u_s)
    label_u_w[label_mask == False] = 255

    return label_u_w

def filter_nonzero_features(features):
    non_zero_mask = torch.any(features != 0, dim=1)
    return features[non_zero_mask]

def filter_nonzero_p(p):
    non_zero_mask = (p != 0)
    return p[non_zero_mask]

def get_edge(predict):
    if len(predict.shape) == 4:
        b, _, w, h = predict.shape
        prob, pred = torch.max(torch.softmax(predict, dim=1), dim=1)
    else:
        pred = predict
    tmp = torch.zeros((b, w+2, h+2))
    tmp[:, 1:w+1, 1:h+1] = pred
    # get 8 neighbors
    u_ij_0 = tmp[:, 0:w, 0:h]
    u_ij_1 = tmp[:, 1:w + 1, 0:h]
    u_ij_2 = tmp[:, 2:w + 2, 0:h]
    u_ij_3 = tmp[:, 0:w, 1:h + 1]
    u_ij_4 = tmp[:, 2:w + 2, 1:h + 1]
    u_ij_5 = tmp[:, 0:w, 2:h + 2]
    u_ij_6 = tmp[:, 1:w + 1, 2:h + 2]
    u_ij_7 = tmp[:, 2:w + 2, 2:h + 2]
    u_ij = torch.cat((u_ij_0.unsqueeze(1),
                        u_ij_1.unsqueeze(1),
                        u_ij_2.unsqueeze(1),
                        u_ij_3.unsqueeze(1),
                        u_ij_4.unsqueeze(1),
                        u_ij_5.unsqueeze(1),
                        u_ij_6.unsqueeze(1),
                        u_ij_7.unsqueeze(1)), dim=1)
    u_ij_sum = torch.sum(u_ij, dim=1)
    edge = torch.where((u_ij_sum == 0) | (u_ij_sum == 8), torch.tensor(0), torch.tensor(1))
    return edge.cuda()


def get_neighbor(feat):
    if len(feat.shape) == 4:
        b, c, w, h = feat.shape
        tmp = torch.zeros((b, c, w+2, h+2))
        tmp[:, :, 1:w+1, 1:h+1] = feat
        # get 8 neighbors
        u_ij_0 = tmp[:, :, 0:w, 0:h]
        u_ij_1 = tmp[:, :, 1:w + 1, 0:h]
        u_ij_2 = tmp[:, :, 2:w + 2, 0:h]
        u_ij_3 = tmp[:, :, 0:w, 1:h + 1]
        u_ij_4 = tmp[:, :, 2:w + 2, 1:h + 1]
        u_ij_5 = tmp[:, :, 0:w, 2:h + 2]
        u_ij_6 = tmp[:, :, 1:w + 1, 2:h + 2]
        u_ij_7 = tmp[:, :, 2:w + 2, 2:h + 2]
        u_ij = torch.cat((u_ij_0.unsqueeze(1),
                          u_ij_1.unsqueeze(1),
                          u_ij_2.unsqueeze(1),
                          u_ij_3.unsqueeze(1),
                          u_ij_4.unsqueeze(1),
                          u_ij_5.unsqueeze(1),
                          u_ij_6.unsqueeze(1),
                          u_ij_7.unsqueeze(1)), dim=1)
    else:
        b, w, h = feat.shape
        tmp = torch.zeros((b, w+2, h+2))
        tmp[:, 1:w+1, 1:h+1] = feat
        # get 8 neighbors
        u_ij_0 = tmp[:, 0:w, 0:h]
        u_ij_1 = tmp[:, 1:w + 1, 0:h]
        u_ij_2 = tmp[:, 2:w + 2, 0:h]
        u_ij_3 = tmp[:, 0:w, 1:h + 1]
        u_ij_4 = tmp[:, 2:w + 2, 1:h + 1]
        u_ij_5 = tmp[:, 0:w, 2:h + 2]
        u_ij_6 = tmp[:, 1:w + 1, 2:h + 2]
        u_ij_7 = tmp[:, 2:w + 2, 2:h + 2]
        u_ij = torch.cat((u_ij_0.unsqueeze(1),
                          u_ij_1.unsqueeze(1),
                          u_ij_2.unsqueeze(1),
                          u_ij_3.unsqueeze(1),
                          u_ij_4.unsqueeze(1),
                          u_ij_5.unsqueeze(1),
                          u_ij_6.unsqueeze(1),
                          u_ij_7.unsqueeze(1)), dim=1)
    return u_ij.cuda()


def calculate_pixel_similarity_l(feat, surrounding_feats):
    batch_size, channels, height, width = feat.shape
    num_surrounding_feats = surrounding_feats.shape[1]

    feat_flat = feat.view(batch_size, channels, -1).permute(0, 2, 1)
    surrounding_feats_flat = surrounding_feats.view(batch_size, num_surrounding_feats, channels, -1).permute(0, 1, 3, 2)
    similarity = F.cosine_similarity(feat_flat.unsqueeze(1).expand(-1, 8, -1, -1), surrounding_feats_flat, dim=3).view(batch_size, 8, height, width)
    similarity = similarity / (torch.sum(similarity, dim=1, keepdim=True) + 1e-10)

    return similarity


def calculate_pixel_similarity_u(feat, surrounding_feats, pred_nei):
    batch_size, channels, height, width = feat.shape
    num_surrounding_feats = surrounding_feats.shape[1]
    feat_flat = feat.view(batch_size, channels, -1).permute(0, 2, 1)
    surrounding_feats_flat = surrounding_feats.view(batch_size, num_surrounding_feats, channels, -1).permute(0, 1, 3, 2)
    similarity = F.cosine_similarity(feat_flat.unsqueeze(1).expand(-1, 8, -1, -1), surrounding_feats_flat, dim=3).view(batch_size, 8, height, width)
    similarity = similarity / (torch.sum(similarity * torch.where(pred_nei==0, 0, 1), dim=1, keepdim=True) + 1e-10)
    return similarity

def calculate_pixel_similarity(feat, surrounding_feats, pred_nei):
    batch_size, channels, height, width = feat.shape
    num_surrounding_feats = surrounding_feats.shape[1]

    feat_flat = feat.view(batch_size, channels, -1).permute(0, 2, 1)
    surrounding_feats_flat = surrounding_feats.view(batch_size, num_surrounding_feats, channels, -1).permute(0, 1, 3, 2)
    similarity = F.cosine_similarity(feat_flat.unsqueeze(1).expand(-1, 8, -1, -1), surrounding_feats_flat, dim=3).view(batch_size, 8, height, width)
    similarity = similarity / (torch.sum(similarity * torch.where(pred_nei == 0, 0, 1), dim=1, keepdim=True) + 1e-10)

    return similarity


def get_aff(predict, target, feat1, feat2, drop_percent):
    if predict.size() != target.size():
        target = F.interpolate(target.unsqueeze(1).float(), size=(129, 129), mode="nearest").squeeze(1).to(torch.int)
        b, h, w = target.shape
        prob, pred = torch.max(torch.softmax(predict, dim=1), dim=1)
        pred_nei = get_neighbor(pred)
        feat1_nei = get_neighbor(feat1)
        feat1_aff = calculate_pixel_similarity_l(feat1, feat1_nei)
        aff = torch.sum(feat1_aff * torch.where(pred_nei == 0, -1, 1), dim=1)
        tg_nei = get_neighbor(target.clone().detach())
        aff_tg = torch.mean(torch.where(tg_nei == 0, -1.0, 1.0), dim=1)
        aff_tg = ((aff_tg + 1) / 2).clone().detach()
        aff = ((aff + 1) / 2).clone().detach()
        tg_probs = F.softmax(aff_tg, dim=1)
        pred_probs = F.softmax(aff, dim=1)
        return pred_probs, tg_probs
    else:
        prob_p, pred_p = torch.max(torch.softmax(predict, dim=1), dim=1)
        prob_t, pred_t = torch.max(torch.softmax(target, dim=1), dim=1)
        b, _, h, w = target.shape

        prob = torch.softmax(target, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        thresh = np.percentile(
            entropy.detach().cpu().numpy().flatten(), drop_percent
        )
        aff_mask = entropy.ge(thresh).bool()
        pred_p = torch.where(pred_p==0, -1, 1)
        pred_t = torch.where(pred_t==0, -1, 1)
        pred_p[aff_mask] = 0
        pred_t[aff_mask] = 0
        pred_p_nei = get_neighbor(pred_p)
        pred_t_nei = get_neighbor(pred_t)
        feat_p_nei = get_neighbor(feat1)
        feat_t_nei = get_neighbor(feat2)
        feat_p_aff = calculate_pixel_similarity_u(feat1, feat_p_nei, pred_p_nei)
        feat_t_aff = calculate_pixel_similarity_u(feat2, feat_t_nei, pred_t_nei)
        aff_p = torch.sum(feat_p_aff * pred_p_nei, dim=1)
        aff_t = torch.sum(feat_t_aff * pred_t_nei, dim=1)

        aff_p = ((aff_p + 1) / 2).clone().detach()
        aff_t = ((aff_t + 1) / 2).clone().detach()
        p_probs = F.softmax(aff_p, dim=1)
        t_probs = F.softmax(aff_t, dim=1)

        return p_probs, t_probs

def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)
    aff_loss_fn = nn.KLDivLoss(reduction='batchmean')

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    best_dice = 0
    last_epoch = 0

    # auto_resume > pretrain
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    # build class-wise memory bank
    memobank = torch.tensor([])
    feat = torch.tensor([])

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        memobank, feat = train(
            model,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            aff_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            memobank,
            feat,
        )

        # Validation
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")

            dice, iou = validate(model, val_loader, epoch, logger)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_dice": best_dice,
                }
                if dice > best_dice:
                    best_dice = dice
                    torch.save(
                        state, osp.join(args.save_dir, "ckpt_best.pth")
                    )

                torch.save(state, osp.join(args.save_dir, "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_dice * 100
                    )
                )
                tb_logger.add_scalar("dice val", dice, epoch)


def train(
    model,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    aff_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    memobank,
    feat,
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    aff_losses = AverageMeter(20)
    data_times = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)

    aff_weight = 1.0

    batch_end = time.time()

    for step in range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        # obtain labeled and unlabeled data
        _, image_l, label_l = next(loader_l_iter)
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()
        _, image_u_weak, image_u_aug, _ = next(loader_u_iter)
        image_u_weak, image_u_aug = image_u_weak.cuda(), image_u_aug.cuda()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            # forward
            outs = model(image_l, torch.tensor([]), [torch.tensor([])])
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)
            aff_w, aff_s = get_aff(outs["pred"], label_l, outs["feat"], None, 0)
            aff_loss = aff_loss_fn(aff_w.log(), aff_s)

            # initialize MB
            if epoch == cfg["trainer"].get("sup_only_epoch", 1) - 1:
                feat = outs["feat"]
                # predict
                rep_por, rep_class = torch.max(F.softmax(outs["pred"], dim=1), dim=1)
                edge = get_edge(outs["pred"])
                # filter wrong pixels
                label_l_resize = F.interpolate(label_l.unsqueeze(1).float(), size=(129, 129), mode="nearest").squeeze(1).to(torch.int)
                rep_mask = (rep_class != label_l_resize)
                rep_por[rep_mask] = 0
                por = (rep_por * edge).unsqueeze(1).permute(0, 2, 3, 1).reshape(-1, 1)
                nonzero_ind = torch.nonzero(por.squeeze())
                a = 16 * 16
                random_ind = torch.randperm(nonzero_ind.size(0))[:a]
                ind = nonzero_ind[random_ind].squeeze()
                b, rep_dim, _, _ = feat.shape
                feat = feat.permute(0, 2, 3, 1).reshape(-1, rep_dim)
                memobank = feat[ind]

            unsup_loss = 0 * rep.sum()

            loss = sup_loss + unsup_loss + aff_weight * aff_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                feat = torch.tensor([])
            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_weak))
            outs_all = model(image_all, memobank, feat)
            pred_all, rep_all = outs_all["pred"], outs_all["rep"]
            pred_l = pred_all[:num_labeled]
            pred_u = pred_all[num_labeled:]

            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs_all["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())
            aff_w, aff_s = get_aff(pred_l, label_l, outs_all["feat"][:num_labeled], None, 0)
            aff_loss = aff_loss_fn(aff_w.log(), aff_s)
            loss = sup_loss + aff_weight * aff_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            memobank = outs_all["memory"].detach()

            k = 0.7 * ((200-epoch) / 200)
            masked_set = get_masked_set(pred_u_large, cube_size=57, ratio=k, masked_ratio=0.7)  # Confidence of unlabeled samples
            image_u_masked = get_masked(image_u_weak, masked_set, cube_size=57)
            image_masked_all = torch.cat((image_l, image_u_masked))

            outs_masked_all = model(image_masked_all, memobank, torch.tensor([]))
            pred_masked_all, rep_masked_all = outs_masked_all["pred"], outs_masked_all["rep"]
            pred_masked_l = pred_masked_all[:num_labeled]
            pred_masked_u = pred_masked_all[num_labeled:]

            pred_masked_l_large = F.interpolate(
                pred_masked_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_masked_u_large = F.interpolate(
                pred_masked_u, size=(h, w), mode="bilinear", align_corners=True
            )
            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs_all["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_masked_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_masked_loss = sup_loss_fn(pred_masked_l_large, label_l.clone())
            aff_w, aff_s = get_aff(pred_masked_l, label_l, outs_masked_all["feat"][:num_labeled], None, 0)
            aff_l_loss = aff_loss_fn(aff_w.log(), aff_s)

            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
            drop_percent = 100 - percent_unreliable
            unsup_loss2 = get_unsupervised_filter_loss(pred_masked_u_large, pred_u_large, drop_percent)
            unsup_loss = unsup_loss2
            aff_w, aff_s = get_aff(pred_masked_u, pred_u, outs_masked_all["feat"][num_labeled:], outs_all["feat"][num_labeled:], drop_percent)
            aff_u_loss = aff_loss_fn(aff_w.log(), aff_s)
            aff_loss = (aff_l_loss + aff_u_loss) / 2
            loss = sup_masked_loss + unsup_loss + aff_weight * aff_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update MB
            feat_all = outs_all["feat"].detach()
            # predict
            rep_por, rep_class = torch.max(F.softmax(pred_all, dim=1), dim=1)
            edge = get_edge(pred_all)
            # filter wrong pixels & unreliable pixels
            # labeled
            label_l_resize = F.interpolate(label_l.clone().unsqueeze(1).float(), size=(129, 129), mode="nearest").squeeze(1).to(torch.int)
            rep_mask_l = (rep_class[:num_labeled] != label_l_resize)
            # unlabeled
            rep_prob = torch.softmax(pred_u.clone(), dim=1)
            rep_entropy = -torch.sum(rep_prob * torch.log(rep_prob + 1e-10), dim=1)
            rep_thresh = np.percentile(
                rep_entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            rep_mask_u = rep_entropy.ge(rep_thresh).bool()
            rep_mask = torch.cat((rep_mask_l, rep_mask_u), dim=0)
            rep_por[rep_mask] = 0

            por = (rep_por * edge).unsqueeze(1).permute(0, 2, 3, 1).reshape(-1, 1)
            nonzero_ind = torch.nonzero(por.squeeze())
            a = 16 * 16
            random_ind = torch.randperm(nonzero_ind.size(0))[:a]
            ind = nonzero_ind[random_ind].squeeze()
            b, rep_dim, _, _ = feat_all.shape
            feat_all = feat_all.permute(0, 2, 3, 1).reshape(-1, rep_dim)  # (33282, 512)
            feat = feat_all[ind]

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_aff_loss = aff_loss.clone().detach()
        dist.all_reduce(reduced_aff_loss)
        aff_losses.update(reduced_aff_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Aff {aff_loss.val:.3f} ({aff_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    aff_loss=aff_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Aff Loss", aff_losses.val, i_iter)
    return memobank.detach(), feat.detach()


def dice_coefficient(pred, gt, smooth=1e-5):
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def sespiou_coefficient(pred, gt, smooth=1e-5):
    """ computational formula:
        iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    return IOU.sum() / N

def validate(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    dice_total = 0
    iou_total = 0
    cnt = 0
    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.cuda()

        images = F.interpolate(
            images, (513, 513), mode="bilinear", align_corners=False
        )
        labels = F.interpolate(labels.unsqueeze(1).float(), size=(513, 513), mode="nearest").squeeze(1).to(torch.int64)

        with torch.no_grad():
            outs = model(images, torch.tensor([]), torch.tensor([]))

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1]

        # start to calculate dice, iou, hd
        dice_total += dice_coefficient(output, labels)
        iou_total += sespiou_coefficient(output, labels)
        cnt += labels.shape[0]

        reduced_dice = dice_total
        reduced_iou = iou_total
        reduced_cnt = torch.tensor(cnt).cuda()

        dist.all_reduce(reduced_dice)
        dist.all_reduce(reduced_iou)
        dist.all_reduce(reduced_cnt)

    dice = dice_total / cnt
    iou = iou_total / cnt

    if rank == 0:
        logger.info(" * epoch {} dice {:.2f} iou {:.2f}".format(epoch, dice * 100, iou * 100))

    return dice, iou


if __name__ == "__main__":
    main()