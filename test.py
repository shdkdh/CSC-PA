import logging
import os
import time
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from PIL import Image

from csc_pa.models.model_helper import ModelBuilder
from hausdorff import hausdorff_distance
from csc_pa.utils.utils import (
    AverageMeter,
    convert_state_dict,
)


# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument("--base_size", type=int, default=(513, 513), help="based size for scaling")
    parser.add_argument("--config", type=str, default="./experiments/BUSI/test_semi_config_1_2.yaml")
    parser.add_argument("--model_path", type=str, default="xxx/BUSI/CSC-PA/1-fold/1_2/ckpt_best.pth", help="evaluation model path")
    parser.add_argument("--save_folder", type=str, default="xxx/CSC-PA/BUSI/1_2/1-fold", help="results save folder")
    return parser

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    global args, logger, cfg, colormap
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]

    assert num_classes > 1

    cfg_dset = cfg["dataset"]
    data_root, fold_path, ratio_path, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["fold_path"], cfg_dset["val"]["ratio_path"], cfg_dset["val"]["data_list"]

    with open(os.path.join(fold_path, 'test.txt'), 'r') as f:
        data_list = f.readlines()
    f.close()
    data_list = [item.split('\t')[0].split('/')[1] for item in data_list]

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])
    checkpoint = torch.load(args.model_path)
    key = "teacher_state" if "teacher_state" in checkpoint.keys() else "model_state"
    logger.info(f"=> load checkpoint[{key}]")

    saved_state_dict = convert_state_dict(checkpoint[key])
    model.load_state_dict(saved_state_dict, strict=False)
    model.cuda()
    logger.info("Load Model Done!")
    valiadte_BUS(args, model, data_root, data_list, mean, std)


@torch.no_grad()
def net_process(model, image):
    b, c, h, w = image.shape
    input = image.cuda()
    output = model(input, torch.tensor([]), torch.tensor([]))["pred"]
    output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    return output

def scale_whole_process(model, image, h, w):
    with torch.no_grad():
        prediction = net_process(model, image)
    prediction = F.interpolate(
        prediction, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0]


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

def valiadte_BUS(args, model, data_root, data_list, mean, std):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    dice_total = 0
    iou_total = 0
    hd_total = 0
    end = time.time()
    with torch.no_grad():
        for i, image_name in enumerate(data_list):
            data_time.update(time.time() - end)
            input_pth = os.path.join(data_root, 'img', image_name)
            label_path = os.path.join(data_root, 'label', image_name[:-4] + '_mask.png') # BUSI
            # label_path = os.path.join(data_root, 'label', image_name) # UDIAT
            image = Image.open(input_pth).convert("RGB")
            image = np.asarray(image).astype(np.float32)
            label = Image.open(label_path).convert("L")
            label = np.asarray(label).astype(np.uint8)
            label[label >= 1] = 1

            image = (image - mean) / std
            image = torch.Tensor(image).permute(2, 0, 1)
            image = image.contiguous().unsqueeze(dim=0)
            h, w = image.size()[-2:]
            image = F.interpolate(image, size=args.base_size, mode='bilinear', align_corners=False)
            prediction = scale_whole_process(model, image, h, w)
            prediction = torch.max(prediction, dim=0)[1].cpu().numpy()
              ##############attention###############
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % 10 == 0:
                logger.info(
                    "Test: [{}/{}] "
                    "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                        i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                    )
                )
            p = np.expand_dims(prediction, axis=0)
            dice_total += dice_coefficient(np.expand_dims(prediction, axis=0), np.expand_dims(label, axis=0)).item()
            iou_total += sespiou_coefficient(np.expand_dims(prediction, axis=0), np.expand_dims(label, axis=0)).item()
            hd_total += hausdorff_distance(prediction, label, distance="euclidean")

            # image_name = data_list[i]
            # save_path = args.save_folder
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # plt.imsave(os.path.join(save_path, image_name), prediction, cmap = 'gray')
    print('dice: ' + str(dice_total / len(data_list)))
    print('iou: ' + str(iou_total / len(data_list)))
    print('hd: ' + str(hd_total / len(data_list)))
    logger.info("<<<<<<<<<<<<<<<<< End  Evaluation <<<<<<<<<<<<<<<<<")


if __name__ == "__main__":
    main()
