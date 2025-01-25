import pytorch_nndct
import argparse
import torch
from utils.general import non_max_suppression
from utils.dataloaders import create_dataloader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from utils.callbacks import Callbacks
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images
from utils.general import (
    LOGGER,
    Profile,
    scale_boxes,
    xywh2xyxy
)
from val import process_batch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--val_data_dir',
    default="/path/to/validation/data/set",
    help='Validation data set directory')
parser.add_argument(
    '--model_dir',
    default="/path/to/quantized_model/",
    help='Quantized model file path. This is usually DetectMultiBackend_int.pt'
)
parser.add_argument(
    '--batch_size',
    default=16,
    type=int,
    help='input data batch size to evaluate model')

args, _ = parser.parse_known_args()

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Forward method of Detect Class placed here. The view and permute of this method aren't available in Vitis AI.
# [(batch_size, 21, 80, 80), (batch_size, 21, 40, 40), (batch_size, 21, 20, 20)] ---> (batch_size, 25200, 7)
def postprocessing(x, nc=2, nl=3, na=3, no=7):
    grid = [torch.empty(0) for _ in range(nl)]
    z = []
    anchor_grid = [torch.empty(0) for _ in range(nl)]
    stride = torch.tensor([ 8., 16., 32.], device=device)

    # Updated anchors from training on the custom dataset
    anchors = torch.tensor([[1.25000,  1.62500, 2.00000,  3.75000,4.12500,  2.87500],
        [1.87500,  3.81250, 3.87500,  2.81250, 3.68750,  7.43750],
        [ 3.62500,  2.81250, 4.87500,  6.18750, 11.65625, 10.18750]], device=device)
    anchors = torch.tensor(anchors).float().view(3,-1,2)

    x_ = []
    for i in range(nl):
        bs, _, ny, nx = x[i].shape  # x(bs,21,20,20) to x(bs,3,20,20,7)
        x_.append(x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())

        if grid[i].shape[2:4] != x_[i].shape[2:4]:
            grid[i], anchor_grid[i] = make_grid(nx,ny,i,anchors,stride)
            
        xy, wh, conf = x_[i].sigmoid().split((2, 2, nc + 1), 4)
        xy = (xy * 2 + grid[i]) * stride[i]  # xy
        wh = (wh * 2) ** 2 * anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, na * nx * ny, no))

    return (torch.cat(z, 1), x_)

# Same function as in the original version of yolo.py
def make_grid(nx=20, ny=20, i=0,anchors = None, stride = None):
    d = anchors[i].device
    t = anchors[i].dtype

    shape = 1, 3, ny, nx, 2  # grid shape
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
    yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = (anchors[i] * stride[i]).view((1, 3, 1, 1, 2)).expand(shape)
    return grid, anchor_grid

# Evaluation function for inference
def evaluate_quantized_model(model_dir='', val_data_dir='', batch_size=16):

    # Load the quantized model with torch.jit.load
    model = torch.jit.load(model_dir, map_location=device)

    # Initialize dictionary with model's classes
    names = {
        0: 'dirt',
        1: 'damage'
    }

    # Folder for saving resutls from inference
    save_dir = 'inference'

    # Initialize dataloader with the validation dataset
    val_dir = val_data_dir
    dataloader = create_dataloader(path=val_dir, batch_size=batch_size, imgsz=640, stride=32, shuffle=True)[0]

    model.eval()
    model = model.to(device)

    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    callbacks=Callbacks()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    confusion_matrix = ConfusionMatrix(nc=2)
    seen = 0
    stats, ap, ap_class = [], [], []
    plots = True
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, total=len(dataloader))  # progress bar
    for batch_i, (image, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")

        # Image preprocessing
        with dt[0]:
            image = image.float()               # uint8 to fp16/32
            image /= 255                        # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = image.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            outputs = model(image)
            preds = postprocessing(outputs)
        
        # Postprocessing
        with dt[2]:
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            lb = []  # for autolabelling
            preds = non_max_suppression(preds, conf_thres=0.3, iou_thres=0.5, labels=lb)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue
            
            # Predictions
            predn = pred.clone()
            scale_boxes(image[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])    # target boxes
                scale_boxes(image[si].shape[1:], tbox, shape, shapes[si][1])    # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)      # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
            callbacks.run("on_val_image_end", pred, predn, path, names, image[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(image, targets, paths, f"{save_dir}/val_batch{batch_i}_labels.jpg")  # labels
            plot_images(image, output_to_target(preds), paths, f"{save_dir}/val_batch{batch_i}_pred.jpg")  # pred

        callbacks.run("on_val_batch_end", batch_i, image, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=2)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in validation set, can not compute metrics without labels")

    # Print results per class
    for i, c in enumerate(ap_class):
        LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    shape = (batch_size, 3, 640, 640)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)


if __name__ == '__main__':

    model_name = 'YOLOv5 nano on Wind Turbines data set'
    print(f"-------- Start {model_name}, evaluation test --------")

    evaluate_quantized_model(model_dir=args.model_dir,
                             val_data_dir=args.val_data_dir,
                             batch_size=args.batch_size)

    print(f"-------- End of {model_name}, evaluation test --------")
