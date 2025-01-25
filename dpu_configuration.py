import vart 
import xir
from vaitrace_py import vai_tracepoint
import numpy as np
import torch
from tqdm import tqdm
import time
import argparse
import threading
from model_inference import postprocessing
from utils.dataloaders import create_dataloader
from utils.general import non_max_suppression
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


# Global variables
all_dpu_runners = []
all_dataloaders = []
shared_metrics = []

# Create a semaphore for shared resources
semaphore_metrics = threading.Semaphore(1)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = {
        0: 'dirt',
        1: 'damage'
    }

# Preprocess images for Input CPU subgraph
def preprocess_images(images, fixscale):

    batch, channels, width, height = images.shape
    images = torch.Tensor.numpy(images)
    #images = images.astype(np.float32)
    #images = images/255.0
    images = (images/255.0)*fixscale
    images = images.astype(np.int8)
    images = np.transpose(images, (0, 2, 3, 1)) # Rearrange the columns of input in the appropriate form. From [1, 3, 640, 640] --> [1, 640, 640, 3]

    return images, batch, channels, width, height

# Locate the DPU subgraphs from the yolov5n_cd_pt.xmodel and return a list of them
def get_child_subgraph_dpu(graph):
    
    assert graph is not None #'graph' should not be None.
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None) # Failed to get root subgraph of input Graph object.
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    subgraphs = []
    for subgraph in child_subgraphs:
        if subgraph.has_attr("device") and subgraph.get_attr("device").upper() == "DPU":
            subgraphs.append(subgraph)
    
    return subgraphs

# Delete all runners after finishing computations
def delete_runners():
    for runner in all_dpu_runners:
        del runner
        
# Run graph runner
def dpu_execute_async(dpu_runner, input_tensors, output_tensors):

    job_id = dpu_runner.execute_async(input_tensors, output_tensors)

    return dpu_runner.wait(job_id)

def load_data(data_dir, threads):
    
    if threads==1:
        dataloader = create_dataloader(path=data_dir, batch_size=1, imgsz=640, stride=32, shuffle=True)[0]
        all_dataloaders.append(dataloader)
    elif threads==2:
        dataloader1 = create_dataloader(path=data_dir+'/data_thread1/images/val', batch_size=1, imgsz=640, stride=32, shuffle=True)[0]
        dataloader2 = create_dataloader(path=data_dir+'/data_thread2/images/val', batch_size=1, imgsz=640, stride=32, shuffle=True)[0]
        all_dataloaders.append(dataloader1)
        all_dataloaders.append(dataloader2)

# Run thread
@vai_tracepoint
def runThread_funct(id, dpu_runner, dataloader):

    save_dir = f'inference_for_thread{id}'

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

    # Get input/output tensors for the runner.
    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()

    # Get input/output tensors dimensions.
    input_tensor_ndim = tuple(input_tensors[0].dims)
    output_tensor1_ndim = tuple(output_tensors[0].dims)
    output_tensor2_ndim = tuple(output_tensors[1].dims)
    output_tensor3_ndim = tuple(output_tensors[2].dims)

    # input/output fixpos
    input_fixpos = input_tensors[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    output1_fixpos = output_tensors[0].get_attr("fix_point")
    output1_scale = 1.0 / (2**output1_fixpos)
    output2_fixpos = output_tensors[1].get_attr("fix_point")
    output2_scale = 1.0 / (2**output2_fixpos)
    output3_fixpos = output_tensors[2].get_attr("fix_point")
    output3_scale = 1.0 / (2**output3_fixpos)

    # Initialize input/output tensors.
    input_data = [np.empty(input_tensor_ndim, dtype=np.int8, order='C')]
    output_data = [
        np.empty(output_tensor1_ndim, dtype=np.int8, order='C'),
        np.empty(output_tensor2_ndim, dtype=np.int8, order='C'),
        np.empty(output_tensor3_ndim, dtype=np.int8, order='C')
    ]

    pbar = tqdm(dataloader, desc=s, total=len(dataloader)) # progress bar
    for batch_i, (image, targets, paths, shapes) in enumerate(pbar):

        # Image preprocessing
        with dt[0]:
            # Assign the input image to the input buffer
            image_, _, _, width, height = preprocess_images(image, input_scale)
            inputImage = input_data[0]
            inputImage[0] = image_

        # Inference
        with dt[1]: 
            dpu_execute_async(dpu_runner, input_data, output_data)
            # Rearrange the columns of outputs in the appropriate form. 
            #  [1, 80, 80, 21] --> [1, 21, 80, 80]
            #  [1, 40, 40, 21] --> [1, 21, 40, 40]
            #  [1, 20, 20, 21] --> [1, 21, 20, 20]
            outputs = [torch.Tensor(np.transpose(output1_scale*output_data[0], (0, 3, 1, 2))), torch.Tensor(np.transpose(output2_scale*output_data[1], (0, 3, 1, 2))), torch.Tensor(np.transpose(output3_scale*output_data[2], (0, 3, 1, 2)))]
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
        # mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=2)  # number of targets per class

    t = np.array([x.t / seen * 1e3 for x in dt])  # speeds per image

    semaphore_metrics.acquire()
    try:
        shared_metrics.append(np.array([seen, nt, p, r, ap50, ap, ap_class, t]))
    finally:
        semaphore_metrics.release()

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

# Start processing
def start_computations(xmodel, val_data_dir, threads):

    # get a list of all DPU subgraphs
    g = xir.Graph.deserialize(xmodel)
    subgraphs = get_child_subgraph_dpu(g)
    print('Found',len(subgraphs),'DPU subgraphs') # 1 big subgraph

    # Create a runner for each DPU subgraph for each thread.
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # Create dataloaders
    load_data(val_data_dir, threads)

    # Start threads
    print('Starting',threads,'threads...')
    threadALL = []
    for thread_id in range(threads):
        thread = threading.Thread(target=runThread_funct, args=(thread_id, all_dpu_runners[thread_id], all_dataloaders[thread_id]))
        threadALL.append(thread)

    time_start = time.time()

    for thread in threadALL:
      thread.start()
    for thread in threadALL:
      thread.join()

    time_end = time.time()
    total_execution_time = time_end - time_start

    delete_runners()

    # Computation of combined metrics.
    seen, num_of_instances, p, r, ap50, ap = 0, 0, 0, 0, 0, 0
    for element in shared_metrics:
        seen += element[0]
        num_of_instances += element[1] 
        p += element[2] 
        r += element[3] 
        ap50 += element[4]
        ap += element[5] 
    
    p /= threads
    r /= threads
    ap50 /= threads
    ap /= threads

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, num_of_instances.sum(), p.mean(), r.mean(), ap50.mean(), ap.mean()))
    if not (num_of_instances.any()):
        LOGGER.warning(f"WARNING ⚠️ no labels found in validation set, can not compute metrics without labels")

    # Print results per class
    for i, c in enumerate(shared_metrics[0][6]): # Choose one of the two ap_class variables.
        LOGGER.info(pf % (names[c], seen, num_of_instances[c], p[i], r[i], ap50[i], ap[i]))

    times = 0
    for i in range(len(shared_metrics)):
        times += shared_metrics[i][7]
    t = tuple(times/threads)
    shape = (1, 640, 640, 3)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    LOGGER.info(f"Total execution time: %.4f seconds" % total_execution_time)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--val_data_dir',
        default="data/data/images/val",
        help='Validation data set directory')
    parser.add_argument(
        '--model_dir',
        default="yolov5n_cd_pt/yolov5n_cd_pt.xmodel",
        help='Compiled model file path. This is yolov5n_cd_pt.xmodel')
    parser.add_argument(
        '--threads',
        default=1,
        type=int,
        help='Number of threads')

    args, _ = parser.parse_known_args()

    model_name = 'YOLOv5 nano on Wind Turbines data set'
    print(f"-------- Start {model_name}, evaluation test on FPGA ---------")
    
    start_computations(args.model_dir, args.val_data_dir, args.threads)

    print(f"-------- End of {model_name}, evaluation test on FPGA --------")


    
