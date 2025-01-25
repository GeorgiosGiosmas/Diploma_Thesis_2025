import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from models.common import DetectMultiBackend
from utils.loss import ComputeLoss
from utils.dataloaders import create_dataloader, LoadImagesAndLabels
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from val import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default="/path/to/custom_data_set/",
    help='Data set directory. This data set will be used for calibration.')
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path.'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
    help='input data batch size')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, inspect float model, calib: quantize, test: export quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()

# Forward function
def forward_loop(model, dataloader):
  model.eval()
  model = model.to(device)
  
  pbar = tqdm(dataloader, total=len(dataloader)) # progress bar
  for batch_i, (image, _, _, _) in enumerate(pbar):
    image = image.float()  # uint8 to fp16/32
    image /= 255           # from 0-255 to 0-1
    outputs = model(image)

    if (args.quant_mode == 'test' and  args.deploy):
      break

def quantization(file_path=''): 

  data_dir = args.data_dir
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  inspect = args.inspect
  config_file = args.config_file
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1

  model = DetectMultiBackend(weights=file_path, device=device, fp16=False)
  
  input = torch.randn([batch_size, 3, 640, 640])
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      if not target:
          raise RuntimeError("A target should be specified for inspector.")
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      inspector = Inspector(target)  # by name
      # start to inspect
      inspector.inspect(quant_model, (input,), device=device, image_format=None)
      sys.exit()
      
  else:
    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################
  
  # Dataloader for calibration
  data_loader = create_dataloader(path=data_dir, batch_size=batch_size, imgsz=640, stride=32, shuffle=True)[0]
  
  # fast finetune model or load finetuned parameter before test
  if finetune == True:
      if quant_mode == 'calib':
        data_loader_fast_finetune = create_dataloader(path='data/data_fast_finetune', batch_size=batch_size, imgsz=640, stride=32, shuffle=True)[0]
        quantizer.fast_finetune(forward_loop, (quant_model, data_loader_fast_finetune))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
  if quant_mode == 'calib':
    # This function call is to do forward loop for model to be quantized.
    # Quantization calibration will be done after it.
    forward_loop(quant_model, data_loader)
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()

  # handle quantization result
  if quant_mode == 'test' and  deploy:
    forward_loop(quant_model, data_loader)
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel()


if __name__ == '__main__':

  model_name = 'best'
  file_path = os.path.join(args.model_dir, model_name + '.pt')

  print(f"-------- Start {model_name} test ")

  # Quantize the Yolov5 model(custom dataset)
  quantization(file_path=file_path)

  print(f"-------- End of {model_name} test ")
