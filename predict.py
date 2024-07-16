import os
import cv2
import json
import argparse
import numpy as np
# import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
# from xys_dataset import Badminton_Dataset
from xys_utils import *
# from torch.nn import DataParallel
# python3 predict.py --video_file pred_result/lindan2.mp4 --threshold 5

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str,default="/ssd2/cz/TrackNetV3/xys_for_test/xyscourt2.mp4")
parser.add_argument('--model_file', type=str, default='/ssd2/cz/TrackNetV3/xys2_new_exp/model_best.pt')
parser.add_argument('--num_frame', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--threshold', type=float, default=7)
parser.add_argument('--start_width', type=int, default=0)
parser.add_argument('--end_width', type=int, default=1720)
parser.add_argument('--save_dir', type=str, default='/ssd2/cz/TrackNetV3/xys_for_test')
args = parser.parse_args()

video_file = args.video_file
model_file = args.model_file
num_frame = args.num_frame
batch_size = args.batch_size
save_dir = args.save_dir
threshold = args.threshold
start_width = args.start_width
end_width = args.end_width

video_name = video_file.split('/')[-1][:-4]
video_format = video_file.split('/')[-1][-3:]
out_video_file = f'{save_dir}/{video_name}_pred{threshold}.{video_format}'
out_csv_file = f'{save_dir}/{video_name}_ball.csv'

checkpoint = torch.load(model_file)
param_dict = checkpoint['param_dict']
model_name = param_dict['model_name']
num_frame = param_dict['num_frame']
input_type = param_dict['input_type']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
 
# Load model
model = get_model(model_name, num_frame, input_type)
model.load_state_dict(checkpoint['model_state_dict'])
model  = torch.nn.DataParallel(model, device_ids =[0,1])
model = model.cuda()

model.eval()

conv_kernel = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
conv_kernel.weight.data.fill_(1)
conv_kernel  = torch.nn.DataParallel(conv_kernel, device_ids =[0,1])
conv_kernel = conv_kernel.cuda()


# Video output configuration
if video_format == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
elif video_format == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    raise ValueError('Invalid video format.')

# Write csv file head
f = open(out_csv_file, 'w')
f.write('Frame,Visibility,X,Y\n')

# Cap configuration
cap = cv2.VideoCapture(video_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
success = True
frame_count = 0
num_final_frame = 0
ratio_h = h / HEIGHT
ratio_w = w / WIDTH
out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

while success:
    print(f'Number of sampled frames: {frame_count}')
    # Sample frames to form input sequence
    frame_queue = []
    for _ in range(num_frame*batch_size):
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            frame_queue.append(frame)

    if not frame_queue:
        break
    
    # If mini batch incomplete
    if len(frame_queue) % num_frame != 0:
        frame_queue = []
        # Record the length of remain frames
        num_final_frame = len(frame_queue) +1
        print(num_final_frame)
        # Adjust the sample timestampe of cap
        frame_count = frame_count - num_frame*batch_size
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        # Re-sample mini batch
        for _ in range(num_frame*batch_size):
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_count += 1
                frame_queue.append(frame)
        if len(frame_queue) % num_frame != 0:
            continue
    
    x = get_frame_unit(frame_queue, num_frame)
    # x  = torch.nn.DataParallel(x, device_ids =[0,1])
    # Inference
    with torch.no_grad():
        y_pred = model(x.cuda())
    

    # 应用卷积核
    output_tensor = conv_kernel(y_pred.to(dtype=torch.float32))
    
    
    for i in range(y_pred.shape[1]):
        if num_final_frame > 0 and i < (num_frame*batch_size - num_final_frame-1):
            print('aaa')
            # Special case of last incomplete mini batch
            # Igore the frame which is already written to the output video
            continue 
        else:

            start_width_in_output_tensor = int(start_width/ratio_w)  # 开始的宽度索引
            end_width_in_output_tensor = int(end_width/ratio_w)    # 结束的宽度索引
            # print("output_tensor[0, i, :, :].shape",output_tensor[0, i, :, start_width_in_output_tensor:end_width_in_output_tensor].shape)
            # 找出输出特征层中的最大值及其位置
            max_index = torch.argmax(output_tensor[0, i, :,start_width_in_output_tensor:end_width_in_output_tensor])
            # 将最大值位置转换为 (x, y) 坐标
            max_2d_index = np.unravel_index(max_index.item(), output_tensor[0, i, :, start_width_in_output_tensor:end_width_in_output_tensor].shape)
            if output_tensor[0, i, max_2d_index[0], max_2d_index[1]+start_width_in_output_tensor] > threshold:
                img = frame_queue[i].copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cx_pred, cy_pred = int(ratio_w*(max_2d_index[1]+start_width_in_output_tensor)), int(ratio_h*max_2d_index[0])
                vis = 1 if cx_pred > 0 and cy_pred > 0 else 0
                # Write prediction result
                f.write(f'{frame_count-(num_frame*batch_size)+i},{vis},{cx_pred},{cy_pred}\n')
                # print(frame_count-(num_frame*batch_size)+i)
                if cx_pred != 0 or cy_pred != 0:
                    cv2.circle(img, (cx_pred, cy_pred), 5, (0, 0, 255), -1)
                out.write(img)
            else:
                img = frame_queue[i].copy()
                cx_pred, cy_pred = 0, 0
                vis = 0
                f.write(f'{frame_count - (num_frame * batch_size) + i},{vis},{cx_pred},{cy_pred}\n')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img)

out.release()
print('Done.')
