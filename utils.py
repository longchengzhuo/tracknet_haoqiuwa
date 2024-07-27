import cv2
import torch
import numpy as np


HEIGHT = 288
WIDTH = 512




def get_one_batch_frame(one_batch_frame_num, cap, frame_count, frame_queue, it_is_last_batch):
    frame_over = 0
    for _ in range(one_batch_frame_num):
        success, frame = cap.read()
        if not success:
            frame_over = 1
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            frame_queue.append(frame)
    # print(f'Number of sampled frames: {frame_count}')
    if frame_over == 1 and frame_count % one_batch_frame_num != 0:
        it_is_last_batch = 1
        inferred_frame_num = one_batch_frame_num - len(frame_queue)
        frame_count = frame_count - one_batch_frame_num
        # print(f"最后一个batch残缺，往前找补，从第{frame_count+1}帧开始重新加载一个batch")
        frame_queue = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        for _ in range(one_batch_frame_num):
            success, frame = cap.read()
            if not success:
                # print(f"读取到第{frame_count}帧结束，有问题")
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_count += 1
                frame_queue.append(frame)
    else:
        it_is_last_batch = 0
        inferred_frame_num = 0

    return frame_count, frame_queue, it_is_last_batch, inferred_frame_num





def post_processing(i, output_tensor, start_width_in_output_tensor, end_width_in_output_tensor, ratio_h, ratio_w,
                    frame_queue, threshold):
    # 找出输出特征层中的最大值及其位置
    max_index = torch.argmax(output_tensor[i, :, start_width_in_output_tensor:end_width_in_output_tensor])
    # 将最大值位置转换为 (x, y) 坐标
    max_2d_index = np.unravel_index(max_index.item(),
                                    output_tensor[i, :, start_width_in_output_tensor:end_width_in_output_tensor].shape)

    if output_tensor[i, max_2d_index[0], max_2d_index[1] + start_width_in_output_tensor] > threshold:
        cx_pred, cy_pred = int(ratio_w * (max_2d_index[1] + start_width_in_output_tensor)), int(
            ratio_h * max_2d_index[0])
        vis = 1
    else:
        cx_pred, cy_pred = 0, 0
        vis = 0
    return vis, cx_pred, cy_pred

def assemble_frames_by_batch(frame_list, num_frame):
    batch = []

    def get_unit(frame_list):
        frames = np.array([]).reshape(0, HEIGHT, WIDTH)
        for img in frame_list:
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = np.moveaxis(img, -1, 0)
            frames = np.concatenate((frames, img), axis=0)
        return frames

    for i in range(0, len(frame_list), num_frame):
        frames = get_unit(frame_list[i: i + num_frame])
        frames /= 255.
        batch.append(frames)

    batch = np.array(batch)
    return torch.FloatTensor(batch)


