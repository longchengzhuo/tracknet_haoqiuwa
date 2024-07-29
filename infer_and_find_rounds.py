from find_rounds import find_rounds
from predict import start_infer, load_model
import pandas as pd
import cv2
import argparse
import time


def infer_and_find_rounds(args, video_file, model, conv_kernel, num_frame):
    inferred_results = {                                                                                                # 缓冲推理结果
        'Frame': [],
        'Visibility': [],
        'X': [],
        'Y': [],
        '开始和结束': [],
        '标记需要清空的列表': []}
    inferred_results = pd.DataFrame(inferred_results)

    cap = cv2.VideoCapture(video_file)                                                                                  # 获取视频配置
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    one_no_start_zero_no_end = 1                                                                                        # 没有找到开始就是1,没有找到结束就是0
    star_or_end_time = 0
    frame_count = 0                                                                                                     # 帧数

    while len(inferred_results) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        inferred_results, frame_count = start_infer(args, inferred_results, cap, w, h, frame_count, model, conv_kernel, num_frame)
        start_and_end_frame_list = []                                                                                   # 储存开始帧和结束帧
        start_and_end_frame_list, star_or_end_time, one_no_start_zero_no_end = find_rounds(args, inferred_results, w,
                                                                                           start_and_end_frame_list,
                                                                                           one_no_start_zero_no_end,
                                                                                           star_or_end_time)
        yield start_and_end_frame_list


if __name__ == "__main__":
    video_file = "/ssd2/cz/TrackNetV3/bt_for_test/bt2.mp4"                                                              # 输入视频地址

    parser = argparse.ArgumentParser()                                                                                  # 以下内容应放入配置中心------------------
    parser.add_argument("--slice_shortest_time", type=float, default=3.4, help="切片最短时间,单位秒")
    parser.add_argument("--x_constraint_ratio", type=float, default=0.18, help="插帧范围限制系数，避免其他场球干扰")
    parser.add_argument("--start_window", type=int, default=20, help="滑动窗口检测开始帧")
    parser.add_argument("--end_window", type=int, default=70, help="滑动窗口检测结束帧")
    parser.add_argument("--delete_window", type=int, default=20, help="滑动窗口删除重复球")
    parser.add_argument('--model_file', type=str, default='/ssd2/cz/TrackNetV3/bt12_train/bt_exp/model_best.pt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=3)
    parser.add_argument('--start_width', type=int, default=0)
    parser.add_argument('--end_width', type=int, default=1720)
    parser.add_argument('--need_to_cut', type=int, default=0, help="是否需要两端砍两刀")
    args = parser.parse_args()                                                                                          # ------------------------------------

    model_file = args.model_file                                                                                        # 装载模型权重
    model, conv_kernel, num_frame = load_model(model_file)
    start_and_end_frame_list = infer_and_find_rounds(args, video_file, model, conv_kernel, num_frame)                   # 开始推理,每推理600帧,就执行找回合任务
    for i in start_and_end_frame_list:
        print("start_and_end_frame_list", i)                                                                            # start_and_end_frame_list:含有开始帧数和结束帧数的列表格式类似于:[[155, 273], [435, 671], [712, 818], [899, 1090]]
