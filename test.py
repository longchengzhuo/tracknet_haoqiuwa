import argparse
from infer_and_find_rounds import infer_and_find_rounds

video_file = "/ssd2/cz/TrackNetV3/bt_for_test/bt2.mp4"  # 输入视频地址

parser = argparse.ArgumentParser()  # 以下内容应放入配置中心------------------
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
args = parser.parse_args()  # ------------------------------------

start_and_end_frame_list = infer_and_find_rounds(args, video_file)  # 开始推理,每推理600帧,就执行找回合任务

for i in start_and_end_frame_list:
    print("start_and_end_frame_list", i)  # start_and_end_frame_list:含有开始帧数和结束帧数