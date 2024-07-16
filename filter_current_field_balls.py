import pandas as pd
import argparse
import subprocess
import os
import cv2
import numpy as np


def time_to_start(end_time):
    """
    从end_time开始，找到下一个回合的开始帧
    :param end_time: 上一回合的结束帧
    :return: 下一回合的开始帧
    """
    global start_time, num_clip, exist_start
    if end_time >= (len(df) - 19):
        start_time = end_time
        exist_start = 0
        return start_time, exist_start
    for i in range(end_time, len(df) - 19):
        # 如果到达末尾，直接返回
        if i + 20 == len(df):
            start_time = end_time
            exist_start = 0
            return start_time, exist_start
        # 计算当前位置及其后20行的第二列总和
        sum_col2 = df.iloc[i:i + 20, 1].sum()
        # 如果总和大于等于9
        if sum_col2 >= 9:
            # 回合序数迭代
            num_clip += 1
            if i < 15:
                start_time = 0
            else:
                # 输出当前行前15行的第一列的数值
                start_time = df.iloc[i - 15, 0]
            print(f"第 {num_clip} 个clip的开始帧是{start_time}")
            # 1代表此行后回合开始
            df.iloc[start_time, 4] = 1
            exist_start = 1
            return start_time, exist_start
        else:
            exist_start = 0


def time_to_end(start_time):
    """
    从start_time开始，找到下一个回合的结束帧
    :param start_time: 当前回合的开始帧
    :return: 当前回合的结束帧
    """
    global end_time, exist_end
    if start_time >= (len(df) - 59):
        end_time = len(df)
        exist_end = 0
        return end_time, exist_end
    for i in range(start_time, len(df) - 59):
        # 如果到达末尾，直接返回
        if i + 60 == len(df):
            end_time = len(df)
            exist_end = 0
            return end_time, exist_end
        # 计算当前位置及其后60行的第二列总和
        sum_col2 = df.iloc[i:i + 60, 1].sum()
        # 如果总和小于10
        if sum_col2 < 10:
            if i+30 >= len(df) - 1:
                end_time = len(df)
            else:
                # 输出当前行后30行的第一列的数值
                end_time = df.iloc[i + 30, 0]
            print(f"第 {num_clip} 个clip的结束帧是{end_time}")
            exist_end = 1
            # 如果切片时间达到要求，在终端进行裁剪
            if (end_time-start_time)/30 > slice_shortest_time:
                # 0代表到此行回合结束
                df.iloc[end_time, 4] = 0
                outputmp4_path = f"{os.path.splitext(os.path.basename(inputmp4_path))[0]}video{num_clip}slice.mp4"
                if os.path.isfile(outputmp4_path):
                    os.remove(outputmp4_path)
                command = ['ffmpeg', '-i', str(inputmp4_path), "-ss", f"{start_time/30:.2f}", "-t", f"{(end_time-start_time)/30:.2f}", "-c:v", "libx264", "-c:a", "aac", str(outputmp4_path)]
                result = subprocess.run(command, capture_output=True, text=True)
                # 检查命令是否成功执行
                if result.returncode == 0:
                    print("Command executed successfully.")
                    print("Output:\n", result.stdout)
                else:
                    print(f"Error executing command: {result.stderr}")
            else:
                df.iloc[start_time, 4] = np.nan
                print(f"第{num_clip}个切片时间太短，不保存")
            return end_time, exist_end
        else:
            exist_end = 0


def insert_frame(insert_start_time):
    """
    从insert_start_time开始，插入帧
    原理：球向上飞行超出摄像机上界限，再到回落于视线中，此过程的空裆时间需视visibility为1。
    （在此过程中，球的y值会逐渐减小，直到变为0，然后再增大，所以分为两个阶段处理）
    :param insert_start_time: 从哪里开始插帧，默认为0（csv全部）
    :return: 无
    """
    # 逐渐变小阶段
    for i in range(insert_start_time, len(df) - 31):
        # 下一个片段开始信号
        next_segment_begins = 0
        # 砍两刀
        x_within_allowed_range = width * x_constraint_ratio < df.iloc[i, 2] < df.iloc[i + 1, 2] < width * (
                    1 - x_constraint_ratio) or width * x_constraint_ratio < df.iloc[i + 1, 2] < df.iloc[
            i, 2] < width * (1 - x_constraint_ratio)
        # 如果出现y值小于70且连续一帧内变小，则准备插帧
        if 70 > df.iloc[i, 3] > df.iloc[i + 1, 3] > 0 and x_within_allowed_range:
            # 最多插30帧
            for j in range(i + 2, i + 32):
                # 在连续变小之后，从第三帧开始，若碰到y值和x值为0的帧，则将其visibility置1
                if df.iloc[j, 2] == 0 and df.iloc[j, 3] == 0:
                    df.iloc[j, 1] = 1
                    # 既然已经出现了空档期，那么在之后遇到球，则为下一个片段开始，所以将信号置为1
                    next_segment_begins = 1
                # 有球，且下一个片段已经开始了，则结束插帧
                elif (df.iloc[j, 2] != 0 or df.iloc[j, 3] != 0) and next_segment_begins == 1:
                    break
    # 逐渐增大阶段
    for i in range(insert_start_time, len(df) - 1):
        # 上一个片段开始信号
        pre_segment_begins = 0
        # 砍两刀
        x_within_allowed_range = width * x_constraint_ratio < df.iloc[i, 2] < df.iloc[i + 1, 2] < width * (
                1 - x_constraint_ratio) or width * x_constraint_ratio < df.iloc[i + 1, 2] < df.iloc[
                                     i, 2] < width * (1 - x_constraint_ratio)
        # 30帧之后再往上插帧，最多插30帧
        if i >= 30:
            if 70 > df.iloc[i + 1, 3] > df.iloc[i, 3] > 0 and x_within_allowed_range:
                for j in reversed(range(i - 30, i + 1)):
                    if df.iloc[j, 2] == 0 and df.iloc[j, 3] == 0:
                        df.iloc[j, 1] = 1
                        pre_segment_begins = 1
                    elif (df.iloc[j, 2] != 0 or df.iloc[j, 3] != 0) and pre_segment_begins == 1:
                        break
        # 30帧之内往上插帧，最多插到开头
        else:
            if 70 > df.iloc[i + 1, 3] > df.iloc[i, 3] > 0 and x_within_allowed_range:
                for j in reversed(range(0, i + 1)):
                    if df.iloc[j, 2] == 0 and df.iloc[j, 3] == 0:
                        df.iloc[j, 1] = 1
                        pre_segment_begins = 1
                    elif (df.iloc[j, 2] != 0 or df.iloc[j, 3] != 0) and pre_segment_begins == 1:
                        break


def delete_wrong_balls(delete_start_time):
    def check_repeats(group):
        col3_counts = group.iloc[:, 2].value_counts()
        col4_counts = group.iloc[:, 3].value_counts()

        # 找出重复次数超过5次的元素
        col3_to_zero = col3_counts[col3_counts > 5].index.tolist()
        col4_to_zero = col4_counts[col4_counts > 5].index.tolist()

        # 将满足条件的行的第2、3、4列设置为0
        mask = (group.iloc[:, 2].isin(col3_to_zero)) | (group.iloc[:, 3].isin(col4_to_zero))
        group.iloc[mask, 1] = 0
        group.iloc[mask, 2] = 0
        group.iloc[mask, 3] = 0

        return group

    # 使用rolling窗口，但是由于rolling窗口不适用于非数值类型的数据，
    # 我们需要手动实现类似的功能
    window_size = 20
    for i in range(delete_start_time, len(df)):
        if i == len(df) - 21:
            # 取出一个窗口大小的数据块
            window_df = df.iloc[i:len(df)]
            # 应用我们的检查函数
            updated_window_df = check_repeats(window_df.copy())
            # 将结果放回原DataFrame
            df.iloc[i:len(df)] = updated_window_df
            break
        else:
            # 取出一个窗口大小的数据块
            window_df = df.iloc[i:i + window_size]
            # 应用我们的检查函数
            updated_window_df = check_repeats(window_df.copy())
            # 将结果放回原DataFrame
            df.iloc[i:i + window_size] = updated_window_df


def main(args):
    global num_clip, end_time, start_time, exist_start, exist_end
    num_clip = 0
    end_time = 0
    start_time = 0
    exist_start = 0
    exist_end = 0
    # 首先删除重复点位
    delete_wrong_balls(0)
    # 首先对整张表插帧，0代表从开头开始插
    insert_frame(0)
    start_time, exist_start = time_to_start(end_time)
    if exist_start == 0:
        print("整个视频没有任何回合")
        return 0
    while 1:
        end_time, exist_end = time_to_end(start_time)
        if exist_end:
            start_time, exist_start = time_to_start(end_time)
        if exist_start == 0 or exist_end == 0:
            break
    df.to_csv(args.output_csv_path, index=False)
    print("筛选完成，已显示时间戳")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="找出每一回合")
    parser.add_argument("--input_csv_path", type=str, default='skcourt2_ball.csv', help="需处理表格的路径")
    parser.add_argument("--output_csv_path", type=str, default='skcourt2_ballnew.csv', help="输出表格的路径")
    parser.add_argument("--inputmp4_path", type=str, default='skcourt2_pred7.mp4', help="输入视频路径")
    parser.add_argument("--slice_shortest_time", type=float, default=3.4, help="切片最短时间,单位秒")
    parser.add_argument("--x_constraint_ratio", type=float, default=0.29, help="插帧范围限制系数，避免其他场球干扰")
    args = parser.parse_args()
    inputmp4_path = args.inputmp4_path
    slice_shortest_time = args.slice_shortest_time
    x_constraint_ratio = args.x_constraint_ratio

    # 获取视频宽度，方便“砍两刀”
    cap = cv2.VideoCapture(inputmp4_path)
    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # 获取视频宽度，使用CAP_PROP_FRAME_WIDTH常量
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # 打印宽度信息
        print(f"The width of the video is: {width} pixels")
    # 释放VideoCapture资源
    cap.release()

    df = pd.read_csv(args.input_csv_path)
    # 在原csv中增加第五列
    df['开始和结束'] = None
    main(args)
