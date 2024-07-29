import numpy as np
import time
def time_to_start(df, end_time, start_window, num_clip):
    """
    从end_time开始，找到下一个回合的开始帧
    :param end_time: 上一回合的结束帧
    :return: 下一回合的开始帧
    """
    if end_time >= (len(df) - start_window):
        start_time = end_time
        exist_start = 0
        return start_time, exist_start, num_clip, df
    for i in range(end_time, len(df) - (start_window - 1)):
        # 如果到达末尾，直接返回
        if i + start_window == len(df):
            start_time = end_time
            exist_start = 0
            return start_time, exist_start, num_clip, df
        # 计算当前位置及其后20行的第二列总和
        sum_col2 = df.iloc[i:i + start_window, 1].sum()
        # 如果总和大于等于9
        if sum_col2 >= 9:
            # 回合序数迭代
            num_clip += 1
            if i < 35:
                start_time = 0
            else:
                # 输出当前行前15行的第一列的数值
                start_time = df.iloc[i - 35, 0]
            # print(f"第 {num_clip} 个clip的开始帧是{start_time}")
            # 1代表此行后回合开始
            df.iloc[start_time, 4] = 1
            exist_start = 1
            return start_time, exist_start, num_clip, df

def time_to_end(df, start_time, end_window, slice_shortest_time, num_clip, start_and_end_frame_list):
    """
    从start_time开始，找到下一个回合的结束帧
    :param start_time: 当前回合的开始帧
    :return: 当前回合的结束帧
    """
    if start_time >= (len(df) - end_window):
        end_time = len(df)
        exist_end = 0
        return end_time, exist_end, start_and_end_frame_list, df
    for i in range(start_time, len(df) - (end_window - 1)):
        # 如果到达末尾，直接返回
        if i + end_window == len(df):
            end_time = len(df)
            exist_end = 0
            return end_time, exist_end, start_and_end_frame_list, df
        # 计算当前位置及其后100行的第二列总和
        sum_col2 = df.iloc[i:i + end_window, 1].sum()
        # 如果总和小于10
        if sum_col2 < 10:
            if i+40 >= len(df):
                end_time = len(df)
            else:
                # 输出当前行后40行的第一列的数值
                end_time = df.iloc[i + 40, 0]
            exist_end = 1
            # 如果切片时间达到要求，在终端进行裁剪
            if (end_time-start_time)/30 > slice_shortest_time:
                # 0代表到此行回合结束
                df.iloc[end_time, 4] = 0
                every_clip_frame_list = [start_time, end_time]
                start_and_end_frame_list.append(every_clip_frame_list)
            else:
                df.iloc[start_time, 4] = np.nan
            return end_time, exist_end, start_and_end_frame_list, df

def insert_frame(df, insert_start_time, width, x_constraint_ratio):
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
        x_within_allowed_range = width * x_constraint_ratio < df.iloc[i, 2] <= df.iloc[i + 1, 2] < width * (
                    1 - x_constraint_ratio) or width * x_constraint_ratio < df.iloc[i + 1, 2] <= df.iloc[
            i, 2] < width * (1 - x_constraint_ratio)
        # 如果出现y值小于70且连续一帧内变小，则准备插帧
        if 130 > df.iloc[i, 3] > df.iloc[i + 1, 3] > 0 and x_within_allowed_range:
            # 最多插30帧
            for j in range(i + 2, i + 32):
                # 在连续变小之后，从第三帧开始，若碰到y值和x值为0的帧，则将其visibility置1
                if df.iloc[j, 2] == 0 and df.iloc[j, 3] == 0:
                    df.iloc[j, 1] = 1
                    # 既然已经出现了空档期，那么在之后遇到球，则为下一个片段开始，所以将信号置为1
                    next_segment_begins = 1
                # 有球，且下一个片段已经开始了，则结束插帧
                elif df.iloc[j, 3] > df.iloc[i + 1, 3]:
                    break
                elif j == i + 12 and next_segment_begins == 0:
                    break
    # 逐渐增大阶段
    for i in range(insert_start_time, len(df) - 1):
        # 上一个片段开始信号
        pre_segment_begins = 0
        # 砍两刀
        x_within_allowed_range = width * x_constraint_ratio < df.iloc[i, 2] <= df.iloc[i + 1, 2] < width * (
                1 - x_constraint_ratio) or width * x_constraint_ratio < df.iloc[i + 1, 2] <= df.iloc[
                                     i, 2] < width * (1 - x_constraint_ratio)
        # 30帧之后再往上插帧，最多插30帧
        if i >= 30:
            if 130 > df.iloc[i + 1, 3] > df.iloc[i, 3] > 0 and x_within_allowed_range:
                for j in reversed(range(i - 30, i)):
                    if df.iloc[j, 2] == 0 and df.iloc[j, 3] == 0:
                        df.iloc[j, 1] = 1
                        pre_segment_begins = 1
                    elif df.iloc[j, 3] > df.iloc[i, 3]:
                        break
                    elif j == i - 10 and pre_segment_begins == 0:
                        break
        # 30帧之内往上插帧，最多插到开头
        else:
            if 130 > df.iloc[i + 1, 3] > df.iloc[i, 3] > 0 and x_within_allowed_range:
                for j in reversed(range(0, i)):
                    if df.iloc[j, 2] == 0 and df.iloc[j, 3] == 0:
                        df.iloc[j, 1] = 1
                        pre_segment_begins = 1
                    elif df.iloc[j, 3] > df.iloc[i, 3]:
                        break
                    elif j == i - 10 and pre_segment_begins == 0:
                        break

    return df

def delete_wrong_balls(df, delete_start_time, delete_window):
    def check_repeats(group):
        duplicated_rows_index = group.iloc[:, 1:4].duplicated(keep=False)
        # 将这些行标记为需要清空
        group.iloc[duplicated_rows_index, 5] = 1
        return group

    def check_wrong_ball(group):
        # 除了中间两个，其他位置都没有球，则整个窗口标记为需要清空
        balls_counts = group.iloc[:7, 1].sum() + group.iloc[13:, 1].sum()
        if balls_counts == 0:
            group.iloc[:, 5] = 1
        return group

    for i in range(delete_start_time, len(df)):
        if i == len(df) - delete_window:
            window_df = df.iloc[i:len(df)]
            df.iloc[i:i + delete_window] = check_repeats(window_df.copy())
            break
        else:
            window_df = df.iloc[i:i + delete_window]
            df.iloc[i:i + delete_window] = check_repeats(window_df.copy())
    repeats_mask = df.iloc[delete_start_time:, 5] == 1
    df.iloc[delete_start_time:, :].iloc[repeats_mask, 1:4] = 0

    for i in range(delete_start_time, len(df)):
        if i == len(df) - delete_window:
            window_df = df.iloc[i:len(df)]
            df.iloc[i:len(df)] = check_wrong_ball(window_df.copy())
            break
        else:
            window_df = df.iloc[i:i + delete_window]
            df.iloc[i:i + delete_window] = check_wrong_ball(window_df.copy())
    repeats_mask = df.iloc[delete_start_time:, 5] == 1
    df.iloc[delete_start_time:, :].iloc[repeats_mask, 1:4] = 0

    return df

def find_rounds(args, df, width, start_and_end_frame_list, one_no_start_zero_no_end, star_or_end_time, last_clip_end_frame):
    slice_shortest_time = args.slice_shortest_time
    x_constraint_ratio = args.x_constraint_ratio
    start_window = args.start_window
    end_window = args.end_window
    delete_window = args.delete_window
    num_clip = 0

    start_time2 = time.time()
    # 首先删除重复点位
    df = delete_wrong_balls(df, last_clip_end_frame, delete_window)
    end_time2 = time.time()
    execution_time = end_time2 - start_time2
    print(f"每次删除重复点位耗时: {execution_time:.6f} 秒")

    # 首先对整张表插帧，0代表从开头开始插
    df = insert_frame(df, last_clip_end_frame, width, x_constraint_ratio)
    end_time3 = time.time()
    execution_time = end_time3 - end_time2
    print(f"每次插帧耗时: {execution_time:.6f} 秒")

    if one_no_start_zero_no_end == 1:
        end_time = star_or_end_time
        start_time, exist_start, num_clip, df = time_to_start(df, end_time, start_window, num_clip)
        if exist_start == 0:
            # print("整个视频没有任何回合")
            one_no_start_zero_no_end = 1
            return start_and_end_frame_list, end_time, one_no_start_zero_no_end
        while 1:
            end_time, exist_end, start_and_end_frame_list, df = time_to_end(df, start_time, end_window, slice_shortest_time, num_clip, start_and_end_frame_list)
            if exist_end:
                start_time, exist_start, num_clip, df = time_to_start(df, end_time, start_window, num_clip)
            if exist_start == 0:
                one_no_start_zero_no_end = 1
                return start_and_end_frame_list, end_time, one_no_start_zero_no_end
            elif exist_end == 0:
                one_no_start_zero_no_end = 0
                return start_and_end_frame_list, start_time, one_no_start_zero_no_end

    elif one_no_start_zero_no_end == 0:
        start_time = star_or_end_time
        end_time, exist_end, start_and_end_frame_list, df = time_to_end(df, start_time, end_window, slice_shortest_time, num_clip, start_and_end_frame_list)
        if exist_end == 0:
            # print("整个视频一直在打")
            one_no_start_zero_no_end = 0
            return start_and_end_frame_list, start_time, one_no_start_zero_no_end
        while 1:
            start_time, exist_start, num_clip, df = time_to_start(df, end_time, start_window, num_clip)
            if exist_start:
                end_time, exist_end, start_and_end_frame_list, df = time_to_end(df, start_time, end_window, slice_shortest_time, num_clip, start_and_end_frame_list)
            if exist_start == 0:
                one_no_start_zero_no_end = 1
                return start_and_end_frame_list, end_time, one_no_start_zero_no_end
            elif exist_end == 0:
                one_no_start_zero_no_end = 0
                return start_and_end_frame_list, start_time, one_no_start_zero_no_end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="找出每一回合")
    parser.add_argument("--slice_shortest_time", type=float, default=3.4, help="切片最短时间,单位秒")
    parser.add_argument("--x_constraint_ratio", type=float, default=0.18, help="插帧范围限制系数，避免其他场球干扰")
    parser.add_argument("--start_window", type=int, default=20, help="滑动窗口检测开始帧")
    parser.add_argument("--end_window", type=int, default=70, help="滑动窗口检测结束帧")
    parser.add_argument("--delete_window", type=int, default=20, help="滑动窗口删除重复球")
    args = parser.parse_args()
    inferred_result = {  # 缓冲推理结果
        'Frame': [],
        'Visibility': [],
        'X': [],
        'Y': [],
        '开始和结束': [],
        '标记需要清空的列表': []}
    df = pd.DataFrame(inferred_result)
    width = 1920
    start_and_end_frame_list = []
    one_no_start_zero_no_end = 1
    star_or_end_time = 0
    start_and_end_frame_list, star_or_end_time, one_no_start_zero_no_end = find_rounds(args, df, width, start_and_end_frame_list, one_no_start_zero_no_end, star_or_end_time)
