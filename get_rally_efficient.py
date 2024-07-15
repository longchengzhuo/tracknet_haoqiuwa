import pandas as pd
import math
import argparse


def get_efficient():
    key_startend_row = key_df.iloc[0:, 4]
    startend_row = df.iloc[0:, 4]
    # 包含所有球的结束数（即总回合数）
    total_real_rally = (key_startend_row == 0).sum()
    # 筛选推理后得到的结束数，不包括3.3秒以内的回合
    total_rally = (startend_row == 0).sum()
    # 初始化总帧数
    correct_rally = 0
    for i in range(len(startend_row)):
        # 在90帧内开始的回合数
        if (key_startend_row[i:i+90] == 1).sum() == 1 and startend_row[i] == 1:
            for j in range(i,len(startend_row)):
                # 在100帧内结束的回合数
                if (key_startend_row[j - 100:j + 1] == 0).sum() == 1 and startend_row[j] == 0:
                    correct_rally += 1
                    break
    print("标准总回合数", total_real_rally)
    print("推理所得回合数", total_rally)
    print("查全率", total_rally/total_real_rally)
    print("正确回合数", correct_rally)
    print("有效率", correct_rally/total_rally)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算筛回合脚本的正确率")
    parser.add_argument("--key_csv_path", type=str, default='E:/biancheng\get_badminton_rounds/testkey.csv', help="标准答案表格")
    parser.add_argument("--csv_path", type=str, default='E:/biancheng\get_badminton_rounds/test.csv', help="推理所获表格")
    args = parser.parse_args()
    key_csv_path = args.key_csv_path
    csv_path = args.csv_path
    # 读取CSV文件
    key_df = pd.read_csv(key_csv_path)
    df = pd.read_csv(csv_path)
    get_efficient()
