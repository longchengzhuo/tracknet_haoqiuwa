import argparse
from utils import *


def get_model(num_frame):
    from model import TrackNetV3 as TrackNet
    model = TrackNet(in_dim=num_frame * 3, out_dim=num_frame)

    return model

def load_model(model_file):
    checkpoint = torch.load(model_file)
    param_dict = checkpoint['param_dict']
    num_frame = param_dict['num_frame']

    model = get_model(num_frame)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    conv_kernel = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
    conv_kernel.weight.data.fill_(1)
    conv_kernel = conv_kernel.cuda()

    return model, conv_kernel, num_frame

def start_infer(args, inferred_results, cap, w, h, frame_count, model, conv_kernel, num_frame):
    need_to_cut = args.need_to_cut                                                                                      # 获取参数
    batch_size = args.batch_size
    threshold = args.threshold
    start_width = args.start_width
    end_width = args.end_width

    if need_to_cut == 0:
        end_width = w

    one_batch_frame_num = num_frame * batch_size                                                                        # 参数初始化
    it_is_last_batch = 0
    ratio_h = h / HEIGHT
    ratio_w = w / WIDTH
    start_width_in_output_tensor = int(start_width / ratio_w)                                                           # 开始的宽度索引
    end_width_in_output_tensor = int(end_width / ratio_w)                                                               # 结束的宽度索引
    success = True

    while success:                                                                                                      # 开始每一轮batch的推理
        frame_queue = []
        if it_is_last_batch:
            break
        frame_count, frame_queue, it_is_last_batch, inferred_frame_num = get_one_batch_frame(one_batch_frame_num, cap,  # 将一个batch所有图像保存在frame_queue中
                                                                            frame_count, frame_queue,it_is_last_batch)
        if not frame_queue:
            break
        x = assemble_frames_by_batch(frame_queue, num_frame)
        with torch.no_grad():
            y_pred = model(x.cuda())                                                                                    # 推理
        output_tensor = conv_kernel(y_pred).reshape(-1, HEIGHT, WIDTH)
        for i in range(output_tensor.shape[0]):
            if it_is_last_batch and i < inferred_frame_num:
                continue                                                                                                # 当最后一轮的frame数小于one_batch_frame_num时，不对已经后处理过的帧进行后处理
            else:
                vis, cx_pred, cy_pred = post_processing(i, output_tensor, start_width_in_output_tensor, end_width_in_output_tensor, ratio_h, ratio_w, frame_queue, threshold)
                inferred_result = {
                    'Frame': frame_count - one_batch_frame_num + i,
                    'Visibility': vis,
                    'X': cx_pred,
                    'Y': cy_pred,
                    '开始和结束': None,
                    '标记需要清空的列表': None}
                inferred_results.loc[len(inferred_results)] = inferred_result

        if len(inferred_results) % 600 == 0:
            return inferred_results, frame_count
    return inferred_results, frame_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='/ssd2/cz/TrackNetV3/bt12_train/bt_exp/model_best.pt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=3)
    parser.add_argument('--start_width', type=int, default=0)
    parser.add_argument('--end_width', type=int, default=1720)
    parser.add_argument('--need_to_cut', type=int, default=0)   # 是否需要两端砍两刀
    args = parser.parse_args()
    inferred_results = {                                                                                                # 缓冲推理结果
        'Frame': [],
        'Visibility': [],
        'X': [],
        'Y': [],
        '开始和结束': [],
        '标记需要清空的列表': []}
    inferred_results = pd.DataFrame(inferred_results)
    video_file = "2024-07-23_18-55-41.mp4"
    cap = cv2.VideoCapture(video_file)                                                                                  # 获取视频配置
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    model_file = args.model_file
    model, conv_kernel, num_frame = load_model(model_file)                                                              # 加载模型

    inferred_results, width_of_raw_mp4, frame_count = start_infer(args, inferred_results, cap, w, h, frame_count, model, conv_kernel, num_frame)
