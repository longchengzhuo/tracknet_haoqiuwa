### git clone -b master <repository-url>
start_and_end_frame_list:含有开始帧数和结束帧数的list

格式类似于:[[155, 273], [435, 671], [712, 818], [899, 1090]]

这表明一共4个切片，第一个切片开始帧为155，结束帧为273，依此类推。

```python
    for i in start_and_end_frame_list:
        print("start_and_end_frame_list", i)
```

每循环一次，将往后推理600帧数，i将逐次更新，如下图所示。

![[Pasted image 20240728065652.png]]

坂田12号场权重：/ssd2/cz/TrackNetV3/bt12_train/bt_exp/model_best.pt

时刻127号场权重：/ssd2/cz/TrackNetV3/sk_train/sk127_exp/model_best.pt

时刻3456号场权重：/ssd2/cz/TrackNetV3/sk_train/sk3456_exp/model_last.pt

新羽胜1号场权重：/ssd2/cz/TrackNetV3/xys1_train/xys1_exp/model_best.pt

新羽胜2号场权重：/ssd2/cz/TrackNetV3/xys2_train/xys2_exp/model_best.pt
