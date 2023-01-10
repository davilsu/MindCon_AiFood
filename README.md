# MindCon_AiFood

## 模型推理步骤

```bash
cd code/
python ./predict_v2.py --batch_size 64 --model_path ./best.ckpt --data_path ./test --out_path ./
```

## 参数介绍

| 参数名称 | 描述 | 默认值 |
| --- | --- | --- |
| batch_size | 一次推理图片数量 | 64 |
| model_path | 权重路径 | |
| data_path | 测试集图片文件夹路径（要求图片名称格式为'xxx.jpg'，xxx为数字）| |
| out_path | 推理结果输出路径 | |

## 测试环境

| 硬件平台 | 运行环境 |
| --- | --- |
| CPU | mindspore_1.7.1-py_3.9-ubuntu_2110-x86_64 |
| GPU | mindspore_1.7.0-cuda_10.1-py_3.7-ubuntu_1804-x86_64 |
| Ascend | mindspore_1.7.0-cann_5.1.0-py_3.7-euler_2.8.3-aarch64 |

## 权重下载链接
<https://xihe.mindspore.cn/models/davilsu/MindCon_AiFood_Weights>
