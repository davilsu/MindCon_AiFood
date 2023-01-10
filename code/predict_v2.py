import numpy as np
import os
from PIL import Image
import argparse
import datetime
from tqdm import tqdm

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset.vision.c_transforms as T
from mindspore.dataset.transforms.c_transforms import Compose

from convnext_v2 import ConvNextV2_H

ms.set_context(mode=ms.context.GRAPH_MODE)


def image_process(image_path: str):
    MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = Compose([
        T.Resize(384, interpolation=T.Inter.BICUBIC),
        T.CenterCrop(384),
        T.Normalize(mean=MEAN, std=STD),
        T.HWC2CHW()
    ])
    images = [x for x in os.listdir(image_path) if x.endswith('.jpg')]
    images = sorted(images, key=lambda x: int(x.split('.jpg')[0]))
    images = [os.path.join(image_path, x) for x in images]
    images = [Image.open(x).convert("RGB") for x in images]
    images = [trans(x) for x in images]
    images = np.stack(images)
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()

    model = ConvNextV2_H(10, drop_rate=0)
    ms.load_param_into_net(model, ms.load_checkpoint(args.model_path))
    model = nn.SequentialCell(model, nn.Softmax()).set_train(False)
    if ms.get_context("device_target") != "CPU":
        model = model.to_float(ms.float16)
    model = ms.Model(model)

    result = np.zeros([0], dtype=np.int64)
    images = image_process(args.data_path)
    split_index = [(x + 1) * args.batch_size for x in range(images.shape[0] // args.batch_size)]
    for x in tqdm(np.split(images, split_index, axis=0), desc="Inference"):
        x = ms.Tensor(x, ms.float32)
        y = np.argmax(model.predict(x).asnumpy(), axis=1).astype(np.int64)
        result = np.concatenate([result, y])

    file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt'
    with open(os.path.join(args.out_path, file_name), 'w') as f:
        for x in result.tolist():
            f.write(f"{x}\n")
        f.write("\n")
