import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class Identity(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x: ms.Tensor):
        return x


class GlobalAvgPool(nn.Cell):
    def __init__(self, keep_dims: bool = False):
        super().__init__()
        self.axis = (2, 3)
        self.mean = ops.ReduceMean(keep_dims=keep_dims)

    def construct(self, x: ms.Tensor):
        return self.mean(x, self.axis)


class DropPath(nn.Cell):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(keep_prob=1.0-drop_prob)
        self.ones = ops.Ones()

    def construct(self, x: ms.Tensor):
        if not self.training:
            return x
        mask = self.ones((x.shape[0], 1, 1, 1), ms.float32)
        mask = self.drop(mask)
        out = mask * x
        return out


class Permute(nn.Cell):
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims)
        self.transpose = ops.Transpose()

    def construct(self, x: ms.Tensor):
        return self.transpose(x, self.dims)

    def extend_repr(self):
        return f"dims={self.dims}"


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim: int):
        super().__init__(normalized_shape=(dim, ), epsilon=1e-6)
        self.to_nhwc = Permute((0, 2, 3, 1))
        self.to_nchw = Permute((0, 3, 1, 2))

    def construct(self, x: ms.Tensor):
        x = self.to_nhwc(x)
        x = super().construct(x)
        x = self.to_nchw(x)
        return x


class GRN(nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = ms.Parameter(ms.Tensor(np.zeros([1, 1, 1, dim], dtype=np.float32), dtype=ms.float32))
        self.beta = ms.Parameter(ms.Tensor(np.zeros([1, 1, 1, dim], dtype=np.float32), dtype=ms.float32))
        self.norm = nn.Norm(axis=(1, 2), keep_dims=True)
        self.mean = ops.ReduceMean(keep_dims=True)
        self.eps = 1e-6

    def construct(self, x: ms.Tensor):
        Gx = self.norm(x)
        Nx = Gx / (self.mean(Gx, 3) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextV2Block(nn.Cell):
    def __init__(self, dim: int, drop_path: float = 0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode="same", group=dim, has_bias=True)
        self.to_nhwc = Permute((0, 2, 3, 1))
        self.norm = nn.LayerNorm((dim, ), epsilon=1e-6)
        self.pwconv1 = nn.Dense(in_channels=dim, out_channels=dim * 4, has_bias=True)
        self.act = nn.GELU()
        self.grn = GRN(dim * 4)
        self.pwconv2 = nn.Dense(in_channels=dim * 4, out_channels=dim, has_bias=True)
        self.to_nchw = Permute((0, 3, 1, 2))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else Identity()

    def construct(self, x):
        inputs = x
        x = self.dwconv(x)
        x = self.to_nhwc(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.to_nchw(x)
        x = self.drop_path(x)
        return x + inputs


class ConvNextV2(nn.Cell):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_rate=0.0):
        super().__init__()
        assert in_chans > 0 and num_classes > 0
        assert len(depths) == len(dims)
        assert drop_rate >= 0.
        dp_rates = np.linspace(0, drop_rate, num=sum(depths), dtype=np.float64).tolist()
        layers = []
        # Stem
        layers.append(
            nn.SequentialCell(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, pad_mode="valid", has_bias=True),
                LayerNorm2d(dims[0])
            )
        )
        # Stages
        for i in range(len(depths)):
            stage = []
            # ConvNeXtV2 blocks
            for j in range(depths[i]):
                dp = dp_rates[sum(depths[:i]) + j]
                stage.append(ConvNextV2Block(dims[i], drop_path=dp))
            layers.append(nn.SequentialCell(*stage))
            # Downsample blocks
            if i < len(depths) - 1:
                layers.append(
                    nn.SequentialCell(
                        LayerNorm2d(dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, pad_mode="valid", has_bias=True),
                    )
                )

        self.features = nn.SequentialCell(*layers)
        self.pooling = nn.SequentialCell(
            GlobalAvgPool(keep_dims=True),
            LayerNorm2d(dims[-1]),
            nn.Flatten(),
            nn.Dropout(keep_prob=0.75)
        )
        self.classifier = nn.SequentialCell(
            nn.Dense(dims[-1], dims[-1] * 2, has_bias=True),
            nn.GELU(),
            nn.Dropout(keep_prob=0.75),
            nn.Dense(dims[-1] * 2, dims[-1], has_bias=True),
            nn.LayerNorm((dims[-1], ), epsilon=1e-6),
            nn.Dense(dims[-1], dims[-1] // 2, has_bias=True),
            nn.GELU(),
            nn.Dropout(keep_prob=0.75),
            nn.Dense(dims[-1] // 2, num_classes, has_bias=True),
        )

    def construct(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = self.classifier(x)
        return x


def ConvNextV2_H(num_classes: int, drop_rate: float = 0.3):
    return ConvNextV2(num_classes=num_classes, depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], drop_rate=drop_rate)
