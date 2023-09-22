from __future__ import annotations

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

import paddlefx

from paddlefx.compiler import custom_compiler


def add(x, y):
    return x + y


optimized_net = paddlefx.optimize(custom_compiler)(add)

x = paddle.rand([1, 224])
out = add(x, x)
res = optimized_net(x, x)
res = optimized_net(x, x)
res = optimized_net(x, x)

np.testing.assert_equal(res.numpy(), out.numpy())
