#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from numpy import pi as π


TR = np.asarray([[1, 0, 0, 50],
                 [0,  1, 0, 40],  # noqa: E241
                 [0,             0,      1, 15],  # noqa: E241
                 [0,             0,      0, 1]])  # noqa: E241

DEG: int = 27
ϕ: float = - DEG / 180 * π
RO = np.asarray([[np.cos(ϕ), -np.sin(ϕ), 0, 0],  # noqa: E241, E201
                 [np.sin(ϕ),  np.cos(ϕ), 0, 0],  # noqa: E241
                 [     0,         0,     1, 0],  # noqa: E241, E201
                 [     0,         0,     0, 1]])  # noqa: E241, E201

X_bar: float = 275
Y_bar: float = 200
Z_bar: float = 0
C1 = np.asarray([[1, 0, 0, X_bar],
                 [0, 1, 0, Y_bar],
                 [0, 0, 1, Z_bar],
                 [0, 0, 0,    1]])  # noqa: E241
C2 = np.linalg.inv(C1)

AFF = C1 @ RO @ C2 @ TR
INV = np.linalg.inv(AFF)
print(f"{AFF=}")
print(f"{RO=}")
print(f"{AFF=}")
print(f"{INV=}")
