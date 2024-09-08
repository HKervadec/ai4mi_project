#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p = np.linspace(0, 1, 100)
    print(p)
    H = p * (-np.log2(p)) + (1 - p) * (-np.log2(1 - p))

    plt.plot(p, H)
    plt.show()

    CE = lambda y_, p_: y_ * (-np.log2(p_)) + (1 - y_) * (-np.log2(1 - p_))

    y = np.linspace(0, 1, 100)
    Ys, Ps = np.meshgrid(y, p)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(Ys, Ps, CE(Ys, Ps))
    ax.set_xlabel("y")
    ax.set_ylabel("p")
    ax.set_zlabel("Cross-entropy")

    ax.plot_wireframe(Ys, Ys, CE(Ys, Ys), color="red")

    ax.plot_wireframe(Ys, 1 - Ys, CE(Ys, 1 - Ys), color="green")
    plt.show()
