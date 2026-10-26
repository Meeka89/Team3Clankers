"""
Carter Owens

1. After training, reshape each of the 16 kernels back to 3 × 3 and render them as a 4 × 4 grid of
grayscale tiles; save to kernels.png.
2. In a one-paragraph markdown note, describe what you see. Some kernels may look like oriented
edges or spots; others may be noisy and hard to read. Say which is which — don’t force a story
onto a tile that doesn’t show one.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():

    npz_path = Path(__file__).with_name("trained_weights.npz")
    data = np.load(npz_path)
    
    kernels = data["kernels"]

    tiles = kernels.T.reshape(16, 3, 3)                   # 16 filters, each 3x3

    fig, axes = plt.subplots(4, 4)

    for i, (tile, ax) in enumerate(zip(tiles, axes.flat)):
        ax.imshow(tile, cmap="gray")
        ax.set_title(f"Kernel {i}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Learned 3x3 Kernels")
    fig.tight_layout()
    fig.savefig("week10/kernels.png")

if __name__ == "__main__":
    main()