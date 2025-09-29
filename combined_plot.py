#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf

def create_combined_plot(results_dir, output_file):
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot validation dice
    dice_val = np.load(f"{results_dir}/dice_val.npy")
    E, N, K = dice_val.shape
    epochs = np.arange(E)
    
    for k in range(1, K):
        axs[0, 0].plot(epochs, dice_val[:, :, k].mean(axis=1), label=f"Class {k}")
    axs[0, 0].plot(epochs, dice_val[:, :, 1:].mean(axis=2).mean(axis=1), label="All classes", linewidth=3)
    axs[0, 0].set_title('Validation Dice Score')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Dice')
    axs[0, 0].legend()
    
    # Plot training dice
    dice_tra = np.load(f"{results_dir}/dice_tra.npy")
    for k in range(1, K):
        axs[0, 1].plot(epochs, dice_tra[:, :, k].mean(axis=1), label=f"Class {k}")
    axs[0, 1].plot(epochs, dice_tra[:, :, 1:].mean(axis=2).mean(axis=1), label="All classes", linewidth=3)
    axs[0, 1].set_title('Training Dice Score')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Dice')
    axs[0, 1].legend()
    
    # Plot validation loss
    loss_val = np.load(f"{results_dir}/loss_val.npy")
    axs[1, 0].plot(epochs, loss_val.mean(axis=1), linewidth=3)
    axs[1, 0].set_title('Validation Loss')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    
    # Plot training loss
    loss_tra = np.load(f"{results_dir}/loss_tra.npy")
    axs[1, 1].plot(epochs, loss_tra.mean(axis=1), linewidth=3)
    axs[1, 1].set_title('Training Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    
    # Save as PDF
    fig.savefig(output_file)
    print(f"Combined plots saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create combined training plots")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with training metrics")
    parser.add_argument("--output", type=str, default="plot.pdf", help="Output PDF file")
    args = parser.parse_args()
    
    create_combined_plot(args.results_dir, args.output)