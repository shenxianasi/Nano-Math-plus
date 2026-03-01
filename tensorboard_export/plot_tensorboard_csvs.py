import os
import math
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def load_csv(path):
    try:
        df = pd.read_csv(path)
        if "step" not in df.columns or "value" not in df.columns:
            return None
        df = df.sort_values("step")
        return df
    except Exception:
        return None

def plot_page(files, output_path, ncols=4, height=2.6, width=4.2):
    n = len(files)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * width, nrows * height))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    for idx, path in enumerate(files):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        df = load_csv(path)
        title = os.path.basename(path).replace(".csv", "")
        if df is None or df.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.plot(df["step"], df["value"], linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel("value", fontsize=8)
        ax.tick_params(labelsize=7)
    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def main():
    base_dir = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(base_dir, "*.csv")) + glob.glob(os.path.join(base_dir, ".*.csv")))
    if not files:
        print("no csv files")
        return
    per_page = 24
    for page_idx, batch in enumerate(chunk_list(files, per_page), start=1):
        output_path = os.path.join(base_dir, f"plots_page_{page_idx}.png")
        plot_page(batch, output_path)
        print(f"saved {output_path}")

if __name__ == "__main__":
    main()
