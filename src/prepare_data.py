"""Decode GTSRB parquet files into compact 32x32 uint8 .npy arrays.

Source: tanganke/gtsrb on Hugging Face (mirror of the official GTSRB benchmark,
Stallkamp et al. 2012). Raw images are NOT redistributed in the deliverable.
"""
import io, sys, numpy as np, pandas as pd
from PIL import Image

SIZE = 32

def convert(name):
    df = pd.read_parquet(f"data/{name}.parquet")
    n = len(df)
    imgs = np.empty((n, SIZE, SIZE, 3), dtype=np.uint8)
    labels = df["label"].to_numpy(dtype=np.int64)
    for i, cell in enumerate(df["image"]):
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(raw)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        imgs[i] = np.asarray(im, dtype=np.uint8)
        if i % 5000 == 0:
            print(f"{name}: {i}/{n}", flush=True)
    np.save(f"data/{name}_x.npy", imgs)
    np.save(f"data/{name}_y.npy", labels)
    print(f"{name}: done shape={imgs.shape} classes={len(np.unique(labels))}", flush=True)

if __name__ == "__main__":
    convert(sys.argv[1])
