import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import CompactCNN, CLASS_NAMES
torch.set_num_threads(2)
m = CompactCNN(); m.load_state_dict(torch.load("final_model.pt", weights_only=False)["model"]); m.eval()
x = np.load("data/test_x.npy", mmap_mode="r")[:24]
y = np.load("data/test_y.npy", mmap_mode="r")[:24]
xt = torch.from_numpy(np.asarray(x)).permute(0, 3, 1, 2).float() / 255.0
with torch.no_grad():
    probs = torch.softmax(m((xt - 0.5) / 0.25), dim=1)
conf, pred = probs.max(1)
fig, axes = plt.subplots(3, 8, figsize=(15, 6.2))
for ax, i in zip(axes.ravel(), range(24)):
    ax.imshow(x[i]); ok = pred[i].item() == int(y[i])
    ax.set_title(f"{conf[i]:.0%} {CLASS_NAMES[pred[i]][:28]}", fontsize=6.5,
                 color="green" if ok else "red")
    ax.axis("off")
fig.suptitle("Live model predictions on held-out official GTSRB test images (green=correct)", fontsize=10)
fig.tight_layout(); fig.savefig("demo_predictions.png", dpi=110)
print("saved demo_predictions.png")
