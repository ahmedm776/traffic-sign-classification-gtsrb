"""Final evaluation: official GTSRB test set + motion-blur corruption, reload check."""
import json, numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import CompactCNN, CLASS_NAMES

torch.set_num_threads(2)

@torch.no_grad()
def predict(model, x, bs=512):
    model.eval(); outs = []
    for i in range(0, len(x), bs):
        xb = (x[i:i+bs].float() / 255.0 - 0.5) / 0.25
        outs.append(model(xb))
    return torch.cat(outs)

def load(name):
    x = torch.from_numpy(np.load(f"data/{name}_x.npy")).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.load(f"data/{name}_y.npy"))
    return x, y

ckpt = torch.load("final_model.pt", weights_only=False)
model = CompactCNN(); model.load_state_dict(ckpt["model"])

# reload determinism: fresh instance must produce identical predictions
model2 = CompactCNN(); model2.load_state_dict(torch.load("final_model.pt", weights_only=False)["model"])
x_te, y_te = load("test")
p1 = predict(model, x_te[:256]); p2 = predict(model2, x_te[:256])
assert torch.equal(p1.argmax(1), p2.argmax(1)), "reload mismatch"
reload_ok = True

results = {"reload_deterministic": reload_ok, "history": ckpt["history"]}
for name in ["test", "motion_blur"]:
    x, y = load(name)
    logits = predict(model, x)
    pred = logits.argmax(1)
    acc = (pred == y).float().mean().item()
    per_class = {}
    for c in range(43):
        m = y == c
        per_class[c] = {"n": int(m.sum()), "acc": round(float((pred[m] == c).float().mean()), 4)}
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["acc"])[:5]
    results[name] = {"n": len(y), "accuracy": round(acc, 4),
                     "worst_classes": [{"id": int(c), "name": CLASS_NAMES[int(c)], **v} for c, v in worst]}
    print(name, "acc", round(acc, 4), flush=True)
    if name == "test":
        cm = np.zeros((43, 43), dtype=int)
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t][p] += 1
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(np.log1p(cm), cmap="viridis"); ax.set_title("GTSRB test confusion matrix (log counts)")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        fig.tight_layout(); fig.savefig("confusion_matrix.png", dpi=110); plt.close(fig)
        # misclassified samples grid
        wrong = (pred != y).nonzero().flatten()[:24]
        k = len(wrong)
        if k:
            cols = 8; rows = int(np.ceil(k / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(12, 1.6 * rows))
            for ax, i in zip(np.atleast_1d(axes).ravel(), wrong.tolist()):
                ax.imshow(x[i].permute(1, 2, 0).numpy())
                ax.set_title(f"T{y[i].item()} P{pred[i].item()}", fontsize=7); ax.axis("off")
            for ax in np.atleast_1d(axes).ravel()[k:]:
                ax.axis("off")
            fig.suptitle(f"First {k} misclassified official-test images (T=true, P=pred)", fontsize=10)
            fig.tight_layout(); fig.savefig("misclassified.png", dpi=110); plt.close(fig)

json.dump(results, open("metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=1)[:900], flush=True)
