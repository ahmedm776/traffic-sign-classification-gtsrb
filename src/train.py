"""Time-budgeted, checkpoint-resumable training for the GTSRB compact CNN.

Why time-budgeted: this build box caps a single command at ~120s, so training
runs in chunks; every chunk saves a full checkpoint (model, optimizer, epoch,
history, RNG) and the next chunk resumes exactly. Final reported metrics come
from the completed run, not a partial one.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import CompactCNN

torch.set_num_threads(2)

def rand_affine_batch(x, max_rot=10.0, max_trans=0.10, scale_lo=0.9, scale_hi=1.1):
    b = x.shape[0]
    ang = (torch.rand(b) * 2 - 1) * (max_rot * 3.14159265 / 180.0)
    scale = torch.empty(b).uniform_(scale_lo, scale_hi)
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    tx = (torch.rand(b) * 2 - 1) * max_trans
    ty = (torch.rand(b) * 2 - 1) * max_trans
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")

@torch.no_grad()
def evaluate(model, x, y, bs=512):
    model.eval(); correct = 0; loss_sum = 0.0
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].float() / 255.0
        xb = (xb - 0.5) / 0.25
        out = model(xb)
        loss_sum += F.cross_entropy(out, y[i:i+bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i+bs]).sum().item()
    return loss_sum / len(x), correct / len(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=100)
    ap.add_argument("--max-epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ckpt", default="ckpt.pt")
    args = ap.parse_args()

    x_all = torch.from_numpy(np.load("data/train_x.npy")).permute(0, 3, 1, 2).contiguous()
    y_all = torch.from_numpy(np.load("data/train_y.npy"))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(x_all), generator=g)
    n_val = 2664
    val_i, tr_i = perm[:n_val], perm[n_val:]
    x_val, y_val = x_all[val_i], y_all[val_i]
    x_tr, y_tr = x_all[tr_i], y_all[tr_i]

    if os.path.exists(args.ckpt):
        c = torch.load(args.ckpt, weights_only=False)
        model = CompactCNN(); model.load_state_dict(c["model"])
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
        opt.load_state_dict(c["opt"])
        start_epoch, hist, rng_state = c["epoch"], c["history"], c["rng"]
        torch.set_rng_state(rng_state)
        print(f"resumed at epoch {start_epoch}", flush=True)
    else:
        model = CompactCNN()
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
        start_epoch, hist = 0, []
        print("fresh run", flush=True)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=0.002)
    for _ in range(start_epoch):
        sched.step()

    t0 = time.time(); done = False
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        order = torch.randperm(len(x_tr))
        for i in range(0, len(order), args.batch):
            idx = order[i:i+args.batch]
            xb = x_tr[idx].float() / 255.0
            xb = rand_affine_batch(xb)
            xb = (xb - 0.5) / 0.25
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vloss, vacc = evaluate(model, x_val, y_val)
        hist.append({"epoch": epoch + 1, "val_loss": round(vloss, 4), "val_acc": round(vacc, 4),
                     "lr": round(sched.get_last_lr()[0], 4)})
        print(f"epoch {epoch+1}/{args.max_epochs} val_loss={vloss:.4f} val_acc={vacc:.4f} "
              f"t={time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                    "history": hist, "rng": torch.get_rng_state()}, args.ckpt)
        if time.time() - t0 > args.seconds:
            print("time budget reached, checkpoint saved", flush=True)
            break
    else:
        done = True
    torch.save({"model": model.state_dict(), "history": hist, "done": done,
                "max_epochs": args.max_epochs}, "final_model.pt")
    print(json.dumps({"done": done, "epochs_completed": hist[-1]["epoch"] if hist else 0}), flush=True)

if __name__ == "__main__":
    main()
