import math, time, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minigpt.parametre import B, T, d, V, nb_block, device
from minigpt.batch import get_batch
from minigpt.model import LLM_hugo

NB_STEPS, EVAL_EVERY, VAL_BATCHES = 20000, 500, 25
WARMUP, LR_MAX, LR_MIN = 300, 6e-4, 6e-5
WEIGHT_DECAY, CLIP = 0.1, 1.0
CKPT = "checkpoints/best_19M.pt"
os.makedirs("checkpoints", exist_ok=True)
torch.manual_seed(0)


def load(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64)).to(device)


train_data, val_data = load("data/s_train.bin"), load("data/s_val.bin")
assert train_data.max() < V and val_data.max() < V
print("train :", len(train_data), "| val :", len(val_data))

model = LLM_hugo()
print("paramètres :", sum(p.numel() for p in model.parameters()))
decay = [p for p in model.parameters() if p.dim() >= 2]       # matrices : decay
no_decay = [p for p in model.parameters() if p.dim() < 2]     # gamma, biais : pas de decay
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=LR_MAX, betas=(0.9, 0.95))
loss_fn = nn.CrossEntropyLoss()
val_batches = [get_batch(val_data, B, T, device) for _ in range(VAL_BATCHES)]


def lr_at(step):
    if step < WARMUP:
        return LR_MAX * step / WARMUP
    p = (step - WARMUP) / max(1, NB_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate():
    model.eval()
    tot = sum(loss_fn(model(x).reshape(B * T, V), y.reshape(B * T)).item() for x, y in val_batches)
    model.train()
    return tot / len(val_batches)


def plot(steps, tr, va):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, tr, label="train"); plt.plot(steps, va, label="validation")
    plt.xlabel("pas"); plt.ylabel("perte"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("loss_curve_19M.png", dpi=120, bbox_inches="tight"); plt.close()


steps, tr_hist, va_hist = [], [], []
best, running, count, t0 = float("inf"), 0.0, 0, time.time()
for step in range(1, NB_STEPS + 1):
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch(train_data, B, T, device)
    optimizer.zero_grad()
    loss = loss_fn(model(x).reshape(B * T, V), y.reshape(B * T))
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()
    running += loss.item(); count += 1

    if step % EVAL_EVERY == 0:
        tl, vl = running / count, evaluate()
        running, count = 0.0, 0
        steps.append(step); tr_hist.append(tl); va_hist.append(vl)
        flag = ""
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(),
                        "config": {"V": V, "d": d, "T": T, "nb_block": nb_block},
                        "step": step, "val_loss": vl}, CKPT)
            flag = "  <- sauvegardé"
        el = time.time() - t0
        print(f"pas {step:6d} | train {tl:.4f} | val {vl:.4f} (ppl {math.exp(vl):.1f}) | "
              f"lr {lr_at(step):.2e} | |g| {gn:.2f} | {el/60:.1f} min{flag}", flush=True)
        plot(steps, tr_hist, va_hist)
print("meilleure val :", best)
