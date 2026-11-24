"""HC-DeepOnet deep-ensemble surrogate for the high-rise office (3 targets).

Inputs : 10 static design variables (branch) + 20 daily weather features (trunk).
Targets: daily heating load [kWh], daily cooling load [kWh], daily discomfort hours [h]
         (occupant-weighted equivalent hours with zone |PMV|>1.5).
Method : branch/trunk DeepOnet with per-target heads; 5-member deep ensemble
         (per-member seed + bootstrap) -> ensemble mean mu_E and disagreement sigma_E;
         two-stage training (stage1 daily MSE; stage2 freeze encoders, daily+annual).
Splits : sample-level train/test (no day leakage); baseline (sample_id=1) excluded.

sigma_E (annual ensemble SD) is the signal the downstream uncertainty gate uses to
decide "trust surrogate" vs "fall back to EnergyPlus". This script also reports how
well sigma_E ranks the true error (Spearman), i.e. whether the gate is meaningful.

Self-contained: fits its own scalers + split; no dependency on demo3 artifacts.
Run: python train_office_surrogate.py            (5 members, GPU if available)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "datasets" / "office_daily_dataset.csv"
SUMMARY_PATH = ROOT / "datasets" / "office_sample_summary.csv"
OUTPUT_DIR = ROOT / "models" / "office_hc_deeponet_ensemble"

STATIC_COLUMNS = [
    "wall_insulation_m", "roof_insulation_m", "window_u_w_m2k", "window_shgc",
    "infiltration_ach", "window_wall_ratio", "cooling_setpoint_c",
    "heating_setpoint_c", "cooling_cop", "heating_cop",
]
# daily weather (lat/long/elev excluded: constant for a single weather file)
DYNAMIC_COLUMNS = [
    "day_of_year_sin", "day_of_year_cos",
    "drybulb_mean_c", "drybulb_min_c", "drybulb_max_c", "temp_range_c", "dewpoint_mean_c",
    "rh_mean_pct", "rh_min_pct", "rh_max_pct", "wind_mean_m_s", "wind_max_m_s",
    "ghi_kwh_m2", "dni_kwh_m2", "dhi_kwh_m2", "hdh16_c_h", "cdh26_c_h",
    "drybulb_mean_3d_c", "ghi_3d_kwh_m2", "wind_mean_3d_m_s",
]
TARGET_COLUMNS = ["daily_heating_load_kwh", "daily_cooling_load_kwh", "daily_discomfort_hours"]
TARGET_SHORT = ["heating", "cooling", "discomfort"]
TARGET_UNITS = ["kWh", "kWh", "h"]
ENERGY_IDX = [0, 1]  # heating+cooling are summable kWh; discomfort (hours) is not

SEED = 42
VAL_SIZE = 0.30
MEMBER_SEEDS = [42, 142, 242, 342, 442]
USE_BOOTSTRAP = True


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    hidden_dim: int = 128
    branch_hidden_dims: tuple[int, ...] = (128, 128)
    trunk_hidden_dims: tuple[int, ...] = (128, 128)
    head_hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.02
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    stage1_epochs: int = 400
    stage2_epochs: int = 150
    stage2_learning_rate: float = 1e-4
    stage2_weight_decay: float = 1e-5
    annual_loss_weight: float = 0.1


# --------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(a, b): return float(math.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))


def r2(a, b):
    denom = float(np.sum((a - np.mean(a)) ** 2))
    return float("nan") if denom < 1e-12 else float(1.0 - np.sum((a - b) ** 2) / denom)


def mape(a, b, eps=1e-6):
    mask = np.abs(a) > eps
    return float("nan") if not np.any(mask) else float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100.0)


def wmape(a, b):
    denom = float(np.sum(np.abs(a)))
    return float("nan") if denom < 1e-12 else float(np.sum(np.abs(a - b)) / denom * 100.0)


def spearman_corr(a, b):
    if len(a) < 2:
        return float("nan")
    val = spearmanr(a, b).statistic
    return float(val) if np.isfinite(val) else 0.0


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


# --------------------------------------------------------------------------- data
def load_sequence_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATASET_PATH)
    frame = frame[frame["simulation_status"] == "ok"].copy()
    frame = frame[frame["sample_type"] != "baseline"].copy()
    frame = frame.sort_values(["sample_id", "day_index"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No valid non-baseline samples found.")
    return frame


def split_sample_ids(frame: pd.DataFrame) -> tuple[list[int], list[int]]:
    ids = sorted(frame["sample_id"].unique().tolist())
    train_ids, val_ids = train_test_split(ids, test_size=VAL_SIZE, random_state=SEED, shuffle=True)
    return sorted(train_ids), sorted(val_ids)


def build_sequence_arrays(frame: pd.DataFrame, sample_ids: list[int]) -> dict[str, np.ndarray]:
    subset = frame[frame["sample_id"].isin(sample_ids)]
    static_rows, dynamic_rows, target_rows, ordered = [], [], [], []
    for sample_id, sdf in subset.groupby("sample_id", sort=True):
        sdf = sdf.sort_values("day_index")
        static_rows.append(sdf[STATIC_COLUMNS].iloc[0].to_numpy(dtype=float))
        dynamic_rows.append(sdf[DYNAMIC_COLUMNS].to_numpy(dtype=float))
        target_rows.append(sdf[TARGET_COLUMNS].to_numpy(dtype=float))
        ordered.append(int(sample_id))
    return {
        "sample_ids": np.asarray(ordered, dtype=int),
        "static": np.asarray(static_rows, dtype=float),
        "dynamic": np.asarray(dynamic_rows, dtype=float),
        "target_raw": np.asarray(target_rows, dtype=float),
    }


def fit_scalers(train_arrays):
    static_scaler, dynamic_scaler, target_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    static_scaler.fit(train_arrays["static"])
    dynamic_scaler.fit(train_arrays["dynamic"].reshape(-1, train_arrays["dynamic"].shape[-1]))
    target_scaler.fit(np.log1p(train_arrays["target_raw"].reshape(-1, train_arrays["target_raw"].shape[-1])))
    return static_scaler, dynamic_scaler, target_scaler


def transform_arrays(arrays, ss, ds, ts):
    static_scaled = ss.transform(arrays["static"])
    dyn = ds.transform(arrays["dynamic"].reshape(-1, arrays["dynamic"].shape[-1])).reshape(arrays["dynamic"].shape)
    tgt = ts.transform(np.log1p(arrays["target_raw"].reshape(-1, arrays["target_raw"].shape[-1]))).reshape(arrays["target_raw"].shape)
    return {**arrays, "static_scaled": static_scaled, "dynamic_scaled": dyn, "target_scaled": tgt}


def subset_arrays(arrays, idx):
    return {k: v[idx] for k, v in arrays.items()}


class SequenceDataset(Dataset):
    def __init__(self, a):
        self.static = torch.from_numpy(a["static_scaled"].astype(np.float32))
        self.dynamic = torch.from_numpy(a["dynamic_scaled"].astype(np.float32))
        self.target_scaled = torch.from_numpy(a["target_scaled"].astype(np.float32))
        self.target_raw = torch.from_numpy(a["target_raw"].astype(np.float32))
        self.sample_ids = a["sample_ids"].astype(int)

    def __len__(self): return len(self.sample_ids)

    def __getitem__(self, i):
        return {"static": self.static[i], "dynamic": self.dynamic[i],
                "target_scaled": self.target_scaled[i], "target_raw": self.target_raw[i],
                "sample_id": int(self.sample_ids[i])}


def collate(batch):
    return {
        "static": torch.stack([b["static"] for b in batch]),
        "dynamic": torch.stack([b["dynamic"] for b in batch]),
        "target_scaled": torch.stack([b["target_scaled"] for b in batch]),
        "target_raw": torch.stack([b["target_raw"] for b in batch]),
        "sample_id": torch.tensor([b["sample_id"] for b in batch], dtype=torch.int64),
    }


def make_loader(arrays, batch_size, shuffle):
    return DataLoader(SequenceDataset(arrays), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# --------------------------------------------------------------------------- model
def feature_extractor(input_dim, hidden_dims, dropout):
    layers, dim = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout)]
        dim = h
    return nn.Sequential(*layers), dim


def regression_head(input_dim, hidden_dims, dropout):
    layers, dim = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout)]
        dim = h
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


class HCDeepOnet(nn.Module):
    def __init__(self, static_dim, dynamic_dim, n_targets, cfg: TrainingConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_targets = n_targets
        self.branch_net, branch_out = feature_extractor(static_dim, cfg.branch_hidden_dims, cfg.dropout)
        self.trunk_net, trunk_out = feature_extractor(dynamic_dim, cfg.trunk_hidden_dims, cfg.dropout)
        self.branch_proj = nn.Linear(branch_out, cfg.hidden_dim)
        self.trunk_proj = nn.Linear(trunk_out, cfg.hidden_dim)
        fusion_dim = cfg.hidden_dim * 3
        self.heads = nn.ModuleList([regression_head(fusion_dim, cfg.head_hidden_dims, cfg.dropout) for _ in range(n_targets)])

    def forward(self, static_x, dynamic_x):
        bsz, steps, _ = dynamic_x.shape
        branch = self.branch_proj(self.branch_net(static_x))
        trunk = self.trunk_proj(self.trunk_net(dynamic_x.reshape(bsz * steps, -1))).reshape(bsz, steps, self.hidden_dim)
        branch_t = branch.unsqueeze(1).expand(-1, steps, -1)
        interaction = branch_t * trunk / math.sqrt(self.hidden_dim)
        fused = torch.cat([branch_t, trunk, interaction], dim=-1)
        return torch.cat([head(fused) for head in self.heads], dim=-1)

    def freeze_encoders(self):
        for p in self.branch_net.parameters(): p.requires_grad = False
        for p in self.trunk_net.parameters(): p.requires_grad = False


def target_scaler_tensors(ts, device):
    mean = torch.tensor(ts.mean_, dtype=torch.float32, device=device).view(1, 1, -1)
    scale = torch.tensor(ts.scale_, dtype=torch.float32, device=device).view(1, 1, -1)
    return mean, scale


def to_original(pred_scaled, mean, scale):
    return torch.clamp(torch.expm1(pred_scaled * scale + mean), min=0.0)


# --------------------------------------------------------------------------- train
def train_stage(model, train_loader, val_loader, device, mean, scale, annual_scale,
                lr, weight_decay, epochs, annual_weight, tag):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    best_state, best_val, best_epoch = None, float("inf"), 0
    # Non-shuffling loader over the SAME training data for clean, val-comparable train loss.
    # Must not shuffle: iterating a RandomSampler would consume the global torch RNG and shift
    # every subsequent training epoch's shuffle, diverging the retrained weights from baseline.
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                                   shuffle=False, collate_fn=collate)
    history = []

    def batch_loss(batch):
        """Returns (total, daily, annual) scalars; total == daily + annual_weight*annual."""
        sx = batch["static"].to(device); dx = batch["dynamic"].to(device)
        ts_scaled = batch["target_scaled"].to(device); tr_raw = batch["target_raw"].to(device)
        pred_scaled = model(sx, dx)
        daily = torch.mean((pred_scaled - ts_scaled) ** 2)
        if annual_weight > 0.0:
            pred_annual = to_original(pred_scaled, mean, scale).sum(dim=1)
            true_annual = tr_raw.sum(dim=1)
            annual = torch.mean(((pred_annual - true_annual) / annual_scale) ** 2)
        else:
            annual = torch.zeros((), device=device)
        return daily + annual_weight * annual, daily, annual

    def eval_losses(loader):
        """eval() + no_grad mean (total, daily, annual) over a loader; dropout off => no RNG draw."""
        model.eval()
        tot = dly = ann = 0.0; n = 0
        with torch.no_grad():
            for batch in loader:
                bs = batch["static"].shape[0]
                t, d, a = batch_loss(batch)
                tot += float(t.item()) * bs; dly += float(d.item()) * bs
                ann += float(a.item()) * bs; n += bs
        n = max(1, n)
        return tot / n, dly / n, ann / n

    for epoch in range(1, epochs + 1):
        model.train(True)
        run_tot, run_n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, _, _ = batch_loss(batch)
            loss.backward()
            opt.step()
            bs = batch["static"].shape[0]
            run_tot += float(loss.item()) * bs; run_n += bs
        train_run = run_tot / max(1, run_n)  # train-mode running loss (dropout on); reference column
        # validation (drives scheduler + best selection, unchanged)
        vloss, vdaily, _ = eval_losses(val_loader)
        sched.step(vloss)
        if vloss < best_val:
            best_val, best_epoch = vloss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # clean, val-comparable train loss (eval mode, non-shuffling loader)
        tloss, tdaily, _ = eval_losses(train_eval_loader)
        history.append({"epoch": epoch, "lr": float(opt.param_groups[0]["lr"]),
                        "train_loss_total": tloss, "val_loss_total": vloss,
                        "train_loss_daily": tdaily, "val_loss_daily": vdaily,
                        "train_loss_running": train_run})
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"    [{tag}] epoch {epoch:03d} | train={tloss:.6f} | val={vloss:.6f} | lr={opt.param_groups[0]['lr']:.2e}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_epoch, history


def predict(model, loader, device, mean, scale):
    model.eval()
    ids, truths, preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = to_original(model(batch["static"].to(device), batch["dynamic"].to(device)), mean, scale).cpu().numpy()
            preds.append(pred); truths.append(batch["target_raw"].numpy()); ids.append(batch["sample_id"].numpy())
    ids = np.concatenate(ids); truths = np.concatenate(truths); preds = np.concatenate(preds)
    order = np.argsort(ids)
    return ids[order], truths[order], preds[order]


def annual_scale_tensor(train_raw, device):
    annual = train_raw["target_raw"].sum(axis=1)
    scale = np.maximum(annual.std(axis=0), 1e-6)
    return torch.tensor(scale, dtype=torch.float32, device=device).view(1, -1)


def metric_tables(ids, truth, pred, split):
    """Per-target daily + annual metrics. truth/pred: (N, days, n_targets)."""
    daily_rows, annual_rows = [], []
    truth_annual, pred_annual = truth.sum(axis=1), pred.sum(axis=1)
    for i, short in enumerate(TARGET_SHORT):
        td, pd_ = truth[:, :, i].reshape(-1), pred[:, :, i].reshape(-1)
        daily_rows.append({"split": split, "target": short, "level": "daily",
                           "r2": r2(td, pd_), "rmse": rmse(td, pd_), "mae": mae(td, pd_),
                           "mape_pct": mape(td, pd_), "wmape_pct": wmape(td, pd_)})
        ta, pa = truth_annual[:, i], pred_annual[:, i]
        annual_rows.append({"split": split, "target": short, "level": "annual",
                            "r2": r2(ta, pa), "rmse": rmse(ta, pa), "mae": mae(ta, pa),
                            "mape_pct": mape(ta, pa), "wmape_pct": wmape(ta, pa),
                            "spearman_r": spearman_corr(ta, pa)})
    # energy total (heating+cooling), unit-consistent
    te = truth_annual[:, ENERGY_IDX].sum(axis=1); pe = pred_annual[:, ENERGY_IDX].sum(axis=1)
    annual_rows.append({"split": split, "target": "energy_total", "level": "annual",
                        "r2": r2(te, pe), "rmse": rmse(te, pe), "mae": mae(te, pe),
                        "mape_pct": mape(te, pe), "wmape_pct": wmape(te, pe),
                        "spearman_r": spearman_corr(te, pe)})
    pred_frame = pd.DataFrame({"sample_id": ids})
    for i, short in enumerate(TARGET_SHORT):
        pred_frame[f"true_annual_{short}"] = truth_annual[:, i]
        pred_frame[f"pred_annual_{short}"] = pred_annual[:, i]
    return pd.DataFrame(daily_rows), pd.DataFrame(annual_rows), pred_frame


def train_member(idx, seed, cfg, device, train_raw, train_scaled, val_scaled, mean, scale, annual_scale):
    member_dir = OUTPUT_DIR / f"member_{idx:02d}_seed_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(train_raw["sample_ids"])
    boot = rng.integers(0, n, size=n) if USE_BOOTSTRAP else np.arange(n)
    train_loader = make_loader(subset_arrays(train_scaled, boot), cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_scaled, cfg.batch_size, shuffle=False)

    set_seed(seed)
    model = HCDeepOnet(len(STATIC_COLUMNS), len(DYNAMIC_COLUMNS), len(TARGET_COLUMNS), cfg).to(device)
    print(f"  member {idx:02d} (seed {seed}) stage1 ...")
    model, _, best_ep1, hist1 = train_stage(model, train_loader, val_loader, device, mean, scale, annual_scale,
                                            cfg.learning_rate, cfg.weight_decay, cfg.stage1_epochs, 0.0, f"m{idx:02d}.s1")
    model.freeze_encoders()
    print(f"  member {idx:02d} stage2 (annual fine-tune) ...")
    model, best_val, best_ep2, hist2 = train_stage(model, train_loader, val_loader, device, mean, scale, annual_scale,
                                                   cfg.stage2_learning_rate, cfg.stage2_weight_decay, cfg.stage2_epochs,
                                                   cfg.annual_loss_weight, f"m{idx:02d}.s2")

    # per-epoch loss history (long format), tagged by member/seed/stage; global_epoch = continuous x-axis
    member = f"m{idx:02d}"
    rows = []
    for stage, hist in [("stage1", hist1), ("stage2", hist2)]:
        offset = 0 if stage == "stage1" else cfg.stage1_epochs
        for h in hist:
            rows.append({"member": member, "seed": seed, "stage": stage,
                         "global_epoch": offset + h["epoch"], **h})
    loss_hist = pd.DataFrame(rows, columns=["member", "seed", "stage", "epoch", "global_epoch", "lr",
                                            "train_loss_total", "val_loss_total",
                                            "train_loss_daily", "val_loss_daily", "train_loss_running"])
    loss_hist.to_csv(member_dir / "loss_history.csv", index=False, encoding="utf-8-sig")

    val_ids, val_truth, val_pred = predict(model, val_loader, device, mean, scale)
    torch.save({"model_state_dict": model.state_dict(), "member_seed": seed,
                "config": asdict(cfg), "static_columns": STATIC_COLUMNS,
                "dynamic_columns": DYNAMIC_COLUMNS, "target_columns": TARGET_COLUMNS},
               member_dir / "model_state.pt")
    return {"idx": idx, "seed": seed, "best_val": best_val,
            "best_epoch_stage1": best_ep1, "best_epoch_stage2": best_ep2,
            "loss_history": loss_hist,
            "val_ids": val_ids, "val_truth": val_truth, "val_pred": val_pred}


def main() -> None:
    global DATASET_PATH, SUMMARY_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Train HC-DeepOnet office surrogate ensemble (3 targets).")
    parser.add_argument("--members", type=int, default=len(MEMBER_SEEDS), help="ensemble members (<=5)")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    SUMMARY_PATH = args.summary_path
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = select_device()
    cfg = TrainingConfig(
        stage1_epochs=args.stage1_epochs or TrainingConfig.stage1_epochs,
        stage2_epochs=args.stage2_epochs or TrainingConfig.stage2_epochs,
    )
    print(f"device: {device} | targets: {TARGET_SHORT} | static {len(STATIC_COLUMNS)} | dynamic {len(DYNAMIC_COLUMNS)}")

    frame = load_sequence_frame()
    train_ids, val_ids = split_sample_ids(frame)
    pd.DataFrame({"sample_id": train_ids + val_ids,
                  "split": ["train"] * len(train_ids) + ["val"] * len(val_ids)}).to_csv(
        OUTPUT_DIR / "split_sample_ids.csv", index=False, encoding="utf-8-sig")
    print(f"samples: train {len(train_ids)} | val {len(val_ids)}")

    train_raw = build_sequence_arrays(frame, train_ids)
    val_raw = build_sequence_arrays(frame, val_ids)
    ss, ds, ts = fit_scalers(train_raw)
    for name, obj in [("static_scaler", ss), ("dynamic_scaler", ds), ("target_scaler", ts)]:
        joblib.dump(obj, OUTPUT_DIR / f"{name}.joblib")
    train_scaled = transform_arrays(train_raw, ss, ds, ts)
    val_scaled = transform_arrays(val_raw, ss, ds, ts)
    mean, scale = target_scaler_tensors(ts, device)
    annual_scale = annual_scale_tensor(train_raw, device)

    seeds = MEMBER_SEEDS[: max(1, min(args.members, len(MEMBER_SEEDS)))]
    members = [train_member(i, s, cfg, device, train_raw, train_scaled, val_scaled, mean, scale, annual_scale)
               for i, s in enumerate(seeds, start=1)]

    # ---- per-epoch loss history (plotting-ready) ----
    loss_history = pd.concat([m["loss_history"] for m in members], ignore_index=True)
    loss_history.to_csv(OUTPUT_DIR / "training_loss_history.csv", index=False, encoding="utf-8-sig")
    loss_summary_rows = []
    for m in members:
        lh = m["loss_history"]
        for stage, best_ep in [("stage1", m["best_epoch_stage1"]), ("stage2", m["best_epoch_stage2"])]:
            sh = lh[lh["stage"] == stage]
            loss_summary_rows.append({
                "member": f"m{m['idx']:02d}", "seed": m["seed"], "stage": stage,
                "epochs": int(len(sh)), "best_epoch": int(best_ep),
                "min_val_total": float(sh["val_loss_total"].min()),
                "final_train_total": float(sh["train_loss_total"].iloc[-1]),
                "final_val_total": float(sh["val_loss_total"].iloc[-1]),
            })
    pd.DataFrame(loss_summary_rows).to_csv(OUTPUT_DIR / "loss_history_summary.csv", index=False, encoding="utf-8-sig")
    print(f"loss history: {OUTPUT_DIR / 'training_loss_history.csv'} ({len(loss_history)} rows)")

    # ---- per-member val metrics ----
    member_rows = []
    for m in members:
        _, ann, _ = metric_tables(m["val_ids"], m["val_truth"], m["val_pred"], "val")
        et = ann[ann["target"] == "energy_total"].iloc[0]
        member_rows.append({"member": f"m{m['idx']:02d}", "seed": m["seed"],
                            "val_energy_total_r2": float(et["r2"]), "val_energy_total_rmse": float(et["rmse"]),
                            "val_energy_total_mape_pct": float(et["mape_pct"])})
    pd.DataFrame(member_rows).to_csv(OUTPUT_DIR / "member_validation_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- ensemble mean + sigma_E ----
    ref_ids = members[0]["val_ids"]
    truth = members[0]["val_truth"]
    pred_stack = np.stack([m["val_pred"] for m in members], axis=0)         # (M, N, days, targets)
    ens_mean = pred_stack.mean(axis=0)
    member_annual = pred_stack.sum(axis=2)                                   # (M, N, targets)
    ens_annual_mean = member_annual.mean(axis=0)                            # (N, targets)
    ens_annual_std = member_annual.std(axis=0)                              # (N, targets) = sigma_E
    truth_annual = truth.sum(axis=1)                                        # (N, targets)

    ens_daily_metrics, ens_annual_metrics, _ = metric_tables(ref_ids, truth, ens_mean, "val_ensemble")
    ens_daily_metrics.to_csv(OUTPUT_DIR / "ensemble_daily_metrics.csv", index=False, encoding="utf-8-sig")
    ens_annual_metrics.to_csv(OUTPUT_DIR / "ensemble_annual_metrics.csv", index=False, encoding="utf-8-sig")

    # uncertainty frame + sigma_E vs |error| calibration per target and for energy total
    unc = {"sample_id": ref_ids}
    calib_rows = []
    for i, short in enumerate(TARGET_SHORT):
        abs_err = np.abs(truth_annual[:, i] - ens_annual_mean[:, i])
        unc[f"true_annual_{short}"] = truth_annual[:, i]
        unc[f"pred_annual_{short}"] = ens_annual_mean[:, i]
        unc[f"sigmaE_annual_{short}"] = ens_annual_std[:, i]
        unc[f"abs_err_annual_{short}"] = abs_err
        calib_rows.append({"target": short, "unit": TARGET_UNITS[i],
                           "mean_sigmaE": float(ens_annual_std[:, i].mean()),
                           "mean_abs_err": float(abs_err.mean()),
                           "spearman_sigmaE_vs_err": spearman_corr(ens_annual_std[:, i], abs_err)})
    # energy total
    et_true = truth_annual[:, ENERGY_IDX].sum(axis=1)
    et_pred = ens_annual_mean[:, ENERGY_IDX].sum(axis=1)
    et_sigma = member_annual[:, :, ENERGY_IDX].sum(axis=2).std(axis=0)
    et_err = np.abs(et_true - et_pred)
    unc["true_annual_energy_total"] = et_true
    unc["pred_annual_energy_total"] = et_pred
    unc["sigmaE_annual_energy_total"] = et_sigma
    unc["abs_err_annual_energy_total"] = et_err
    calib_rows.append({"target": "energy_total", "unit": "kWh",
                       "mean_sigmaE": float(et_sigma.mean()), "mean_abs_err": float(et_err.mean()),
                       "spearman_sigmaE_vs_err": spearman_corr(et_sigma, et_err)})
    pd.DataFrame(unc).to_csv(OUTPUT_DIR / "validation_uncertainty_frame.csv", index=False, encoding="utf-8-sig")
    calib = pd.DataFrame(calib_rows)
    calib.to_csv(OUTPUT_DIR / "sigmaE_calibration.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----
    lines = [
        "# 办公 HC-DeepOnet 深集成代理模型",
        "",
        f"- 设备: `{device}` | 集成成员: `{len(members)}` (seeds `{seeds}`, bootstrap `{USE_BOOTSTRAP}`)",
        f"- 训练/验证样本: `{len(train_ids)}` / `{len(val_ids)}`（按样本划分，无逐日泄漏；基线排除）",
        f"- 输入: 10 维设计变量 + 20 维逐日气象 | 目标: {TARGET_SHORT}（{TARGET_UNITS}）",
        f"- 配置: `{asdict(cfg)}`",
        "",
        "## 集成验证精度（逐日）",
        "", dataframe_to_markdown(ens_daily_metrics.round(6)), "",
        "## 集成验证精度（年度）",
        "", dataframe_to_markdown(ens_annual_metrics.round(6)), "",
        "## σ_E 不确定性校准（集成分歧 vs 真实误差，Spearman 越高越能做门控）",
        "", dataframe_to_markdown(calib.round(6)), "",
        "## 单成员验证（energy_total）",
        "", dataframe_to_markdown(pd.DataFrame(member_rows).round(6)), "",
        "## 训练 loss 记录（供绘图）",
        "",
        "- `training_loss_history.csv`：全成员逐 epoch 长表。列：`member, seed, stage, epoch, global_epoch, lr, "
        "train_loss_total, val_loss_total, train_loss_daily, val_loss_daily, train_loss_running`。",
        "- `train_loss_total`/`val_loss_total` 为 eval 模式下、与验证可比的训练/验证 loss；`*_daily` 为逐日 MSE 分量"
        "（跨 stage1/stage2 可比，stage2 的 total 额外含年度一致项）；`train_loss_running` 为训练态(含 dropout)运行均值，仅作参考。",
        "- `global_epoch`：stage1 用本身 epoch，stage2 加 `stage1_epochs` 偏移，便于画连续收敛曲线。",
        "- 每成员另存 `member_*/loss_history.csv`；`loss_history_summary.csv` 汇总每成员每阶段的 `best_epoch` 与末值。",
        f"- 绘图：`python plot_training_loss.py`（输出 PNG 到 `{(OUTPUT_DIR / 'evaluation').name}/`）。",
        "",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps({
        "device": str(device), "member_seeds": seeds, "use_bootstrap": USE_BOOTSTRAP,
        "train_sample_count": len(train_ids), "val_sample_count": len(val_ids),
        "static_columns": STATIC_COLUMNS, "dynamic_columns": DYNAMIC_COLUMNS,
        "target_columns": TARGET_COLUMNS, "config": asdict(cfg),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ENSEMBLE VALIDATION (annual) ===")
    print(ens_annual_metrics.to_string(index=False))
    print("\n=== sigma_E CALIBRATION ===")
    print(calib.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
