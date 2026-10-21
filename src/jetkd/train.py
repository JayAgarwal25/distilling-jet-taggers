"""One training loop for training from scratch and for distillation.

The two differ only in whether teacher logits are attached. Training counts
iterations and validates every `recipe.validate_every` steps; the checkpoint with
the lowest validation BCE is kept, and training stops early after `es_patience`
validations without improvement. The data order of epoch e is a fixed function of
the seed and e, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import jetkd
from jetkd.data import TopTaggingSplit, collate
from jetkd.distillation import KDConfig, kd_loss
from jetkd.metrics import binary_metrics
from jetkd.models import build
from jetkd.recipes import Recipe, build_optimizer, build_scheduler, recipe_for
from jetkd.teachers import deterministic


@dataclass
class RunConfig:
    model: str
    seed: int = 1001
    teacher: str | None = None
    teacher_seed: int | None = None
    teacher_logits: str | None = None   # train-split logits of the teacher (.npz)
    kd: KDConfig = field(default_factory=KDConfig)
    recipe: Recipe | None = None        # default: the model's own recipe
    data: str = "data/toptagging_full.npz"
    out: str = "runs"
    device: str = "cuda"
    workers: int = 4
    limit: int | None = None            # first rows of each split only, for quick tests

    def __post_init__(self):
        self.recipe = self.recipe or recipe_for(self.model)
        if (self.teacher is None) != (self.teacher_logits is None):
            raise ValueError("a distillation run needs both `teacher` and `teacher_logits`")

    @property
    def id(self) -> str:
        arm = f"kd_{self.teacher}_t{self.teacher_seed}" if self.teacher else "scratch"
        return f"{self.model}__{arm}__s{self.seed}"


def duration(seconds: float) -> str:
    """Seconds as `2h 05m`, `7m` or `12s`."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = round(seconds / 60)
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m}m"


def load_teacher_logits(path: str, labels: torch.Tensor) -> np.ndarray:
    """Teacher logits for the first len(labels) rows, after checking they line up with them."""
    with np.load(path) as f:
        logits, sids, cached = f["logits"], f["sids"], f["labels"]
    n = len(labels)
    if not np.array_equal(sids, np.arange(len(sids))) or len(sids) < n:
        raise ValueError(f"{path}: rows are not the split's rows in order")
    if not np.array_equal(cached[:n], labels.numpy()):
        raise ValueError(f"{path}: labels disagree with the dataset")
    return logits[:n]


class OrderSampler(Sampler):
    def __init__(self, order: torch.Tensor):
        self.order = order

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self) -> int:
        return len(self.order)


class Trainer:
    def __init__(self, cfg: RunConfig):
        self.cfg, self.recipe = cfg, cfg.recipe
        self.out = Path(cfg.out) / cfg.id
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)

        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)   # ParT and MIParT trim sequences with Python's random
        self.model = build(cfg.model).to(self.device)
        self.data = {s: TopTaggingSplit(s, cfg.data, cfg.limit) for s in ("train", "val")}
        self.teacher_logits = (load_teacher_logits(cfg.teacher_logits, self.data["train"].label)
                               if cfg.teacher else None)

        self.batches_per_epoch = -(-len(self.data["train"]) // self.recipe.batch_size)
        self.optimizer = build_optimizer(self.model, self.recipe)
        self.scheduler, self.cadence = build_scheduler(
            self.optimizer, self.recipe, self.batches_per_epoch)
        self.history: list[dict] = []
        self.best = {"val_loss": float("inf"), "step": -1}
        self.start_step = 0
        self.patience = 0
        self.stopped_early = False
        self._resume()

    # Every loader has its own generator, so no loader draws from the global RNG
    # and shifts the dropout masks of later steps.
    def _generator(self, tag: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.cfg.seed * 1_000_003 + tag)

    def _loader(self, split: str, tag: int, order: torch.Tensor | None = None, batch_size=None):
        return DataLoader(
            self.data[split] if split in self.data else TopTaggingSplit(split, self.cfg.data,
                                                                        self.cfg.limit),
            batch_size=batch_size or self.recipe.batch_size,
            sampler=None if order is None else OrderSampler(order),
            num_workers=self.cfg.workers, collate_fn=collate,
            pin_memory=self.device.type == "cuda", generator=self._generator(tag))

    def _batches(self, start_step: int):
        """Training batches from `start_step` on, epoch after epoch."""
        epoch, skip = divmod(start_step, self.batches_per_epoch)
        while True:
            order = torch.randperm(len(self.data["train"]), generator=self._generator(epoch))
            yield from self._loader("train", epoch, order[skip * self.recipe.batch_size:])
            epoch, skip = epoch + 1, 0

    def _step(self, batch) -> dict:
        t = None
        if self.teacher_logits is not None:
            t = torch.from_numpy(self.teacher_logits[batch.sid.numpy()]).to(self.device)
        b = batch.to(self.device)
        loss, parts = kd_loss(self.model(b), b.label, t, self.cfg.kd)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip_grad_norm)
        self.optimizer.step()
        if self.cadence == "step":
            self.scheduler.step()
        soft = parts["soft"].item() if parts["soft"] is not None else 0.0
        return {"loss": loss.item(), "hard": parts["hard"].item(), "soft": soft, "n": len(batch.label)}

    @torch.no_grad()
    def evaluate(self, split: str, loader=None):
        """Metrics, logits, labels and row indices of a split."""
        self.model.eval()
        loader = loader or self._loader(split, -2, batch_size=max(self.recipe.batch_size, 512))
        z, y, sid = [], [], []
        for batch in loader:
            z.append(self.model(batch.to(self.device)).float().cpu())
            y.append(batch.label)
            sid.append(batch.sid)
        z, y, sid = (torch.cat(a).numpy() for a in (z, y, sid))
        return binary_metrics(z, y), z, y, sid

    def fit(self, stop_after: int | None = None) -> dict:
        val_loader = self._loader("val", -1)
        self._write_config()
        self._log_start()
        batches = self._batches(self.start_step)
        last = min(stop_after or self.recipe.iterations, self.recipe.iterations)
        window = {"loss": 0.0, "hard": 0.0, "soft": 0.0, "n": 0}
        t0 = t_start = time.time()
        at = self.start_step
        for step in range(self.start_step, last):
            self.model.train()
            out = self._step(next(batches))
            for k in ("loss", "hard", "soft"):
                window[k] += out[k] * out["n"]
            window["n"] += out["n"]
            at = step + 1

            done = at - self.start_step
            if done in (1, 10, 100, 1000) or at % self.recipe.validate_every == 0:
                spent = time.time() - t_start
                estimate = spent / done * (self.recipe.iterations - self.start_step)
                self.log(f"finished step {at:,} after {duration(spent)}, "
                         f"training time estimate {duration(estimate)}")

            if (step + 1) % self.recipe.validate_every == 0:
                val, *_ = self.evaluate("val", val_loader)
                train = {k: window[k] / window["n"] for k in ("loss", "hard", "soft")}
                window = {"loss": 0.0, "hard": 0.0, "soft": 0.0, "n": 0}
                self.history.append({
                    "step": step + 1, "epoch": round((step + 1) / self.batches_per_epoch, 3),
                    "seconds": round(time.time() - t0, 1),
                    "lr": self.optimizer.param_groups[0]["lr"], "train": train, "val": val})
                t0 = time.time()
                if val["loss"] < self.best["val_loss"]:
                    self.best = {"val_loss": val["loss"], "step": step + 1, "val_auc": val["auc"]}
                    self.patience = 0
                    self._save("best.pt", step + 1)
                else:
                    self.patience += 1
                self._save("last.pt", step + 1)
                (self.out / "history.json").write_text(
                    json.dumps({"best": self.best, "history": self.history}, indent=2))
                self.log(f"validation at step {step + 1:,}: train loss {train['loss']:.5f} "
                         f"val BCE {val['loss']:.5f} val AUC {val['auc']:.5f}, "
                         f"best at step {self.best['step']:,}")
                if self.patience > self.recipe.es_patience:
                    self.stopped_early = True
                    self.log(f"stopped early: no better validation in {self.recipe.es_patience} checks")
                    break

            # weaver steps flat+decay at the start of each epoch, the first one included
            if self.cadence == "epoch" and step % self.batches_per_epoch == 0:
                self.scheduler.step()
        self.log(f"finished training at step {at:,} after {duration(time.time() - t_start)}")
        return self.best

    def _save(self, name: str, step: int) -> None:
        state = {
            "step": step, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "best": self.best, "history": self.history, "patience": self.patience,
            "recipe": self.recipe.as_record(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }
        torch.save(state, self.out / (name + ".tmp"))
        (self.out / (name + ".tmp")).replace(self.out / name)

    def _resume(self) -> None:
        path = self.out / "last.pt"
        if not path.is_file():
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["recipe"] != self.recipe.as_record():
            raise RuntimeError(f"{self.out} was trained with another recipe; move it aside")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.best, self.history = state["best"], state["history"]
        self.patience, self.start_step = state["patience"], state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        self.log(f"resumed at step {self.start_step:,}")

    def log(self, message: str) -> None:
        """Print a line and append it to runs/<id>/train.log."""
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.out / "train.log").open("a") as f:
            f.write(line + "\n")

    def _log_start(self) -> None:
        r, cfg = self.recipe, self.cfg
        n = sum(p.numel() for p in self.model.parameters())
        self.log(f"run {cfg.id}: {cfg.model} with {n:,} parameters on {self.device}, writing to {self.out}")
        self.log(f"{len(self.data['train']):,} training and {len(self.data['val']):,} validation jets, "
                 f"{self.batches_per_epoch:,} steps per epoch")
        self.log(f"recipe {r.name}: {r.optimizer}, lr {r.lr:g}, weight decay {r.weight_decay:g}, "
                 f"batch {r.batch_size}, {r.scheduler} schedule")
        self.log(f"distilled from {cfg.teacher} run {cfg.teacher_seed} ({cfg.teacher_logits}), "
                 f"alpha {cfg.kd.alpha:g}, temperature {cfg.kd.temperature:g}"
                 if cfg.teacher else "trained from scratch")
        self.log(f"starting to train for {r.iterations:,} steps = {r.iterations / self.batches_per_epoch:.1f} "
                 f"epochs, validating every {r.validate_every:,} steps, early stopping after "
                 f"{r.es_patience} validations without improvement")

    def _write_config(self) -> None:
        cfg = asdict(self.cfg)
        cfg["recipe"] = self.recipe.as_record()
        record = {"run": self.cfg.id, "config": cfg,
                  "n_parameters": sum(p.numel() for p in self.model.parameters()),
                  "batches_per_epoch": self.batches_per_epoch,
                  "versions": {"jetkd": jetkd.__version__, "torch": torch.__version__}}
        (self.out / "config.json").write_text(json.dumps(record, indent=2))


def run(cfg: RunConfig, test: bool = True) -> dict:
    """Train, restore the selected checkpoint, then evaluate and save predictions."""
    trainer = Trainer(cfg)
    best = trainer.fit()
    trainer.log(f"best checkpoint at step {best['step']:,}, val BCE {best['val_loss']:.5f}")
    state = torch.load(trainer.out / "best.pt", map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(state["model"])
    results = {"best": best, "stopped_early": trainer.stopped_early,
               "recipe": trainer.recipe.as_record()}
    for split in ("val", "test") if test else ("val",):
        trainer.log(f"evaluating the best checkpoint on the {split} split")
        with deterministic():
            metrics, z, y, sid = trainer.evaluate(split)
        results[split] = metrics
        np.savez_compressed(trainer.out / f"predictions_{split}.npz", logits=z, labels=y, sids=sid)
        trainer.log(f"{split}: AUC {metrics['auc']:.5f} rej50 {metrics['rej50']:.1f} "
                    f"rej30 {metrics['rej30']:.1f}")
    (trainer.out / "results.json").write_text(json.dumps(results, indent=2))
    trainer.log(f"done: {trainer.out / 'results.json'}")
    return results
