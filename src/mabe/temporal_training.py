"""Leakage-safe Phase 9 training with early stopping, EMA, and fixed-fold validation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor

from mabe.phase9_config import Phase9Config, TemporalArchitectureConfig
from mabe.temporal_augmentation import TemporalAugmenter
from mabe.temporal_data import (
    ActionRichWindowSampler,
    MaterializedWindow,
    Route,
    WindowRef,
    materialize_window,
)
from mabe.temporal_model import (
    ExponentialMovingAverage,
    TemporalBehaviorModel,
    combined_temporal_loss,
    trainable_parameters,
)


@dataclass(frozen=True, slots=True)
class FoldTrainingResult:
    held_out_fold: int
    checkpoint_path: Path
    best_epoch: int
    best_validation_loss: float
    validation_average_precision: float
    training_seconds: float
    device: str
    parameter_count: int
    peak_memory_mb: float = 0.0


def resolve_temporal_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_temporal_model(
    *,
    config: Phase9Config,
    lab_count: int,
    solo_actions: int,
    pair_actions: int,
    solo_input_features: int | None = None,
    pair_input_features: int | None = None,
) -> TemporalBehaviorModel:
    landmark_count = len(config.data.landmarks)
    solo_features = 3 * landmark_count + 8
    pair_features = 2 * solo_features + 8
    return TemporalBehaviorModel(
        solo_input_features=solo_input_features or solo_features,
        pair_input_features=pair_input_features or pair_features,
        solo_actions=solo_actions,
        pair_actions=pair_actions,
        lab_count=lab_count,
        metadata_features=3,
        config=config.architecture,
    )


def _materialize(
    refs: list[WindowRef],
    sampler: ActionRichWindowSampler,
    *,
    config: Phase9Config,
    augmenter: TemporalAugmenter | None,
) -> list[MaterializedWindow]:
    output: list[MaterializedWindow] = []
    for ref in refs:
        artifact = sampler.artifacts[(ref.lab_id, ref.video_id)]
        raw = artifact.temporal_window(ref.center_index, config.data.window_frames)
        selected = ref
        if augmenter is not None:
            raw, agent, target = augmenter(raw, ref.agent_index, ref.target_index)
            selected = replace(ref, agent_index=agent, target_index=target)
        output.append(
            materialize_window(
                artifact,
                selected,
                length=config.data.window_frames,
                window=raw,
            )
        )
    return output


def _batch(values: list[MaterializedWindow], device: torch.device) -> tuple[Tensor, ...]:
    features = torch.from_numpy(np.stack([value.features for value in values])).to(device)
    targets = torch.from_numpy(np.stack([value.targets for value in values])).to(device)
    loss_mask = torch.from_numpy(np.stack([value.loss_mask for value in values])).to(device)
    metadata = torch.from_numpy(np.stack([value.metadata for value in values])).to(device)
    labs = torch.tensor([value.lab_index for value in values], dtype=torch.long, device=device)
    action_validity = loss_mask.amax(dim=1).bool()
    return features, targets, loss_mask, metadata, labs, action_validity


def _same_lab_partner_indices(labs: Tensor, generator: np.random.Generator) -> list[int]:
    lab_values = [int(value) for value in labs.detach().cpu().tolist()]
    partners: list[int] = []
    for index, lab in enumerate(lab_values):
        choices = [candidate for candidate, value in enumerate(lab_values) if value == lab and candidate != index]
        partners.append(
            int(choices[int(generator.integers(0, len(choices)))]) if choices else index
        )
    return partners


def _augment_materialized_batch(
    features: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
    labs: Tensor,
    *,
    config: Phase9Config,
    generator: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply same-lab MixUp/CutMix without inventing cross-lab supervision."""
    augmentation = config.augmentation
    if features.shape[0] < 2 or (
        augmentation.mixup_probability <= 0
        and augmentation.cutmix_probability <= 0
    ):
        return features, targets, loss_mask, loss_mask.amax(dim=1).bool()
    partners = _same_lab_partner_indices(labs, generator)
    source_features = features.clone()
    source_targets = targets.clone()
    source_mask = loss_mask.clone()
    for index, partner in enumerate(partners):
        if partner == index:
            continue
        if generator.random() < augmentation.mixup_probability:
            fraction = float(
                generator.beta(augmentation.mixup_alpha, augmentation.mixup_alpha)
            )
            features[index] = (
                fraction * source_features[index]
                + (1 - fraction) * source_features[partner]
            )
            targets[index] = (
                fraction * source_targets[index]
                + (1 - fraction) * source_targets[partner]
            )
            loss_mask[index] = torch.minimum(
                source_mask[index], source_mask[partner]
            )
        if generator.random() < augmentation.cutmix_probability:
            length = features.shape[1]
            fraction = float(
                generator.uniform(
                    augmentation.cutmix_min_fraction,
                    augmentation.cutmix_max_fraction,
                )
            )
            width = max(1, round(length * fraction))
            start = int(generator.integers(0, max(1, length - width + 1)))
            stop = min(length, start + width)
            features[index, start:stop] = source_features[partner, start:stop]
            targets[index, start:stop] = source_targets[partner, start:stop]
            loss_mask[index, start:stop] = source_mask[partner, start:stop]
    return features, targets, loss_mask, loss_mask.amax(dim=1).bool()


def _lab_loss_weights(
    sampler: ActionRichWindowSampler,
    *,
    held_out_fold: int,
    power: float,
) -> dict[int, float]:
    counts: dict[int, int] = {}
    for artifact in sampler.artifacts.values():
        if artifact.fold == held_out_fold:
            continue
        lab_index = int(artifact.metadata["lab_index"])
        counts[lab_index] = counts.get(lab_index, 0) + 1
    if not counts:
        raise ValueError("cannot compute lab weights without training artifacts")
    raw = {lab: count ** (-power) for lab, count in counts.items()}
    normalizer = sum(raw.values()) / len(raw)
    return {lab: value / normalizer for lab, value in raw.items()}


def _sample_weights(
    labs: Tensor, lab_weights: dict[int, float], *, device: torch.device
) -> Tensor:
    return torch.tensor(
        [lab_weights.get(int(value), 1.0) for value in labs.cpu().tolist()],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def _validate_route(
    model: TemporalBehaviorModel,
    refs: list[WindowRef],
    sampler: ActionRichWindowSampler,
    *,
    route: Route,
    config: Phase9Config,
    device: torch.device,
    lab_weights: dict[int, float],
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    targets_all: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
    probability_all: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
    batch_size = config.training.batch_size
    for start in range(0, len(refs), batch_size):
        values = _materialize(
            refs[start : start + batch_size], sampler, config=config, augmenter=None
        )
        features, targets, loss_mask, metadata, labs, validity = _batch(values, device)
        logits = model(route, features, labs, metadata, validity)
        loss = combined_temporal_loss(
            logits,
            targets,
            loss_mask,
            positive_weight=config.training.positive_class_weight,
            soft_f1_weight=config.training.soft_f1_weight,
            focal_gamma=config.training.focal_gamma,
            sample_weights=_sample_weights(labs, lab_weights, device=device),
        )
        losses.append(float(loss.cpu()))
        active = loss_mask.bool()
        targets_all.append(targets[active].float().cpu().numpy())
        probability_all.append(torch.sigmoid(logits)[active].float().cpu().numpy())
    target = np.concatenate(targets_all) if targets_all else np.zeros(0, dtype=np.float32)
    probability = (
        np.concatenate(probability_all) if probability_all else np.zeros(0, dtype=np.float32)
    )
    if target.size == 0 or np.unique(target).size < 2:
        average_precision = 0.0
    else:
        average_precision = float(average_precision_score(target, probability))
    return float(np.mean(losses)) if losses else math.inf, average_precision


def train_temporal_fold(
    *,
    held_out_fold: int,
    manifest: pl.DataFrame,
    cache_root: Path,
    hard_negatives: pl.DataFrame,
    output_root: Path,
    config: Phase9Config,
    configuration_sha256: str,
) -> FoldTrainingResult:
    started = time.perf_counter()
    torch.manual_seed(config.training.seed + held_out_fold)
    np.random.seed(config.training.seed + held_out_fold)
    device = resolve_temporal_device(config.execution.device)
    sampler = ActionRichWindowSampler(
        manifest, cache_root=cache_root, config=config, hard_negatives=hard_negatives
    )
    lab_weights = _lab_loss_weights(
        sampler,
        held_out_fold=held_out_fold,
        power=config.training.lab_loss_weight_power,
    )
    first = next(iter(sampler.artifacts.values()))
    solo_input_features = len(first.metadata["solo_feature_names"])
    pair_input_features = len(first.metadata["pair_feature_names"])
    solo_actions = int(first.metadata["solo_action_count"])
    pair_actions = int(first.metadata["pair_action_count"])
    lab_count = manifest["lab_id"].n_unique()
    model = build_temporal_model(
        config=config,
        lab_count=lab_count,
        solo_actions=solo_actions,
        pair_actions=pair_actions,
        solo_input_features=solo_input_features,
        pair_input_features=pair_input_features,
    ).to(device)
    ema = ExponentialMovingAverage(model, config.training.ema_decay)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = config.training.epochs * config.training.steps_per_epoch

    def schedule(step: int) -> float:
        if config.training.warmup_steps and step < config.training.warmup_steps:
            return (step + 1) / config.training.warmup_steps
        progress = (step - config.training.warmup_steps) / max(
            1, total_steps - config.training.warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp_enabled = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    routes: tuple[Route, ...] = ("solo", "pair")
    validation = {
        route: sampler.plan(
            held_out_fold=held_out_fold,
            route=route,
            count=config.training.validation_windows_per_route,
            seed=config.training.seed + 100_000 + held_out_fold,
            validation=True,
        )
        for route in routes
    }
    checkpoint_path = output_root / f"fold_{held_out_fold}" / "model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_ap = -math.inf
    best_epoch = -1
    stale = 0
    for epoch in range(config.training.epochs):
        model.train()
        per_route_batches = math.ceil(config.training.steps_per_epoch / 2)
        plans = {
            route: sampler.plan(
                held_out_fold=held_out_fold,
                route=route,
                count=per_route_batches * config.training.batch_size,
                seed=config.training.seed + held_out_fold * 1000 + epoch * 10 + index,
            )
            for index, route in enumerate(routes)
        }
        offsets = {"solo": 0, "pair": 0}
        augmenter = TemporalAugmenter(
            config.augmentation, config.training.seed + held_out_fold * 1000 + epoch
        )
        batch_generator = np.random.default_rng(
            config.training.seed + held_out_fold * 100_000 + epoch
        )
        epoch_losses: list[float] = []
        for step in range(config.training.steps_per_epoch):
            route: Route = "solo" if step % 2 == 0 else "pair"
            offset = offsets[route]
            refs = plans[route][offset : offset + config.training.batch_size]
            offsets[route] += config.training.batch_size
            values = _materialize(refs, sampler, config=config, augmenter=augmenter)
            features, targets, loss_mask, metadata, labs, validity = _batch(values, device)
            features, targets, loss_mask, validity = _augment_materialized_batch(
                features,
                targets,
                loss_mask,
                labs,
                config=config,
                generator=batch_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(route, features, labs, metadata, validity)
                loss = combined_temporal_loss(
                    logits,
                    targets,
                    loss_mask,
                    positive_weight=config.training.positive_class_weight,
                    soft_f1_weight=config.training.soft_f1_weight,
                    focal_gamma=config.training.focal_gamma,
                    sample_weights=_sample_weights(labs, lab_weights, device=device),
                )
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before_step
            if optimizer_stepped:
                scheduler.step()
                ema.update(model)
            epoch_losses.append(float(loss.detach().cpu()))
        validation_values = [
            _validate_route(
                ema.shadow,
                validation[route],
                sampler,
                route=route,
                config=config,
                device=device,
                lab_weights=lab_weights,
            )
            for route in routes
        ]
        validation_loss = float(np.mean([value[0] for value in validation_values]))
        validation_ap = float(np.mean([value[1] for value in validation_values]))
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(epoch_losses)),
                "validation_loss": validation_loss,
                "validation_average_precision": validation_ap,
            }
        )
        improved = (
            validation_ap > best_ap
            if config.training.early_stopping_metric == "average_precision"
            else validation_loss < best_loss
        )
        if improved:
            best_loss = validation_loss
            best_ap = validation_ap
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema_version": 1,
                    "temporal_version": config.temporal_version,
                    "configuration_sha256": configuration_sha256,
                    "held_out_fold": held_out_fold,
                    "state_dict": ema.shadow.state_dict(),
                    "solo_input_features": solo_input_features,
                    "pair_input_features": pair_input_features,
                    "solo_actions": solo_actions,
                    "pair_actions": pair_actions,
                    "lab_count": lab_count,
                    "metadata_features": 3,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "validation_average_precision": best_ap,
                    "early_stopping_metric": config.training.early_stopping_metric,
                    "architecture": config.architecture.model_dump(mode="json"),
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= config.training.early_stopping_patience:
                break
    history_path = checkpoint_path.parent / "learning_curve.parquet"
    pl.DataFrame(history).write_parquet(history_path, compression="zstd", statistics=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    memory = psutil.Process().memory_info()
    peak_memory_mb = float(getattr(memory, "peak_wset", memory.rss)) / (1024 * 1024)
    metadata_path = checkpoint_path.parent / "model_card.json"
    result = FoldTrainingResult(
        held_out_fold=held_out_fold,
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        validation_average_precision=best_ap,
        training_seconds=time.perf_counter() - started,
        device=str(device),
        parameter_count=parameter_count,
        peak_memory_mb=peak_memory_mb,
    )
    metadata_path.write_text(
        json.dumps(
            asdict(result)
            | {
                "checkpoint_path": checkpoint_path.as_posix(),
                "configuration_sha256": configuration_sha256,
                "architecture": config.architecture.model_dump(mode="json"),
                "early_stopping_metric": config.training.early_stopping_metric,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def load_temporal_checkpoint(
    path: Path, *, config: Phase9Config, device: torch.device
) -> TemporalBehaviorModel:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("temporal_version") != config.temporal_version:
        raise ValueError("checkpoint temporal version does not match the active configuration")
    checkpoint_architecture = TemporalArchitectureConfig.model_validate(
        payload.get("architecture")
    )
    if checkpoint_architecture != config.architecture:
        raise ValueError("checkpoint architecture does not match configs/temporal.yaml")
    model = build_temporal_model(
        config=config,
        lab_count=int(payload["lab_count"]),
        solo_actions=int(payload["solo_actions"]),
        pair_actions=int(payload["pair_actions"]),
        solo_input_features=int(payload["solo_input_features"]),
        pair_input_features=int(payload["pair_input_features"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
