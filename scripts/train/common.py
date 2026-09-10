"""三个训练脚本共用的数据、离线评估、日志和模型存档函数。"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from model.moe_td3 import (
    Args, OfflineMoETD3BC, PHASE_NAMES, PHASE_MAPPING_VERSION, PHASE_TRANSITION_VERSION,
    PreparedOfflineData, SequenceReplayBuffer, _window_end_indices, device,
    fill_buffer_from_episodes, normalize_episodes, prepare_offline_data,
    read_offline_episodes, set_seed,
)


def load_checkpoint(path):
    """只读取用户指定的本地训练存档，不自动挑选最新文件。"""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if (payload.get("format_version") != 1
            or payload.get("phase_mapping_version") != PHASE_MAPPING_VERSION
            or payload.get("phase_transition_version") != PHASE_TRANSITION_VERSION):
        raise ValueError("训练存档格式或阶段版本不匹配")
    return payload


def load_data(args, saved=None):
    """后续阶段读取存档中的划分和统计，禁止重新计算验证集相关统计。"""
    path = Path(args.data_path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    fingerprint = digest.hexdigest()
    if saved is not None and fingerprint != saved["sha256"]:
        raise ValueError("数据文件已变化，不能复用原 episode 索引；请使用原数据或新建实验")
    episodes = read_offline_episodes(str(path))
    if saved is None:
        prepared = prepare_offline_data(episodes, args)
    else:
        mean = np.asarray(saved["mean"], dtype=np.float32)
        std = np.asarray(saved["std"], dtype=np.float32)
        if mean.shape != (args.state_dim,) or std.shape != mean.shape or np.any(std <= 0):
            raise ValueError("存档中的归一化统计维度或标准差不正确")
        prepared = PreparedOfflineData(
            normalize_episodes([episodes[i] for i in saved["train_indices"]], mean, std),
            normalize_episodes([episodes[i] for i in saved["validation_indices"]], mean, std),
            mean, std, saved["train_indices"], saved["validation_indices"],
            np.asarray(saved["train_phase_counts"], dtype=np.int64),
            np.asarray(saved["validation_phase_counts"], dtype=np.int64), args.seq_len,
        )
    info = {
        "path": str(path), "sha256": fingerprint,
        "mean": prepared.mean.tolist(), "std": prepared.std.tolist(),
        "train_indices": prepared.train_indices, "validation_indices": prepared.validation_indices,
        "train_phase_counts": prepared.train_phase_counts.tolist(),
        "validation_phase_counts": prepared.validation_phase_counts.tolist(),
    }
    if not prepared.train_phase_counts.sum() or not prepared.validation_phase_counts.sum():
        raise ValueError("训练集或验证集没有完整历史窗口，请增加回合长度或调整划分")
    print(f"训练/验证回合: {len(prepared.train_episodes)}/{len(prepared.validation_episodes)}")
    print(f"阶段顺序: {', '.join(PHASE_NAMES)}")
    print(f"训练窗口分布: {info['train_phase_counts']}；验证窗口分布: {info['validation_phase_counts']}")
    missing = [name for name, count in zip(PHASE_NAMES, prepared.train_phase_counts) if count == 0]
    if missing:
        print(f"提示：{', '.join(missing)} 没有对应训练窗口，仅报告缺样，不中断训练")
    return prepared, info


def validation_buffer(episodes, args, count):
    holder = SimpleNamespace(state_dim=args.state_dim, action_dim=args.action_dim,
                             buffer=SequenceReplayBuffer(int(count), args.state_dim, args.action_dim, args.seq_len))
    fill_buffer_from_episodes(holder, episodes)
    return holder.buffer


def action_metrics(prediction, target):
    difference = np.asarray(prediction) - np.asarray(target)
    return {"mse": float(np.mean(difference ** 2)),
            "mse_xyz": np.mean(difference ** 2, axis=0).tolist(),
            "mae_xyz": np.mean(np.abs(difference), axis=0).tolist(),
            "z_sign_accuracy": float(np.mean(np.sign(prediction[:, 2]) == np.sign(target[:, 2])))}


@torch.no_grad()
def evaluate(agent, buffer, episodes):
    """批量验证使用真实上一阶段；顺序验证只回传模型自身阶段，状态仍来自离线数据。"""
    was_training = agent.actor.training
    agent.actor.eval()
    soft, hard, expert, selected = [], [], [], []
    try:
        for start in range(0, len(buffer), agent.args.batch_size):
            end = min(start + agent.args.batch_size, len(buffer))
            state = torch.as_tensor(buffer.s[start:end], device=device)
            previous = torch.as_tensor(buffer.phase_previous[start:end, 0], device=device)
            out = agent.actor(state, previous)
            soft.append(out["action"].cpu().numpy())
            if not agent.single_head:
                rows = torch.arange(end - start, device=device)
                phase = torch.as_tensor(buffer.phase[start:end, 0], device=device)
                hard.append(out["all_actions"][rows, out["selected_phase"]].cpu().numpy())
                expert.append(out["all_actions"][rows, phase].cpu().numpy())
                selected.append(out["selected_phase"].cpu().numpy())
        targets = buffer.a[:len(buffer), -1]
        metrics = {"samples": len(buffer), "soft": action_metrics(np.concatenate(soft), targets)}
        if agent.single_head:
            metrics["action_mse"] = metrics["soft"]["mse"]
            return metrics
        truth = buffer.phase[:len(buffer), 0]
        choices = np.concatenate(selected)
        confusion = np.zeros((len(PHASE_NAMES), len(PHASE_NAMES)), dtype=np.int64)
        np.add.at(confusion, (truth, choices), 1)
        expert_error = np.mean((np.concatenate(expert) - targets) ** 2, axis=1)
        metrics.update({
            "hard": action_metrics(np.concatenate(hard), targets),
            "router_accuracy": float(np.mean(choices == truth)),
            "confusion_matrix": confusion.tolist(),
            "expert_bc_mse": {name: float(expert_error[truth == i].mean()) if np.any(truth == i) else None
                              for i, name in enumerate(PHASE_NAMES)},
        })
        rollout_soft, rollout_hard, rollout_targets, rollout_truth, rollout_choices = [], [], [], [], []
        for episode in episodes:
            ends = set(_window_end_indices(np.asarray([step["done"] for step in episode]), agent.seq_len))
            if not ends:
                continue
            previous = 0
            states = np.stack([step["observation"] for step in episode])
            for index, step in enumerate(episode):
                history = states[max(0, index + 1 - agent.seq_len):index + 1]
                if len(history) < agent.seq_len:
                    history = np.concatenate([np.repeat(history[:1], agent.seq_len - len(history), axis=0), history])
                out = agent.actor(torch.as_tensor(history[None], device=device),
                                  torch.tensor([previous], device=device))
                previous = int(out["selected_phase"].item())
                if index in ends:
                    rollout_soft.append(out["action"][0].cpu().numpy())
                    rollout_hard.append(out["all_actions"][0, previous].cpu().numpy())
                    rollout_targets.append(step["action"])
                    rollout_truth.append(step["phase"])
                    rollout_choices.append(previous)
                if step["done"]:
                    previous = 0
        rollout_targets = np.asarray(rollout_targets)
        metrics["rollout_soft"] = action_metrics(np.asarray(rollout_soft), rollout_targets)
        metrics["rollout_hard"] = action_metrics(np.asarray(rollout_hard), rollout_targets)
        metrics["rollout_router_accuracy"] = float(np.mean(np.asarray(rollout_choices) == rollout_truth))
        return metrics
    finally:
        agent.actor.train(was_training)


def save_checkpoint(path, agent, stage, data_info, best_score):
    """同时保存目标网络、优化器和随机状态，支持显式恢复同阶段训练。"""
    payload = {
        "format_version": 1, "stage": stage, "args": asdict(agent.args),
        "phase_mapping_version": PHASE_MAPPING_VERSION, "phase_transition_version": PHASE_TRANSITION_VERSION,
        "step": agent.train_step, "best_score": best_score, "last_actor_info": agent.last_actor_info,
        "data": data_info,
        "random_state": random.getstate(), "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    for name in ("actor", "critic1", "critic2", "actor_target", "critic1_target", "critic2_target",
                 "actor_optimizer", "critic1_optimizer", "critic2_optimizer"):
        payload[name] = getattr(agent, name).state_dict()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_training(agent, payload):
    for name in ("actor", "critic1", "critic2", "actor_target", "critic1_target", "critic2_target",
                 "actor_optimizer", "critic1_optimizer", "critic2_optimizer"):
        getattr(agent, name).load_state_dict(payload[name])
    agent.train_step = payload["step"]
    agent.last_actor_info = payload["last_actor_info"]
    random.setstate(payload["random_state"])
    np.random.set_state(payload["numpy_state"])
    torch.set_rng_state(payload["torch_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_state"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_state"]])


def append_log(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run_stage(stage, output_dir, *, args=None, source_checkpoint=None, resume_checkpoint=None,
              overrides=None, eval_every=1000):
    """新阶段从指定上游存档初始化；续训只延长总步数，沿用原阶段超参数。"""
    output_dir = Path(output_dir).resolve()
    overrides = overrides or {}
    source = load_checkpoint(resume_checkpoint or source_checkpoint) if (resume_checkpoint or source_checkpoint) else None
    if resume_checkpoint:
        if source["stage"] != stage or Path(resume_checkpoint).resolve().parent != output_dir:
            raise ValueError("续训文件必须属于当前阶段和当前输出目录")
        total_steps = overrides.get("training_steps", args.training_steps if args else source["args"]["training_steps"])
        args = replace(Args(**source["args"]), training_steps=total_steps)
        print("续训沿用存档超参数，仅更新目标总训练步数")
    elif stage == 0:
        if source is not None:
            raise ValueError("Stage 0 从新模型开始；恢复训练请指定 resume_checkpoint")
        args = replace(args or Args(), **overrides)
    else:
        if source is None or source["stage"] != stage - 1:
            raise ValueError(f"Stage {stage} 需要明确指定 Stage {stage - 1} 存档")
        args = replace(Args(**source["args"]), **overrides)
    if stage not in (0, 1, 2) or min(args.training_steps, args.log_every, args.save_every, eval_every, args.batch_size) <= 0:
        raise ValueError("阶段必须为 0/1/2，训练步数、间隔和 batch_size 必须为正数")
    args.validate()
    if not resume_checkpoint and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录已有内容，请换实验目录或显式续训: {output_dir}")
    set_seed(args.seed)
    prepared, data_info = load_data(args, source["data"] if source else None)
    # 使用实际窗口数分配回放池，避免固定大容量浪费或环形覆盖掉早期回合。
    args = replace(args, capacity=int(prepared.train_phase_counts.sum()), ckpt_dir=str(output_dir))
    agent = OfflineMoETD3BC(args, single_head=stage == 0)
    fill_buffer_from_episodes(agent, prepared.train_episodes)
    validation = validation_buffer(prepared.validation_episodes, args, prepared.validation_phase_counts.sum())
    if source and not resume_checkpoint:
        agent.critic1.load_state_dict(source["critic1"])
        agent.critic2.load_state_dict(source["critic2"])
        if stage == 1:
            agent.initialize_from_single_head(source["actor"])
        else:
            agent.actor.load_state_dict(source["actor"])
        agent.sync_targets()
    if stage == 1:
        agent.configure_pretraining(prepared.train_phase_counts)
    if resume_checkpoint:
        restore_training(agent, source)
        if args.training_steps <= agent.train_step:
            raise ValueError("续训目标总步数必须大于存档中的已完成步数")
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == 0 and not resume_checkpoint:
        shared = output_dir.parent / "shared"
        if shared.exists() and any(shared.iterdir()):
            raise FileExistsError(f"共用数据目录已有内容，请为新实验更换目录: {shared}")
        prepared.save(str(shared))
        (shared / "source.json").write_text(json.dumps(data_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(args), ensure_ascii=False, indent=2), encoding="utf-8")
    best_score = source["best_score"] if resume_checkpoint else float("inf")
    while agent.train_step < args.training_steps:
        info = agent.pretrain_one_step() if stage == 1 else agent.train_one_step()
        step = agent.train_step
        if not all(np.isfinite(value) for value in info.values()):
            raise FloatingPointError(f"Stage {stage} 第 {step} 步出现非有限训练指标")
        if step == 1 or step % args.log_every == 0 or step == args.training_steps:
            append_log(output_dir / "train.jsonl", {**info, "step": step})
            print(f"Stage {stage} | {step}/{args.training_steps} | Actor={info['actor_loss']:.5f} | BC={info['bc_loss']:.5f}")
        if step % eval_every == 0 or step == args.training_steps:
            metrics = evaluate(agent, validation, prepared.validation_episodes)
            append_log(output_dir / "validation.jsonl", {"step": step, **metrics})
            score = metrics["action_mse"] if stage == 0 else metrics["hard" if stage == 1 else "rollout_hard"]["mse"]
            print(f"验证动作 MSE: {score:.6f}")
            if score < best_score:
                best_score = score
                save_checkpoint(output_dir / "best.pt", agent, stage, data_info, best_score)
        if step % args.save_every == 0:
            save_checkpoint(output_dir / f"step_{step}.pt", agent, stage, data_info, best_score)
    final_path = output_dir / "final.pt"
    save_checkpoint(final_path, agent, stage, data_info, best_score)
    print(f"Stage {stage} 完成: {final_path}")
    return final_path
