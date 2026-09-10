"""Stage 2：联合优化 MoE Actor、Router 与双 Critic。"""

from model.moe_td3 import PROJECT_ROOT
from scripts.train.common import run_stage

# 输入输出。
EXPERIMENT_DIR = PROJECT_ROOT / "checkpoints" / "MoE_TD3" / "experiment_01"
INPUT_CHECKPOINT = EXPERIMENT_DIR / "stage1" / "final.pt"
OUTPUT_DIR = EXPERIMENT_DIR / "stage2"
RESUME_CHECKPOINT = None

# 联合训练：新建本阶段优化器，Actor/encoder/双 Critic 均恢复训练。
TRAINING_SETTINGS = dict(
    training_steps=100_000, batch_size=64, lr_actor=1e-4, lr_critic=1e-3,
    bc_weight_init=1.0, bc_weight_final=0.2, bc_anneal_steps=150_000,
    router_loss_weight=1.0, switch_loss_weight=0.01,
    log_every=100, save_every=10_000,
)
EVAL_EVERY = 1000


def run(*, input_checkpoint=INPUT_CHECKPOINT, output_dir=OUTPUT_DIR,
        resume_checkpoint=RESUME_CHECKPOINT, settings=None, eval_every=EVAL_EVERY):
    return run_stage(2, output_dir, source_checkpoint=input_checkpoint,
                     resume_checkpoint=resume_checkpoint,
                     overrides=TRAINING_SETTINGS if settings is None else settings, eval_every=eval_every)


if __name__ == "__main__":
    run()
