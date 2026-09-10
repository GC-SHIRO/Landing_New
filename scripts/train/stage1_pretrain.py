"""Stage 1：从单头模型初始化五专家，监督训练 Router 和阶段动作。"""

from model.moe_td3 import PROJECT_ROOT
from scripts.train.common import run_stage

# 输入输出：显式选定上游存档，不自动挑选最新模型。
EXPERIMENT_DIR = PROJECT_ROOT / "checkpoints" / "MoE_TD3" / "experiment_01"
INPUT_CHECKPOINT = EXPERIMENT_DIR / "stage0" / "final.pt"
OUTPUT_DIR = EXPERIMENT_DIR / "stage1"
RESUME_CHECKPOINT = None

# 预训练：冻结 encoder 时只更新 Router 和动作头，Critic 始终不更新。
TRAINING_SETTINGS = dict(
    training_steps=20_000, batch_size=64, lr_actor=1e-4,
    pretrain_freeze_encoder=True, pretrain_encoder_lr_scale=0.1,
    pretrain_bc_weight=1.0, router_loss_weight=1.0,
    log_every=100, save_every=5000,
)
EVAL_EVERY = 1000


def run(*, input_checkpoint=INPUT_CHECKPOINT, output_dir=OUTPUT_DIR,
        resume_checkpoint=RESUME_CHECKPOINT, settings=None, eval_every=EVAL_EVERY):
    return run_stage(1, output_dir, source_checkpoint=input_checkpoint,
                     resume_checkpoint=resume_checkpoint,
                     overrides=TRAINING_SETTINGS if settings is None else settings, eval_every=eval_every)


if __name__ == "__main__":
    run()
