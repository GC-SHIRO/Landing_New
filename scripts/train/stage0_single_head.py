"""Stage 0：训练单头 Transformer TD3-BC 基线。"""

from model.moe_td3 import Args, PROJECT_ROOT
from scripts.train.common import run_stage

# 输入输出：新实验修改 experiment_01，三个脚本使用同一实验目录。
EXPERIMENT_DIR = PROJECT_ROOT / "checkpoints" / "MoE_TD3" / "experiment_01"
OUTPUT_DIR = EXPERIMENT_DIR / "stage0"
RESUME_CHECKPOINT = None

# 数据与训练：网络结构默认值集中在模型 Args 中，需要时在此覆盖。
TRAINING_ARGS = Args(
    data_path=str(PROJECT_ROOT / "data" / "expert_global" / "global_expert.jsonl"),
    training_steps=100_000, batch_size=64, lr_actor=1e-4, lr_critic=1e-3,
    validation_fraction=0.2, seed=1, log_every=100, save_every=10_000,
)
EVAL_EVERY = 1000


def run(*, args=None, output_dir=OUTPUT_DIR, resume_checkpoint=RESUME_CHECKPOINT, eval_every=EVAL_EVERY):
    return run_stage(0, output_dir, args=args or TRAINING_ARGS,
                     resume_checkpoint=resume_checkpoint, eval_every=eval_every)


if __name__ == "__main__":
    run()
