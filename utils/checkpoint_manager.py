import os
import torch

def save_checkpoint(
    step,
    actor,
    critic,
    actor_opt,
    critic_opt,
    save_dir,
    best=False
):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
    }

    filename = "best_checkpoint.pth" if best else f"checkpoint_step_{step}.pth"
    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)

    print(f"Checkpoint saved at step {step} -> {path}")


def load_checkpoint(path, actor, critic, actor_opt=None, critic_opt=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    if actor_opt and critic_opt:
        actor_opt.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        critic_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return checkpoint.get("step", 0)

