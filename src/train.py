"""
Training loops for image and audio denoisers.

Audio branch supports:
    - Poisson corruption
    - Gaussian corruption
    - Bernoulli corruption

through a unified corruption interface.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.diffusion_audio import corrupt_audio


# ============================================================
# IMAGE TRAINING
# ============================================================

def train_img_model(
    model,
    name,
    train_loader,
    diff,
    timesteps,
    epochs,
    lr,
    device,
):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
    )

    print(f"\nTraining {name}...")

    for ep in range(epochs):

        model.train()

        total = 0.0

        for x, _ in tqdm(
            train_loader,
            leave=False,
            desc=f"{name} Ep {ep + 1}",
        ):

            x = x.to(device)

            t = torch.randint(
                0,
                timesteps,
                (x.size(0),),
                device=device,
            )

            noise = torch.randn_like(x)

            xt = diff.q_sample(
                x,
                t,
                noise,
            )

            pred = model(
                xt,
                t,
            )

            loss = F.mse_loss(
                pred,
                noise,
            )

            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            total += loss.item()

        sched.step()

        print(
            f"Epoch {ep + 1}/{epochs} "
            f"Loss: {total / len(train_loader):.4f}"
        )

    return model


# ============================================================
# AUDIO TRAINING
# ============================================================

def train_audio_model(
    model,
    name,
    train_loader,
    T_audio,
    max_rate,
    epochs,
    lr,
    device,
    corruption="poisson",
):
    """
    corruption:
        - poisson
        - gaussian
        - bernoulli
    """

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
    )

    print(
        f"\nTraining {name} "
        f"[corruption={corruption}]..."
    )

    for ep in range(epochs):

        model.train()

        total = 0.0

        for x0 in tqdm(
            train_loader,
            leave=False,
            desc=f"{name} Ep {ep + 1}",
        ):

            x0 = x0.to(device)

            t = torch.randint(
                0,
                T_audio,
                (x0.size(0),),
                device=device,
            )

            noisy, target = corrupt_audio(
                x0=x0,
                t=t,
                T_audio=T_audio,
                corruption=corruption,
                device=device,
                max_rate=max_rate,
            )

            pred = model(
                noisy,
                t,
            )

            loss = F.mse_loss(
                pred,
                target,
            )

            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            total += loss.item()

        sched.step()

        avg_loss = total / len(train_loader)

        print(
            f"Epoch {ep + 1}/{epochs} "
            f"Loss: {avg_loss:.6f}"
        )

    return model
