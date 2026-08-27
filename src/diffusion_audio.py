"""
Audio corruption processes for timestep-conditioned denoising.

Supports:
    - Poisson
    - Gaussian
    - Bernoulli

The model always predicts:
    residual = noisy - clean

allowing a fair comparison of corruption mechanisms.
"""

import torch


def corrupt_audio(
    x0: torch.Tensor,
    t: torch.Tensor,
    T_audio: int,
    corruption: str = "poisson",
    device: str = "cuda",
    max_rate: float = 1.0,
    scale: float = 15.0,
):
    """
    Parameters
    ----------
    x0 : clean waveform in [0,1]
    t : timestep
    T_audio : total timesteps
    corruption : poisson | gaussian | bernoulli

    Returns
    -------
    noisy
    residual_target = noisy - x0
    """

    t = t.float().view(-1, 1).to(device)

    # normalized timestep
    tau = t / float(T_audio - 1)

    # cosine schedule
    gamma = torch.cos(tau * torch.pi / 2)

    # -------------------------------------------------
    # POISSON
    # -------------------------------------------------

    if corruption.lower() == "poisson":

        signal = gamma * x0

        rate = torch.clamp(
            signal * max_rate,
            min=1e-4,
            max=max_rate
        )

        counts = torch.poisson(rate * scale)

        poisson_noise = counts / scale

        noisy = (
            signal
            + torch.sqrt(1 - gamma**2) * poisson_noise
        )

    # -------------------------------------------------
    # GAUSSIAN
    # -------------------------------------------------

    elif corruption.lower() == "gaussian":

        noise = torch.randn_like(x0)

        noisy = (
            gamma * x0
            + torch.sqrt(1 - gamma**2) * noise
        )

    # -------------------------------------------------
    # BERNOULLI
    # -------------------------------------------------

    elif corruption.lower() == "bernoulli":

        prob = torch.clamp(
            gamma * x0,
            min=0.0,
            max=1.0
        )

        spikes = torch.bernoulli(prob)

        noisy = (
            gamma * x0
            + torch.sqrt(1 - gamma**2) * spikes
        )

    else:
        raise ValueError(
            f"Unknown corruption type: {corruption}"
        )

    noisy = noisy.clamp(0, 1)

    target = noisy - x0

    return noisy, target
