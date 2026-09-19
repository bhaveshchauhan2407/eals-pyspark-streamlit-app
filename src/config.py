from dataclasses import dataclass


@dataclass
class FastALSConfig:
    factors: int = 10
    max_iter: int = 10
    reg: float = 0.01
    w0: float = 1.0
    alpha: float = 0.5
    init_mean: float = 0.0
    init_stdev: float = 0.001
    show_progress: bool = True
    show_loss: bool = True
    top_k: int = 10
    random_seed: int = 42