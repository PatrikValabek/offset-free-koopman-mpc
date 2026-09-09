from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.problem import Problem
from neuromancer.system import Node, System
from neuromancer.trainer import Trainer


class PredictionWControl(torch.nn.Module):
    """
    Baseline class for Koopman control model
    Implements discrete-time dynamical system:
        x_k+1 = K x_k + u_k
    with variables:
        x_k - latent states
        u_k - latent control inputs
    """

    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x, u):
        """
        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nx])
        :return: (torch.Tensor, shape=[batchsize, nx])
        """
        x = self.K(x) + u
        return x


def autoencoder_hsizes(
    n_in: int,
    n_out: int,
    depth: int,
    width_mult: float = 1.0,
) -> list[int]:
    """Geometric hidden widths between n_in and n_out for depth hidden layers."""
    if depth <= 0:
        return []
    sizes = np.geomspace(max(n_in, 1), max(n_out, 1), num=depth + 2)[1:-1]
    return [max(1, round(float(s) * width_mult)) for s in sizes]


def get_nonlin(name: str) -> type[torch.nn.Module]:
    mapping = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "gelu": torch.nn.GELU,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown activation {name!r}; choose from {list(mapping)}")
    return mapping[key]


@dataclass
class LossWeights:
    y_loss: float = 10.0
    x_loss: float = 10.0
    onestep_loss: float = 1.0
    reconstruction_loss: float = 1.0


@dataclass
class TrainConfig:
    nsteps: int = 80
    bs: int = 80
    lr: float = 1e-3
    epochs: int = 4000
    warmup: int = 100
    patience: int = 300


def load_cstr_separator_data(
    data_path: str,
    n_train: int = 12960,
    n_dev: int = 2880,
):
    data = np.load(data_path, allow_pickle=True)
    sim = {
        "Y": np.array(data["Y"], dtype=float),
        "U": np.array(data["U"], dtype=float),
    }
    train = {k: v[:n_train] for k, v in sim.items()}
    dev = {k: v[n_train : n_train + n_dev] for k, v in sim.items()}
    test = {k: v[n_train + n_dev :] for k, v in sim.items()}
    y_names = list(data["y_names"]) if "y_names" in data.files else None
    u_names = list(data["u_names"]) if "u_names" in data.files else None
    return sim, train, dev, test, y_names, u_names


def get_data_loaders(
    train_sim: dict,
    dev_sim: dict,
    test_sim: dict,
    nsteps: int,
    bs: int,
    scaler,
    scalerU,
):
    ny = train_sim["Y"].shape[1]
    nu = train_sim["U"].shape[1]

    def _windowed(sim_dict, as_single_batch: bool = False):
        nsim = sim_dict["Y"].shape[0]
        nbatch = nsim // nsteps
        length = nbatch * nsteps
        x = torch.tensor(
            scaler.transform(sim_dict["Y"][:length]),
            dtype=torch.float32,
        )
        u = torch.tensor(
            scalerU.transform(sim_dict["U"][:length]),
            dtype=torch.float32,
        )
        if as_single_batch:
            x = x.reshape(1, length, ny)
            u = u.reshape(1, length, nu)
        else:
            x = x.reshape(nbatch, nsteps, ny)
            u = u.reshape(nbatch, nsteps, nu)
        return {"Y": x, "Y0": x[:, 0:1, :], "U": u}

    train_dict = _windowed(train_sim, as_single_batch=False)
    dev_dict = _windowed(dev_sim, as_single_batch=False)
    test_dict = _windowed(test_sim, as_single_batch=True)
    train_full_dict = _windowed(train_sim, as_single_batch=True)

    train_data = DictDataset(train_dict, name="train")
    dev_data = DictDataset(dev_dict, name="dev")
    train_loader = DataLoader(
        train_data,
        batch_size=bs,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=bs,
        collate_fn=dev_data.collate_fn,
        shuffle=True,
    )
    return train_loader, dev_loader, test_dict, train_full_dict


def build_koopman_problem(
    ny: int,
    nu: int,
    nz: int,
    matrix_C: bool,
    encoder_depth: int,
    width_mult: float,
    nonlin: str,
    loss_weights: LossWeights,
    nsteps: int,
):
    nonlin_cls = get_nonlin(nonlin)
    enc_hsizes = autoencoder_hsizes(ny, nz, encoder_depth, width_mult)
    dec_hsizes = autoencoder_hsizes(nz, ny, encoder_depth, width_mult)

    f_y = blocks.MLP(
        ny,
        nz,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=nonlin_cls,
        hsizes=enc_hsizes,
    )
    encode_Y0 = Node(f_y, ["Y0"], ["x"], name="encoder_Y0")
    encode_Y = Node(f_y, ["Y"], ["x_latent"], name="encoder_Y")

    f_u = torch.nn.Linear(nu, nz, bias=False)
    encode_U = Node(f_u, ["U"], ["u_latent"], name="encoder_U")

    if matrix_C:
        f_y_inv = torch.nn.Linear(nz, ny, bias=False)
    else:
        f_y_inv = blocks.MLP(
            nz,
            ny,
            bias=True,
            linear_map=torch.nn.Linear,
            nonlin=nonlin_cls,
            hsizes=dec_hsizes,
        )
    decode_y = Node(f_y_inv, ["x"], ["yhat"], name="decoder_y")

    K = torch.nn.Linear(nz, nz, bias=False)
    koopman = Node(PredictionWControl(K), ["x", "u_latent"], ["x"], name="K")
    dynamics_model = System([koopman], name="Koopman", nsteps=nsteps)
    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]

    Y = variable("Y")
    yhat = variable("yhat")
    x_latent = variable("x_latent")
    u_latent = variable("u_latent")
    x = variable("x")
    xu_latent = x_latent + u_latent

    y_loss = loss_weights.y_loss * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"
    onestep_loss = loss_weights.onestep_loss * (yhat[:, 1, :] == Y[:, 1, :]) ^ 2
    onestep_loss.name = "onestep_loss"
    reconstruction_loss = loss_weights.reconstruction_loss * (yhat[:, 0, :] == Y[:, 0, :]) ^ 2
    reconstruction_loss.name = "reconstruction_loss"
    x_loss = loss_weights.x_loss * (x[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"

    loss = PenaltyLoss(
        [y_loss, x_loss, onestep_loss, reconstruction_loss],
        constraints=[],
    )
    problem = Problem(nodes, loss)
    return problem, K, f_u, f_y_inv


def make_trainer(
    problem: Problem,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    eval_data: dict,
    train_cfg: TrainConfig,
    verbose: bool = False,
):
    from neuromancer.trainer import Callback

    optimizer = torch.optim.Adam(problem.parameters(), lr=train_cfg.lr)

    class NoOpCallback(Callback):
        pass

    class PrintCallback(Callback):
        def end_eval(self, trainer, output):
            if trainer.current_epoch % trainer.epoch_verbose == 0:
                train_loss = output.get(
                    "mean_train_loss",
                    output.get("train_loss", "N/A"),
                )
                dev_loss = output.get(
                    "mean_dev_loss",
                    output.get("dev_loss", "N/A"),
                )
                print(
                    f"epoch: {trainer.current_epoch}  "
                    f"train_loss: {train_loss}  "
                    f"dev_loss: {dev_loss}"
                )

    callback = PrintCallback() if verbose else NoOpCallback()

    return Trainer(
        problem,
        train_loader,
        dev_loader,
        eval_data,
        optimizer,
        callback=callback,
        patience=train_cfg.patience,
        warmup=train_cfg.warmup,
        epochs=train_cfg.epochs,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
        epoch_verbose=1 if verbose else 10**9,
    )


def train_koopman(
    problem: Problem,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    eval_data: dict,
    train_cfg: TrainConfig,
    verbose: bool = False,
):
    trainer = make_trainer(
        problem,
        train_loader,
        dev_loader,
        eval_data,
        train_cfg,
        verbose=verbose,
    )
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    return best_model, trainer


def rollout_predictions(problem: Problem, data_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop rollout; returns (true_y, pred_y) in standardized space."""
    ny = data_dict["Y"].shape[-1]
    problem.nodes[3].nsteps = data_dict["Y"].shape[1]
    outputs = problem.step(data_dict)
    pred = outputs["yhat"][:, 1:-1, :].detach().numpy().reshape(-1, ny)
    true = data_dict["Y"][:, 1 : pred.shape[0] + 1, :].detach().numpy().reshape(-1, ny)
    return true, pred


def evaluate_mae_physical(
    problem: Problem,
    data_dict: dict,
    scaler,
    y_names: Optional[list[str]] = None,
) -> dict[str, float | np.ndarray]:
    true_std, pred_std = rollout_predictions(problem, data_dict)
    true_phys = scaler.inverse_transform(true_std)
    pred_phys = scaler.inverse_transform(pred_std)
    mae_per_output = np.mean(np.abs(true_phys - pred_phys), axis=0)
    result = {
        "mae_per_output": mae_per_output,
        "mae_sum": float(np.sum(mae_per_output)),
        "mae_mean": float(np.mean(mae_per_output)),
    }
    if y_names is not None:
        result["mae_by_name"] = {name: float(v) for name, v in zip(y_names, mae_per_output)}
    return result


def extract_matrices_from_modules(
    K: torch.nn.Linear,
    f_u: torch.nn.Linear,
    f_y_inv: torch.nn.Module,
    matrix_C: bool,
    problem: Problem,
    train_full_dict: Optional[dict] = None,
):
    A = K.weight.detach().numpy()
    B = f_u.weight.detach().numpy()
    nz = A.shape[0]
    if matrix_C:
        C = f_y_inv.weight.detach().numpy()
    else:
        if train_full_dict is None:
            raise ValueError("train_full_dict required when matrix_C=False")
        ny = f_y_inv.out_features if hasattr(f_y_inv, "out_features") else f_y_inv.weight.shape[0]
        problem.nodes[3].nsteps = train_full_dict["Y"].shape[1]
        train_outputs = problem.step(train_full_dict)
        Y = train_full_dict["Y"].reshape(-1, ny).detach().numpy()
        Z = train_outputs["x"][:, :-1, :].detach().numpy().reshape(-1, nz)
        C, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
        C = C.T
    return A, B, C


def extract_matrices(
    problem: Problem,
    matrix_C: bool,
    train_full_dict: Optional[dict] = None,
):
    """Legacy helper kept for notebook compatibility."""
    dynamics = problem.nodes[3]
    koopman_node = dynamics.nodes[0]
    pred_module = koopman_node.op if hasattr(koopman_node, "op") else koopman_node.module
    K = pred_module.K
    f_u = problem.nodes[2].op if hasattr(problem.nodes[2], "op") else problem.nodes[2].module
    f_y_inv = problem.nodes[4].op if hasattr(problem.nodes[4], "op") else problem.nodes[4].module
    return extract_matrices_from_modules(
        K, f_u, f_y_inv, matrix_C, problem, train_full_dict
    )


def evaluate_jacobian(node, x_input, input_name="x", output_name="yhat"):
    """
    Compute the Jacobian dyhat/dx for a Neuromancer Node.

    Args:
        node: Neuromancer Node (a torch.nn.Module that takes a dict input)
        x_input: torch.Tensor of shape [input_dim] (NOT batched)
        input_name: str, key name expected in the input dict
        output_name: str, key name returned in the output dict

    Returns:
        Jacobian matrix: torch.Tensor of shape [output_dim, input_dim]
    """
    x = x_input.detach().float().clone().requires_grad_(True)
    input_dict = {input_name: x}
    output_dict = node(input_dict)
    yhat = output_dict[output_name]

    if yhat.ndim != 1:
        raise ValueError(f"Expected 1D output, got shape {yhat.shape}")

    jacobian_rows = []
    for i in range(yhat.shape[0]):
        grad_output = torch.zeros_like(yhat)
        grad_output[i] = 1.0
        grad_i = grad(yhat, x, grad_outputs=grad_output, retain_graph=True)[0]
        jacobian_rows.append(grad_i)

    return torch.stack(jacobian_rows, dim=0).detach().numpy()
