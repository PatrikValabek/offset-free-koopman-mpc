import os
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn

# NeuroMANCER imports to reconstruct the graph similarly to koopman_mpc.py
from neuromancer.system import Node, System
from neuromancer.problem import Problem
from neuromancer.modules import blocks
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable

# Add src to path for helper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
import helper


class PredictionWControl(nn.Module):
    """
    Koopman control model for prediction with control inputs.
    Implements discrete-time dynamical system:
        x_k+1 = K @ x_k + u_k
    with variables:
        x_k - latent states (shape: [batch, nz])
        u_k - latent control inputs (shape: [batch, nz])
    """

    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x, u):
        """
        :param x: (torch.Tensor, shape=[batchsize, nz])
        :param u: (torch.Tensor, shape=[batchsize, nz])
        :return: (torch.Tensor, shape=[batchsize, nz])
        """
        x = self.K(x) + u
        return x

def load_model():
    """
    Load the trained Koopman model and scalers.
    Returns problem object with loaded weights.
    """
    # Model parameters (must match training configuration)
    nz = 13  # latent state dimension
    ny = 3   # output dimension (T1, T2, T4)
    nu = 3   # input dimension (u1, u2, u3)
    nsteps = 80  # prediction horizon
    
    # Network architecture (must match training)
    cons = 10
    layers = [6*cons, 12*cons, 18*cons]  # [12, 24, 36]
    layers_dec = [18*cons, 12*cons, 6*cons]  # [36, 24, 12]
    matrix_C = False
    
    global problem, f_u, K, scaler, scalerU
    
    # Load scalers
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    scaler = joblib.load(os.path.join(data_path, 'scaler.pkl'))
    scalerU = joblib.load(os.path.join(data_path, 'scalerU.pkl'))
    
    # Build the model architecture
    # 1. Output encoder (Y -> latent space)
    f_y = blocks.MLP(
        ny,
        nz,
        bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=layers,
    )
    encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
    encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')
    
    # 2. Input encoder (U -> latent space)
    f_u = torch.nn.Linear(nu, nz, bias=False)
    encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')
    
    # 3. Decoder (latent space -> Y)
    if not matrix_C:
        f_y_inv = blocks.MLP(
            nz, 
            ny, 
            bias=True,
            linear_map=torch.nn.Linear,
            nonlin=torch.nn.ELU,
            hsizes=layers_dec
        )
    else:
        f_y_inv = torch.nn.Linear(nz, ny, bias=False)
    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')
    
    # 4. Koopman operator (A matrix)
    K = torch.nn.Linear(nz, nz, bias=False)
    
    # 5. Koopman dynamics with control
    Koopman = Node(PredictionWControl(K), ['x', 'u_latent'], ['x'], name='K')
    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)
    
    # 6. Assemble nodes
    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
    
    # 7. Define loss (needed for Problem construction)
    Y = variable("Y")
    yhat = variable('yhat')
    x_latent = variable('x_latent')
    u_latent = variable('u_latent')
    x = variable('x')
    xu_latent = x_latent + u_latent
    
    y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"
    onestep_loss = 1.*(yhat[:, 1, :] == Y[:, 1, :])^2
    onestep_loss.name = "onestep_loss"
    reconstruction_loss = 20.*(yhat[:, 0, :] == Y[:, 0, :])^2
    reconstruction_loss.name = "reconstruction_loss"
    x_loss = 1. * (x[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"
    
    objectives = [y_loss, x_loss, onestep_loss, reconstruction_loss]
    loss = PenaltyLoss(objectives, constraints=[])
    
    # 8. Create problem
    problem = Problem(nodes, loss)
    
    # 9. Load trained weights
    model_path = os.path.join(data_path, 'model_C_False.pth')
    problem.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    
    return problem


def get_y(x):
    """
    Decode latent state to output space.
    
    Args:
        x: Latent state (numpy array, shape [nz, 1] or [nz,])
    
    Returns:
        y: Output in original (unscaled) space (numpy array, shape [ny, 1])
    """
    # Ensure x is the right shape
    x = np.array(x).reshape(1, -1)
    
    # Decode using the decoder node
    y_scaled = problem.nodes[4]({"x": torch.from_numpy(x).float()})
    y_scaled = y_scaled["yhat"][0].detach().numpy().reshape(1, -1)
    
    # Inverse transform to get original scale
    y = scaler.inverse_transform(y_scaled)
    
    return y.T  # shape [ny, 1]


def get_x(y):
    """
    Encode output to latent state.
    
    Args:
        y: Output measurement in original (unscaled) space (numpy array, shape [ny, 1] or [ny,])
    
    Updates global variable x with the encoded latent state.
    """
    global x
    y = np.array(y).reshape(1, -1)
    

    # Encode using the encoder node
    x_dict = problem.nodes[0]({"Y0": torch.from_numpy(y.reshape(1, -1, 3)).float()})
    x = x_dict["x"][0].detach().numpy().reshape(-1, 1)


def y_plus(u):
    """
    Predict next output given control input.
    
    Args:
        u: Control input in original (unscaled) space (numpy array, shape [nu, 1] or [nu,])
    
    Returns:
        y: Next output in original (unscaled) space (numpy array, shape [ny, 1])
    
    Updates global variable x with the new latent state.
    """
    global x
    u = np.array(u).reshape(-1, 1)
    
    # Compute next latent state using Koopman dynamics
    x_plus = A @ x + B @ u
    
    # Decode to output space (scaled)
    y_scaled = problem.nodes[4]({"x": torch.from_numpy(x_plus.reshape(1, -1)).float()})
    y_scaled = y_scaled["yhat"][0].detach().numpy().reshape(1, -1)
    
    # Inverse transform to get original scale
    y = y_scaled
    
    # Update latent state
    x = x_plus
    
    return y.flatten()  # shape [ny, 1]


def init():
    """
    Initialize the model and extract A, B matrices.
    Call this function once before using get_x, get_y, or y_plus.
    """
    global A, B, problem, K, f_u, scaler, scalerU
    
    # Load the trained model
    problem = load_model()
    
    # Extract A (Koopman operator) and B (input encoder) matrices
    # PyTorch Linear layers store weights as [out_features, in_features]
    # For Linear(nz, nz): weight is [nz, nz], we need [nz, nz] for A @ x
    # For Linear(nu, nz): weight is [nz, nu], we need [nz, nu] for B @ u
    A = K.weight.detach().numpy()  # Shape: [nz, nz] = [26, 26]
    B = f_u.weight.detach().numpy()  # Shape: [nz, nu] = [26, 3]
    
    print(f"Model initialized successfully!")
    print(f"A matrix shape: {A.shape}")
    print(f"B matrix shape: {B.shape}")
    print(f"Scalers loaded for {scaler.n_features_in_} outputs and {scalerU.n_features_in_} inputs")
    

