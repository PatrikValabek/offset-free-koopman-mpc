import torch
from torch.autograd import grad

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
    

def evaluate_jacobian(node, x_input, input_name='x', output_name='yhat'):
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
    # Ensure input is float32 and requires grad
    x = x_input.detach().float().clone().requires_grad_(True)

    # Forward pass through the node
    input_dict = {input_name: x}
    output_dict = node(input_dict)
    yhat = output_dict[output_name]  # Expect shape [output_dim]

    if yhat.ndim != 1:
        raise ValueError(f"Expected 1D output, got shape {yhat.shape}")

    # Compute Jacobian row by row
    jacobian_rows = []
    for i in range(yhat.shape[0]):
        grad_output = torch.zeros_like(yhat)
        grad_output[i] = 1.0
        grad_i = grad(yhat, x, grad_outputs=grad_output, retain_graph=True)[0]
        jacobian_rows.append(grad_i)

    return torch.stack(jacobian_rows, dim=0).detach().numpy()
