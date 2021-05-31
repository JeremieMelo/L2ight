import torch
from .layers.custom_linear import (DFALinear, dfa_fuse_logits,
                                   dfa_relu)

__all__ = ["DFA_MLP"]

class DFA_MLP(torch.nn.Module):
    def __init__(self, n_feat, n_class, device=torch.device("cuda")) -> None:
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = DFALinear(n_feat, 64, n_class, bias=False, device=device)
        self.fc2 = DFALinear(64, 64, n_class, bias=False, device=device)
        self.fc3 = DFALinear(64, 10, n_class, bias=False, dfa=False, device=device)
        self.base_matrix = None #torch.nn.init.kaiming_normal_(torch.empty(n_class, 64, device=device))
        self.fc1.reset_feedback_matrix(sparsity=0.3, base_matrix=self.base_matrix)
        self.fc2.reset_feedback_matrix(sparsity=0.3, base_matrix=self.base_matrix)
        self.fc3.reset_feedback_matrix(sparsity=0.3, base_matrix=self.base_matrix)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        grad_final = torch.empty(x.size(0), self.n_class, device=x.device, dtype=x.dtype)

        x, grad_final = self.fc1(x, grad_final)
        x, grad_final = dfa_relu(x, grad_final)
        x, grad_final = self.fc2(x, grad_final)
        x, grad_final = dfa_relu(x, grad_final)
        x, grad_final = self.fc3(x, grad_final)
        out = dfa_fuse_logits(x, grad_final)
        return out

