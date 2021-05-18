import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import Base_Model
from utils.pytorch_helper import FF, get_init_function

class SNUH(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.tau = torch.Tensor([0.99])
    
    def define_parameters(self):
        self.venc = VarEncoder(self.data.vocab_size, self.hparams.dim_hidden,
                              self.hparams.num_features,
                              self.hparams.num_layers)

        self.cenc = CorrEncoder(self.data.vocab_size, self.hparams.dim_hidden,
                            self.hparams.num_features,
                            self.hparams.num_layers)

        self.dec = Decoder(self.hparams.num_features, self.data.vocab_size)
        

        self.apply(get_init_function(self.hparams.init))
    
    def forward(self, X, edge1, edge2, weight, num_node, num_edges):
        q_mu, q_sigma = self.venc(X, self.hparams.temperature)

        eps = torch.randn_like(q_mu)
        Z_st = q_mu + q_sigma * eps

        log_likelihood = self.dec(Z_st, X.sign())
        kl = self.compute_kl(q_mu, q_sigma, edge1, edge2, weight, num_node, num_edges)

        loss = -log_likelihood + self.hparams.beta * kl
        return {'loss': loss, 'log_likelihood': log_likelihood, 'kl': kl}
    
    def compute_kl(self, q_mu, q_sigma, edge1, edge2, weight, num_node, num_edges):
        q_mu1, q_sigma1 = self.venc(edge1, self.hparams.temperature)
        q_mu2, q_sigma2 = self.venc(edge2, self.hparams.temperature)

        kl_node = torch.mean(torch.sum(q_mu**2 + q_sigma**2 - 1 - 2*torch.log(q_sigma + 1e-8), dim=1))

        gamma = self.cenc(edge1, edge2)
        kl_edge = torch.mean(torch.sum(0.5 * (q_mu1**2 + q_mu2**2 + q_sigma1**2 + q_sigma2**2 - 2 * self.tau * gamma * q_sigma1 * q_sigma2 - 2 * self.tau * q_mu1 * q_mu2)\
                    / (1 - self.tau**2) - 0.5 * (q_mu1**2 + q_mu2**2 + q_sigma1**2 + q_sigma2**2)\
                    - 0.5 * torch.log(1 - gamma + 1e-8) + 0.5 * torch.log(1 - self.tau**2), dim=1) * weight)

        return kl_node + kl_edge * num_edges / num_node
    
    def encode_discrete(self, X):
        mu, _ = self.venc(X, self.hparams.temperature)
        return mu.round()
    
    def get_median_threshold_binary_code(self, device, is_val=True):
        train_doc = self.data.X_train.to(device)
        train_mu, _ = self.venc(train_doc, self.hparams.temperature)
        train_y = self.data.Y_train

        if is_val:
            # validation
            test_doc = self.data.X_val.to(device)
            test_mu, _ = self.venc(test_doc, self.hparams.temperature)
            test_y = self.data.Y_val
        else:
            # testing
            test_doc = self.data.X_test.to(device)
            test_mu, _ = self.venc(test_doc, self.hparams.temperature)
            test_y = self.data.Y_test
        
        mid_val, _ = torch.median(train_mu, dim=0)
        train_b = (train_mu > mid_val).type(torch.FloatTensor).to(device)
        test_b = (test_mu > mid_val).type(torch.FloatTensor).to(device)

        del train_mu
        del test_mu
        return train_b, test_b, train_y, test_y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]
        
    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'num_layers': [0],
            'num_trees': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'num_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
            'temperature': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'beta': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument('--num_trees', type=int, default=10,
                            help='num of trees [%(default)d]')
        parser.add_argument("--temperature", type=float, default=0.1,
                            help='temperature for binarization [%(default)g]')
        parser.add_argument("--alpha", type=float, default=0.1,
                            help='temperature for sampling neighbors [%(default)g]')
        parser.add_argument('--beta', type=float, default=0.05,
                            help='beta term (as in beta-VAE) [%(default)g]')
        
        return parser

class VarEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.dim_output = dim_output
        self.ff = FF(dim_input, dim_hidden, 2 * dim_output, num_layers)
    
    def forward(self, x, temperature):
        gaussian_params = self.ff(x)
        mu = torch.sigmoid(gaussian_params[:, :self.dim_output] / temperature)
        sigma = F.softplus(gaussian_params[:, self.dim_output:])
        return mu , sigma

class CorrEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.dim_output = dim_output
        self.ff = FF(2 * dim_input, dim_hidden, dim_output, num_layers)
    
    def forward(self, x1, x2):
        net = torch.cat([torch.cat([x1, x2], dim=1), torch.cat([x2, x1], dim=1)], dim=0)
        corr_params = self.ff(net).reshape([2, -1, self.dim_output])
        corr_params = (corr_params[0] + corr_params[1]) / 2.0
        correlation_coefficient = (1. - 1e-8) * (2. * torch.sigmoid(corr_params) - 1.)
        return correlation_coefficient

class Decoder(nn.Module):  # As in VDSH, NASH, BMSH
    def __init__(self, dim_encoding, vocab_size):
        super().__init__()
        self.E = nn.Embedding(dim_encoding, vocab_size)
        self.b = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, Z, targets):  # (B x m), (B x V binary)
        scores = Z @ self.E.weight + self.b # B x V
        log_probs = scores.log_softmax(dim=1)
        log_likelihood = (log_probs * targets).sum(1).mean()
        return log_likelihood