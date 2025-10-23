import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

import math
import torch.nn.functional as F

import sys
class matrix_generator(nn.Module):

    def __init__(self, h_dim):
        super(matrix_generator, self).__init__()
        self.h_dim = h_dim
        self.fc = nn.Linear(h_dim, h_dim * h_dim)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.h_dim, self.h_dim)
        return x
    
class matrix_u_generator(nn.Module):

    def __init__(self, h_dim, rank):
        super(matrix_u_generator, self).__init__()
        self.h_dim = h_dim
        self.rank = rank
        self.fc = nn.Linear(h_dim, h_dim * rank)

    def forward(self, x):
        x = self.fc(x).view(-1, self.rank, self.h_dim)
        return x
    
class matrix_v_generator(nn.Module):

    def __init__(self, h_dim, rank):
        super(matrix_v_generator, self).__init__()
        self.h_dim = h_dim
        self.rank = rank
        self.fc = nn.Linear(h_dim, h_dim * rank)

    def forward(self, x):
        x = self.fc(x).view(-1, self.rank, self.h_dim)
        return x
class SAMPONet(nn.Module):

    def __init__(self, args, n_focus_on, h_dim, device=torch.device("cpu")) -> None:
        super(SAMPONet, self).__init__()
        self.args = args
        self.n_focus_on = n_focus_on
        self.do_pruning = args.do_pruning
        assert self.args.generate_uv_matrix == True or self.args.generate_matrix_directly == True or self.args.use_ampo == True, ("No solution is chosen ! Choose one from use_ampo, generate_matrix_directly and generate_uv_matrix.")

        if self.args.use_ampo == True:
            self.CT_W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
            self.CT_W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))

        if self.args.generate_matrix_directly == True:
            self.CT_W_query_generator = matrix_generator(h_dim=h_dim)
            self.CT_W_key_generator = matrix_generator(h_dim=h_dim)

        if self.args.generate_uv_matrix == True:
            self.CT_W_qk_generator_U = matrix_u_generator(h_dim=h_dim, rank=int(h_dim/args.sampo_rank_divide))
            self.CT_W_query_generator_V = matrix_v_generator(h_dim=h_dim, rank=int(h_dim/args.sampo_rank_divide))
            self.CT_W_key_generator_V = matrix_v_generator(h_dim=h_dim, rank=int(h_dim/args.sampo_rank_divide))

        assert self.args.generate_uv_matrix == True or self.args.generate_matrix_directly == True or self.args.use_ampo == True, ("No solution is chosen ! Choose one from use_ampo, generate_matrix_directly and generate_uv_matrix.")
        
        self.device = device

        self.init_parameters()
        self.to(self.device)

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, vs, ve):
        if self.args.use_ampo == True:
            CT_W_query = self.CT_W_query
            CT_W_key = self.CT_W_key

        if self.args.generate_matrix_directly == True:
            CT_W_query = self.CT_W_query_generator(vs)
            CT_W_key = self.CT_W_key_generator(vs) 

        if self.args.generate_uv_matrix == True:
            CT_W_qk_U = self.CT_W_qk_generator_U(vs) 
            CT_W_query_V = self.CT_W_query_generator_V(vs) 
            CT_W_key_V = self.CT_W_key_generator_V(vs) 
            CT_W_query = torch.matmul(CT_W_qk_U, CT_W_query_V) 
            CT_W_key = torch.matmul(CT_W_qk_U, CT_W_key_V) 
        
        Q = torch.matmul(vs, CT_W_query)
        K = torch.matmul(ve, CT_W_key) 

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(1, 2)) 
        compat = compat.squeeze(-2)
        score = F.softmax(compat, dim=-1)

        score_sort_index = torch.argsort(score, dim=-1, descending=True)
        if self.do_pruning == True:
            score_sort_drop_index = score_sort_index[..., :self.n_focus_on]
        else:
            score_sort_drop_index = score_sort_index[..., :]
        
        score_sort_drop_index = score_sort_drop_index.clone().detach().unsqueeze(-1).repeat(1,1,ve.shape[-1])
        pruned_ve = ve.gather(1, score_sort_drop_index)
        return pruned_ve

class SAMPO_Module(nn.Module):

    def __init__(self, args, obs_space, n_enemies, n_allies, device=torch.device("cpu")) -> None:
        super(SAMPO_Module, self).__init__()
        self.args = args
        self.obs_space = obs_space
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self.n_enemy_feature = self.obs_space[2][1]
        self.n_ally_feature = self.obs_space[1][1]
        self.n_self_feature = self.obs_space[4][1]
        self.do_pruning = self.args.do_pruning
        self.n_focus_on = self.args.n_focus_on 

        if self.do_pruning == True:
            assert self.n_focus_on <= self.n_allies, ("the number of focus on agents must be less than or equal to the number of allies")
            self.ds_output_dim = 4*(self.n_focus_on * 2 + 1)
        else:
            self.ds_output_dim = 4*(n_allies + n_enemies + 1)

        self.init_layers()

    def init_layers(self,):
        self.ally_BatchNorm = nn.BatchNorm1d(num_features=self.n_allies)
        self.enemy_BatchNorm = nn.BatchNorm1d(num_features=self.n_enemies)

        ally_obs_space = self.obs_space.copy(); ally_obs_space[0] = self.n_ally_feature
        enemy_obs_space = self.obs_space.copy(); enemy_obs_space[0] = self.n_enemy_feature
        self_obs_space = self.obs_space.copy(); self_obs_space[0] = self.n_self_feature
        assert ally_obs_space[0] == enemy_obs_space[0], ("Need different pre-process MLP for enemy and ally !", ally_obs_space[0], enemy_obs_space[0])
        
        self.other_preproc_MLP = MLPBase(args=self.args, obs_shape=ally_obs_space, output_dim=self.args.sampo_hidden_size)
        self.self_preproc_MLP = MLPBase(args=self.args, obs_shape=self_obs_space, output_dim=self.args.sampo_hidden_size)

        if self.do_pruning == True:
            ds_obs_space = self.obs_space.copy(); ds_obs_space[0] = self.args.sampo_hidden_size*(self.n_focus_on*2+1)
        else:
            ds_obs_space = self.obs_space.copy(); ds_obs_space[0] = self.args.sampo_hidden_size*(self.n_allies+self.n_enemies+1)
        self.down_sampling_MLP = MLPBase(args=self.args, obs_shape=ds_obs_space, output_dim=self.ds_output_dim)

        self.feat_ally = SAMPONet(args=self.args, n_focus_on=self.n_focus_on, h_dim=self.args.sampo_hidden_size, device=self.device)
        self.feat_enemy = SAMPONet(args=self.args, n_focus_on=self.n_focus_on, h_dim=self.args.sampo_hidden_size, device=self.device)

        self.to(self.device)

    def forward(self, input):
        input = check(input).to(**self.tpdv)
        idx = 0
        ally_feature = input[:, idx:self.n_allies*self.n_ally_feature]
        idx += self.n_allies*self.n_ally_feature
        enemy_feature = input[:, idx:idx+self.n_enemies*self.n_enemy_feature]
        idx += self.n_enemies*self.n_enemy_feature
        self_feature = input[:, idx:idx+self.n_self_feature]

        ally_feature = ally_feature.reshape(ally_feature.shape[0], self.n_allies, self.n_ally_feature)
        enemy_feature = enemy_feature.reshape(enemy_feature.shape[0], self.n_enemies, self.n_enemy_feature)
        self_feature = self_feature.unsqueeze(1)

        ve_ally = self.other_preproc_MLP(ally_feature)
        ve_enemy = self.other_preproc_MLP(enemy_feature)
        ve_self = self.self_preproc_MLP(self_feature)

        pruned_ve_ally = self.feat_ally(vs=ve_self, ve=ve_ally)
        pruned_ve_enemy = self.feat_enemy(vs=ve_self, ve=ve_enemy)
        output = torch.cat([pruned_ve_ally.flatten(1), pruned_ve_enemy.flatten(1), ve_self.flatten(1)], dim=-1)

        output = self.down_sampling_MLP(output)

        return output
