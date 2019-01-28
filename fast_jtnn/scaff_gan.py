import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from mol_tree import Vocab, MolTree
from nnutils import create_var, avg_pool, index_select_ND, GRU
from jtnn_enc import JTNNEncoder

class ScaffoldGAN(nn.Module):

    def __init__(self, jtnn, hidden_size, beta, gumbel=False):
        super(ScaffoldGAN, self).__init__()
        self.hidden_size = hidden_size 
        self.beta = beta
        self.gumbel = gumbel

        self.netG = Generator(jtnn.decoder)
        self.netD = nn.Sequential(
            nn.Linear(jtnn.hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, 1)
        )

    def reset_netG(self, jtnn):
        self.netG = Generator(jtnn.decoder)

    def encode_real(self, y_batch, jtnn):
        #Generate real y_root features
        y_batch, y_jtenc_holder, _ = y_batch
        y_root_vecs = self.netG(y_jtenc_holder, depth=15)
        return y_root_vecs

    def encode_fake(self, z_batch, jtnn):
        #Generate fake cond features
        z_batch, z_jtenc_holder, z_mpn_holder = z_batch
        z_tree_vecs, _, z_mol_vecs = jtnn.encode(z_jtenc_holder, z_mpn_holder)
        z_tree_vecs_noised, z_mol_vecs_noised = jtnn.fuse_noise(z_tree_vecs, z_mol_vecs)
        
        #Generate fake root features
        pred_root_vecs = []
        for i in xrange(len(z_batch)):
            root_vec,_ = jtnn.decoder.soft_decode(
                    z_tree_vecs_noised[i].unsqueeze(0), z_mol_vecs_noised[i].unsqueeze(0),
                    gumbel=self.gumbel, slope=1.0, temp=1.0
            )
            pred_root_vecs.append(root_vec)

        pred_root_vecs = torch.cat(pred_root_vecs, dim=0)
        return pred_root_vecs

    def train_D(self, x_batch, y_batch, jtnn):
        real_vecs = self.encode_real(y_batch, jtnn).detach()
        fake_vecs = self.encode_fake(x_batch, jtnn).detach()
        real_score = self.netD(real_vecs)
        fake_score = self.netD(fake_vecs)
        score = fake_score.mean() - real_score.mean() #maximize -> minimize minus
        score.backward()

        #Gradient Penalty
        inter_gp, inter_norm = self.gradient_penalty(real_vecs, fake_vecs)
        inter_gp.backward()
        return -score.item(), inter_norm
    
    def train_G(self, x_batch, y_batch, jtnn):
        real_vecs = self.encode_real(y_batch, jtnn)
        fake_vecs = self.encode_fake(x_batch, jtnn)
        real_score = self.netD(real_vecs)
        fake_score = self.netD(fake_vecs)
        score = real_score.mean() - fake_score.mean() 
        score.backward()
        return score.item()

    def gradient_penalty(self, real_vecs, fake_vecs):
        eps = create_var(torch.rand(real_vecs.size(0), 1))
        inter_data = eps * real_vecs + (1 - eps) * fake_vecs
        inter_data = autograd.Variable(inter_data, requires_grad=True)
        inter_score = self.netD(inter_data).squeeze(-1)

        inter_grad = autograd.grad(inter_score, inter_data, 
                grad_outputs=torch.ones(inter_score.size()).cuda(),
                create_graph=True, retain_graph=True, only_inputs=True)[0]

        inter_norm = inter_grad.norm(2, dim=1)
        inter_gp = ((inter_norm - 1) ** 2).mean() * self.beta
        #inter_norm = (inter_grad ** 2).sum(dim=1)
        #inter_gp = torch.max(inter_norm - 1, self.zero).mean() * self.beta

        return inter_gp, inter_norm.mean().item()

class Generator(nn.Module):

    def __init__(self, jtnn_decoder):
        super(Generator, self).__init__()
        self.hidden_size = jtnn_decoder.hidden_size
        self.embedding = jtnn_decoder.embedding
        self.W_z = jtnn_decoder.W_z
        self.W_r = jtnn_decoder.W_r
        self.U_r = jtnn_decoder.U_r
        self.W_h = jtnn_decoder.W_h

    def forward(self, holder, depth):
        fnode = create_var(holder[0])
        fmess = create_var(holder[1])
        node_graph = create_var(holder[2])
        mess_graph = create_var(holder[3])
        scope = holder[4]

        fnode = self.embedding(fnode)
        x = index_select_ND(fnode, 0, fmess)
        h = create_var(torch.zeros(mess_graph.size(0), self.hidden_size))

        mask = torch.ones(h.size(0), 1)
        mask[0] = 0 #first vector is padding
        mask = create_var(mask)

        for it in xrange(depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            h = GRU(x, h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            h = h * mask

        mess_nei = index_select_ND(h, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)
        root_vecs = [node_vecs[st] for st,le in scope]
        return torch.stack(root_vecs, dim=0)

def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx: continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


