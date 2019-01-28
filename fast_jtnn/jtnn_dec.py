import torch
import torch.nn as nn
import torch.nn.functional as F
from mol_tree import Vocab, MolTree, MolTreeNode
from nnutils import create_var, GRU
from chemutils import enum_assemble
import copy

MAX_NB = 15
MAX_DECODE_LEN = 100
MAX_SOFT_DECODE_LEN = 60

class JTNNDecoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding, use_molatt):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding
        self.use_molatt = use_molatt

        #GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        #Word Prediction Weights (attention matrix no bias)
        self.W_t = nn.Linear(hidden_size, hidden_size, bias=False)
        if use_molatt:
            self.W = nn.Linear(3 * hidden_size, hidden_size)
            self.W_g = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.W = nn.Linear(2 * hidden_size, hidden_size)
            self.W_g = None

        #Stop Prediction Weights (attention matrix no bias)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)
        self.U_t = nn.Linear(hidden_size, hidden_size, bias=False)
        if use_molatt:
            self.U = nn.Linear(3 * hidden_size, hidden_size)
            self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.U = nn.Linear(2 * hidden_size, hidden_size)
            self.U_g = None

        #Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(size_average=False)
        self.stop_loss = nn.BCEWithLogitsLoss(size_average=False)

    def attention(self, hiddens, contexts, x_tree_vecs, x_mol_vecs, mode):
        if mode == 'word':
            V, V_t, V_g, V_o = self.W, self.W_t, self.W_g, self.W_o
        elif mode == 'stop':
            V, V_t, V_g, V_o = self.U, self.U_t, self.U_g, self.U_o
        else:
            raise ValueError('attention mode is wrong')

        tree_vecs = x_tree_vecs.index_select(0, contexts)
        tree_att = torch.bmm( tree_vecs, V_t(hiddens).unsqueeze(-1) )
        tree_contexts = (F.softmax(tree_att, dim=1) * tree_vecs).sum(dim=1)

        if self.use_molatt:
            mol_vecs = x_mol_vecs.index_select(0, contexts)
            mol_att = torch.bmm( mol_vecs, V_g(hiddens).unsqueeze(-1) )
            mol_contexts = (F.softmax(mol_att, dim=1) * mol_vecs).sum(dim=1)
            input_vec = torch.cat([hiddens, tree_contexts, mol_contexts], dim=1)
        else:
            input_vec = torch.cat([hiddens, tree_contexts], dim=1)

        output_vec = F.relu( V(input_vec) )
        return V_o(output_vec)

    def forward(self, mol_batch, x_tree_vecs, x_mol_vecs):
        pred_hiddens,pred_contexts,pred_targets = [],[],[]
        stop_hiddens,stop_contexts,stop_targets = [],[],[]
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        #Predict Root
        batch_size = len(mol_batch)
        pred_hiddens.append(create_var(torch.zeros(len(mol_batch),self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append( create_var( torch.LongTensor(range(batch_size)) ) )

        max_iter = max([len(tr) for tr in traces])
        padding = create_var(torch.zeros(self.hidden_size), False)
        h = {}

        for t in xrange(max_iter):
            prop_list = []
            batch_list = []
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei,cur_o_nei = [],[]

            for node_x, real_y, _ in prop_list:
                #Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                #Current clique embedding
                cur_x.append(node_x.wid)

            #Clique embedding
            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)

            #Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            #Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(prop_list):
                node_x,node_y,direction = m
                x,y = node_x.idx,node_y.idx
                h[(x,y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i) 
                stop_target.append(direction)

            #Hidden states for stop prediction
            cur_batch = create_var(torch.LongTensor(batch_list))
            stop_hidden = torch.cat([cur_x,cur_o], dim=1)
            stop_hiddens.append( stop_hidden )
            stop_contexts.append( cur_batch )
            stop_targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))
                pred_contexts.append( cur_batch )

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append( new_h.index_select(0, cur_pred) )
                pred_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var(torch.LongTensor(cur_x))
        cur_x = self.embedding(cur_x)
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x,cur_o], dim=1)
        stop_hiddens.append( stop_hidden )
        stop_contexts.append( create_var( torch.LongTensor(range(batch_size)) ) )
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.attention(pred_hiddens, pred_contexts, x_tree_vecs, x_mol_vecs, 'word')
        pred_targets = create_var(torch.LongTensor(pred_targets))

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        #Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu( self.U_i(stop_hiddens) )
        stop_scores = self.attention(stop_hiddens, stop_contexts, x_tree_vecs, x_mol_vecs, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = create_var(torch.Tensor(stop_targets))
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()
    
    def decode(self, x_tree_vecs, x_mol_vecs):
        assert x_tree_vecs.size(0) == 1

        stack = []
        init_hiddens = create_var( torch.zeros(1, self.hidden_size) )
        zero_pad = create_var(torch.zeros(1,1,self.hidden_size))
        contexts = create_var( torch.LongTensor(1).zero_() )

        #Root Prediction
        root_score = self.attention(init_hiddens, contexts, x_tree_vecs, x_mol_vecs, 'word')
        _,root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = MolTreeNode(self.vocab.get_smiles(root_wid))
        root.wid = root_wid
        root.idx = 0
        stack.append( (root, self.vocab.get_slots(root.wid)) )

        all_nodes = [root]
        h = {}
        for step in xrange(MAX_DECODE_LEN):
            node_x,fa_slot = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(torch.LongTensor([node_x.wid]))
            cur_x = self.embedding(cur_x)

            #Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
            stop_hiddens = F.relu( self.U_i(stop_hiddens) )
            stop_score = self.attention(stop_hiddens, contexts, x_tree_vecs, x_mol_vecs, 'stop')
            
            backtrack = (stop_score.item() < 0) 

            if not backtrack: #Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.attention(new_h, contexts, x_tree_vecs, x_mol_vecs, 'word')

                _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:5]:
                    slots = self.vocab.get_slots(wid)
                    node_y = MolTreeNode(self.vocab.get_smiles(wid))
                    if have_slots(fa_slot, slots) and can_assemble(node_x, node_y):
                        next_wid = wid
                        next_slots = slots
                        break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = MolTreeNode(self.vocab.get_smiles(next_wid))
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y,next_slots) )
                    all_nodes.append(node_y)

            if backtrack: #Backtrack, use if instead of else
                if len(stack) == 1: 
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

    def soft_decode(self, x_tree_vecs, x_mol_vecs, gumbel, slope, temp):
        assert x_tree_vecs.size(0) == 1

        soft_embedding = lambda x: x.matmul(self.embedding.weight)
        if gumbel:
            sample_softmax = lambda x: F.gumbel_softmax(x, tau=temp)
        else:
            sample_softmax = lambda x: F.softmax(x / temp, dim=1)

        stack = []
        init_hiddens = create_var( torch.zeros(1, self.hidden_size) )
        zero_pad = create_var(torch.zeros(1,1,self.hidden_size))
        contexts = create_var( torch.LongTensor(1).zero_() )

        #Root Prediction
        root_score = self.attention(init_hiddens, contexts, x_tree_vecs, x_mol_vecs, 'word')
        root_prob = sample_softmax(root_score)

        root = MolTreeNode("")
        root.embedding = soft_embedding(root_prob)
        root.prob = root_prob
        root.idx = 0
        stack.append(root)

        all_nodes = [root]
        all_hiddens = []
        h = {}
        for step in xrange(MAX_SOFT_DECODE_LEN):
            node_x = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            #Predict stop
            cur_x = node_x.embedding
            cur_h = cur_h_nei.sum(dim=1)
            stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
            stop_hiddens = F.relu( self.U_i(stop_hiddens) )
            stop_score = self.attention(stop_hiddens, contexts, x_tree_vecs, x_mol_vecs, 'stop')
            all_hiddens.append( stop_hiddens )
            
            forward = 0 if stop_score.item() < 0 else 1
            stop_prob = F.hardtanh(slope * stop_score + 0.5, min_val=0, max_val=1).unsqueeze(1)
            stop_val_ste = forward + stop_prob - stop_prob.detach()

            if forward == 1: #Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.attention(new_h, contexts, x_tree_vecs, x_mol_vecs, 'word')
                pred_prob = sample_softmax(pred_score)

                node_y = MolTreeNode("")
                node_y.embedding = soft_embedding(pred_prob)
                node_y.prob = pred_prob
                node_y.idx = len(all_nodes)
                node_y.neighbors.append(node_x)

                h[(node_x.idx,node_y.idx)] = new_h[0] * stop_val_ste
                stack.append(node_y)
                all_nodes.append(node_y)
            else:
                if len(stack) == 1: #At root, terminate
                    return torch.cat([cur_x,cur_h], dim=1), all_nodes

                node_fa = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0] * (1.0 - stop_val_ste)
                node_fa.neighbors.append(node_x)
                stack.pop()

        #Failure mode: decoding unfinished
        cur_h_nei = [ h[(node_y.idx,root.idx)] for node_y in root.neighbors ]
        if len(cur_h_nei) > 0:
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
        else:
            cur_h_nei = zero_pad
        cur_h = cur_h_nei.sum(dim=1)

        stop_hiddens = torch.cat([root.embedding, cur_h], dim=1)
        stop_hiddens = F.relu( self.U_i(stop_hiddens) )
        all_hiddens.append( stop_hiddens )

        return torch.cat([root.embedding,cur_h], dim=1), all_nodes
        
"""
Helper Functions:
"""

def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        dfs(stack, y, x.idx)
        stack.append( (y,x,0) )

def sorted_dfs(x_stack, x, fa_idx):
    tot = 0
    all_stacks = []
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        y_stack = [(x,y,1)]
        dfs(y_stack, y, x.idx)
        y_stack.append( (y,x,0) )
        all_stacks.append(y_stack)

    all_stacks = sorted(all_stacks, key=lambda x:len(x))
    for stk in all_stacks:
        x_stack.extend(stk)

def have_slots(fa_slots, ch_slots):
    if len(fa_slots) > 2 and len(ch_slots) > 2:
        return True
    matches = []
    for i,s1 in enumerate(fa_slots):
        a1,c1,h1 = s1
        for j,s2 in enumerate(ch_slots):
            a2,c2,h2 = s2
            if a1 == a2 and c1 == c2 and (a1 != "C" or h1 + h2 >= 4):
                matches.append( (i,j) )

    if len(matches) == 0: return False

    fa_match,ch_match = zip(*matches)
    if len(set(fa_match)) == 1 and 1 < len(fa_slots) <= 2: #never remove atom from ring
        fa_slots.pop(fa_match[0])
    if len(set(ch_match)) == 1 and 1 < len(ch_slots) <= 2: #never remove atom from ring
        ch_slots.pop(ch_match[0])

    return True
    
def can_assemble(node_x, node_y):
    neis = node_x.neighbors + [node_y]
    for i,nei in enumerate(neis):
        nei.nid = i

    neighbors = [nei for nei in neis if nei.mol.GetNumAtoms() > 1]
    neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
    singletons = [nei for nei in neis if nei.mol.GetNumAtoms() == 1]
    neighbors = singletons + neighbors
    cands = enum_assemble(node_x, neighbors)
    return len(cands) > 0

if __name__ == "__main__":
    smiles = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1","O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2", "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3", "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1", 'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1', "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]
    for s in smiles:
        print s
        tree = MolTree(s)
        for i,node in enumerate(tree.nodes):
            node.idx = i

        stack = []
        dfs(stack, tree.nodes[0], -1)
        for x,y,d in stack:
            print x.smiles, y.smiles, d
        print '------------------------------'
