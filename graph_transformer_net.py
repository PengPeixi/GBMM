############## Version 2 ############################

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
# from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

import numpy as np

import numpy as np
import dgl
from mambapy.mamba import Mamba, MambaConfig
from multi_head_attention import MultiHeadedAttention as MHA_block
# from aff_net.fusion import iAFF

def get_adjacent_indices_for_dgl(conv_output_shape):
    """
    生成展平后张量中每个向量的相邻索引列表，并返回dgl图所需的源和目标列表。

    参数:
    conv_output_shape (tuple): 卷积输出的空间维度 (h, w, d)。

    返回:
    src_list (list of int): 边的源节点列表。
    dst_list (list of int): 边的目标节点列表。
    """
    h, w = conv_output_shape  # 输出的高度、宽度、深度
    total_elements = h * w  # 总的元素个数
    
    # 生成展平的索引 (h * w * d) 的矩阵
    indices = np.arange(total_elements).reshape(h, w)
    
    src_list = []
    dst_list = []
    
    for i in range(h):
        for j in range(w):
                current_idx = indices[i, j]
                
                # 检查每个方向的相邻点，确保不越界
                
                # y方向相邻
                if i > 0:
                    neighbor_idx = indices[i - 1, j]  # 上面
                    src_list.append(current_idx)
                    dst_list.append(neighbor_idx)
                
                if i < h - 1:
                    neighbor_idx = indices[i + 1, j]  # 下面
                    src_list.append(current_idx)
                    dst_list.append(neighbor_idx)
                
                # x方向相邻
                if j > 0:
                    neighbor_idx = indices[i, j - 1]  # 左边
                    src_list.append(current_idx)
                    dst_list.append(neighbor_idx)
                
                if j < w - 1:
                    neighbor_idx = indices[i, j + 1]  # 右边
                    src_list.append(current_idx)
                    dst_list.append(neighbor_idx)       
    
    return src_list, dst_list


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # img_size = (img_size, img_size)
        # patch_size = (patch_size, patch_size)
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # self.norm = nn.BatchNorm2d(1)
        self.norm = nn.Identity()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # B, C, H, W, D = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        # x = self.relu(x)
        return x

class GraphTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_dim = 768
        num_heads = 8
        out_dim = 768
        dropout = 0.0
        n_layers = 4
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True

        n_gpu = 1
        self.device, _ = self._prepare_device(n_gpu)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        
        self.mlayers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.mlayers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        
        self.cross_layers = nn.ModuleList([ MHA_block(2, 122) for _ in range(n_layers) ]) 
        self.self_layers = nn.ModuleList([ MHA_block(2, 122) for _ in range(n_layers) ]) 
        self.proj_layers = nn.ModuleList([ nn.Linear(768,122) for _ in range(n_layers) ]) 
        
        self.MLP_layer = MLPReadout(122, 2)
        self.patch_embed = PatchEmbed(img_size=64, patch_size=16, in_c=3, embed_dim=768)
        self.patch_embed_mask = PatchEmbed(img_size=64, patch_size=16, in_c=3, embed_dim=768)
        self.src_list, self.dst_list = get_adjacent_indices_for_dgl((4,4))
        self.MLP_layer_0 = nn.Linear(18,1)
        # self.MLP_layer_0 = nn.Linear(16*2+1,1)
        # self.MLP_layer_0 = nn.Linear(128,1)
        self.n_classes=2
        config = MambaConfig(d_model=122, n_layers=2)
        self.mamba = Mamba(config)
        self.pooling =  nn.AdaptiveAvgPool2d((1, 768))
        self.pooling_proj =  nn.Linear(768, 122)
        self.pooling_proj_e =  nn.Linear(768, 122)
        self.hc_pooling =  nn.AdaptiveAvgPool2d((1, 122))
        self.eproj =  nn.Linear(768*2,768)
        # self.proj_clinical_2 = nn.Linear(122,768)
        self.proj_clinical = nn.Linear(122,122)
        # self.proj_ds = nn.Linear(768*2,122)
        # self.proj_clinical = nn.Sequential(
        #     nn.Linear(122,768),
        #     # nn.BatchNorm2d(1),
        #     nn.Identity(),
        #     nn.ReLU(inplace=True),
        # )
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, 768))
        self.pos_embed_m = nn.Parameter(torch.zeros(1, 16, 768))
        # self.fusion = iAFF(channels=1, r=1)
        
    # def forward(self, g, h, e):
    def forward(self, images, masks, clinical_datas):        
        
        h = self.patch_embed(images)
        h = h + self.pos_embed
        hm = self.patch_embed_mask(masks)
        hm = hm + self.pos_embed_m
        # h=h+hm
        # hc = self.proj_clinical(clinical_datas.unsqueeze(-1)) #.expand(-1, 64, -1)
        hc = self.proj_clinical(torch.matmul(clinical_datas.unsqueeze(-1),clinical_datas.unsqueeze(-2)))
        # print(hc.shape)
        # h=h+hc
        # h = torch.cat((h,self.proj_clinical(clinical_datas).unsqueeze(1)),1)
        g_lst = []
        for idx in range(h.shape[0]):
            g = dgl.graph(([], []))
            g = g.to(self.device)
            # g.add_nodes(65)
            g.add_nodes(16)
            g.ndata['feat'] = h[idx]
            g.add_edges(self.src_list, self.dst_list)
            e = self.eproj(torch.cat((hm[idx][self.src_list],hm[idx][self.dst_list]),-1))
            # g.add_edges(self.src_list+[64]*64, self.dst_list+list(range(64)))
            # g.edata['feat'] = self.proj_clinical(clinical_datas[idx].unsqueeze(0).expand(288, -1))
            g.edata['feat']=e
            g_lst.append(g)
        batched_graph = dgl.batch(g_lst)
        # e = batched_graph.edata['feat']
        h = batched_graph.ndata['feat']
        e = batched_graph.edata['feat']

        # gm_lst = []
        # for idx in range(hm.shape[0]):
        #     gm = dgl.graph(([], []))
        #     gm = gm.to(self.device)
        #     gm.add_nodes(64)
        #     gm.ndata['feat'] = hm[idx]
        #     gm.add_edges(self.src_list, self.dst_list)
        #     gm_lst.append(gm)
        # batched_graph_mask = dgl.batch(gm_lst)
        # hm = batched_graph_mask.ndata['feat']        
        
        # convnets
        # for conv, sml, cml, convm in zip(self.layers,self.self_layers,self.cross_layers, self.mlayers):
        for conv, sml, cml, pproj in zip(self.layers,self.self_layers,self.cross_layers, self.proj_layers):
        # for conv in self.layers:
            # h = conv(batched_graph, h)
            h,e = conv(batched_graph, h,e)
            # hm = convm(batched_graph_mask, hm)
            # h=h+hm
            hc, _ = sml(hc,hc,hc)
            hc, _ = cml(pproj(h.view(-1,16,768)),hc,hc)
            # print(ic_attn.shape)

        # # output
        # h=self.MLP_layer_0(h.view(-1,64,768).transpose(1, 2)).squeeze(-1)
        # h_out = self.MLP_layer(h)
        e=self.pooling_proj_e(self.pooling(e.view(-1,48,768)))
        h=self.pooling_proj(h.view(-1,16,768))
        hc=self.hc_pooling(hc)
        h = self.mamba(torch.cat((h,e,hc),1))
        # h=self.mamba(h.view(-1,65,768))
        # h=self.mamba(h.view(-1,64,768))
        # h=self.pooling(h.unsqueeze(1))
        h=self.MLP_layer_0(h.transpose(1, 2)).squeeze(-1)
        h_out = self.MLP_layer(h.squeeze(1).squeeze(-2))
        
        return h_out, h.squeeze(1).squeeze(-2)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    