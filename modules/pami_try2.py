from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.distributed as dist
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from lavis.models import load_model_and_preprocess
from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip
import numpy as np
from modules.module_clip import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
import torch.nn as nn
import torch.nn.functional as F
from modules.inference_utils import load_model, load_image, predict, annotate
import nltk
from nltk.corpus import stopwords

import torch
from transformers import AutoImageProcessor, AutoModel, pipeline
import geoopt

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
logger = logging.getLogger(__name__)
allgather = AllGather.apply
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from mamba_ssm import Mamba

pair = lambda x: x if isinstance(x, tuple) else (x, x)
from detr.models.backbone import Backbone, Joiner
from detr.models.detr import DETR, PostProcess
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.segmentation import DETRsegm, PostProcessPanoptic
from detr.models.transformer import Transformer
logger = logging.getLogger(__name__)
allgather = AllGather.apply
def iou_loss(pred, target, eps=1e-7):
    """
    pred, target: [N,4] in cxcywh format (normalized or absolute, but consistent)
    Returns scalar IoU loss
    """
    pred = box_cxcywh_to_xyxy(pred)
    target = box_cxcywh_to_xyxy(target)

    # intersection
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # union
    area1 = box_area(pred)
    area2 = box_area(target)
    union = area1 + area2 - inter + eps

    iou = inter / union
    loss = 1.0 - iou
    return loss.mean()

import pickle as pkl
f = open("/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/ravar/benchmarks/exp_new_model/try_2_2_sem/modules/worlds_feature.pkl", "rb")
worlds_feature = pkl.load(f)
f.close()

def box_cxcywh_to_xyxy(box):
    # (cx,cy,w,h) -> (x1,y1,x2,y2)
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)
bce = F.binary_cross_entropy_with_logits
def auc_rank_loss(logits, y, margin=0.2):
    pos = logits[y==1]; neg = logits[y==0]
    if len(pos)==0 or len(neg)==0: 
        return logits.new_tensor(0.)
    # hard negatives
    k = min(64, neg.numel())
    hard_neg, _ = torch.topk(neg, k=k)
    diff = pos.unsqueeze(1) - hard_neg.unsqueeze(0) - margin
    return torch.log1p(torch.exp(-diff)).mean()

def focal_binary_cross_entropy(logits, targets, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = loss.mean()
    return loss


def focal_binary_cross_entropy_label_smooth(
    logits, targets, gamma=2.0, alpha=None, label_smoothing=0.3
):
    # flatten
    l = logits.reshape(-1).float()
    t = targets.reshape(-1).float()
    T = 0.98
    # 1) label smoothing: 1 -> 1-ε, 0 -> ε
    if label_smoothing > 0.0:
        eps = label_smoothing
        t = t * (1.0 - eps) + (1.0 - t) * eps

    # 2) standard BCE (logits version for stability), no reduction yet
    bce = F.binary_cross_entropy_with_logits(l/T, t, reduction="none")

    # 3) focal modulating factor; p = sigmoid(l), p_t = p for positives else 1-p
    p = torch.sigmoid(l)
    p_t = torch.where(t >= 0.5, p, 1.0 - p)
    focal_factor = (1.0 - p_t).pow(gamma)

    loss = focal_factor * bce

    # 4) optional class weighting alpha (weight positives by alpha, negatives by 1-alpha)
    if alpha is not None:
        alpha_t = torch.where(t >= 0.5, torch.as_tensor(alpha, device=l.device), 1.0 - torch.as_tensor(alpha, device=l.device))
        loss = alpha_t * loss

    return loss.mean()


def _make_detr(backbone_name: str, dilation=False, num_classes=91, mask=False):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    detr = DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=100)
    if mask:
        return DETRsegm(detr)
    return detr
def detr_resnet50(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    DETR R50 with 6 encoder and 6 decoder layers.

    Achieves 42/62.4 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet50", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def pdist(a, b):
    aa = (a*a).sum(-1, keepdim=True)
    bb = (b*b).sum(-1, keepdim=True)
    return aa + bb.transpose(-2,-1) - 2*a@b.transpose(-2,-1)

def mmd_rbf(x, y, sigma=1.0):
    X = x.reshape(-1, x.size(-1))     # [(B*T), D]
    Y = y.reshape(-1, y.size(-1))
    Kxx = torch.exp(-pdist(X,X)/(2*sigma**2)).mean()
    Kyy = torch.exp(-pdist(Y,Y)/(2*sigma**2)).mean()
    Kxy = torch.exp(-pdist(X,Y)/(2*sigma**2)).mean()
    return Kxx + Kyy - 2*Kxy


class SideAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=128):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden_size)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=4, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        #self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
        #                     padding=1, groups=dim)
        '''self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, window))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))'''
        '''trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)'''
        pool_size = int(agent_num ** 0.5)
        #self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.pool = nn.Sequential(nn.Linear(dim, int(dim//2)), nn.GELU(), nn.Linear(int(dim//2), self.num_heads*self.agent_num*dim))
    def forward(self, x, attn_1=None, attn_2=None, agent_tk1=None, agent_tk2=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).contiguous().reshape(b, n, 3, c).contiguous().permute(2, 0, 1, 3).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c
        agent_tokens = self.pool(q)
        #print(agent_tokens.shape)
        q = q.reshape(b, n, num_heads, head_dim).contiguous().permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, n, num_heads, head_dim).contiguous().permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, n, num_heads, head_dim).contiguous().permute(0, 2, 1, 3).contiguous()
        agent_tokens = agent_tokens.mean(1).contiguous().reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        #position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window), mode='bilinear')
        #position_bias1 = position_bias1.contiguous().reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1).contiguous()
        #position_bias2 = (self.ah_bias + self.aw_bias).contiguous().reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1).contiguous()
        #position_bias = position_bias1 + position_bias2
        #position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.contiguous().transpose(-2, -1).contiguous())
        agent_rep = agent_attn
        if attn_1 != None and attn_2 != None:
            agent_attn = (agent_attn * torch.nn.functional.softmax(attn_1 + attn_2, -1) + agent_attn * torch.nn.functional.softmax(attn_1 + attn_2, 1) + agent_attn)/3
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v
        #agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        #agent_bias1 = agent_bias1.contiguous().reshape(1, num_heads, self.agent_num, -1).contiguous().permute(0, 1, 3, 2).contiguous().repeat(b, 1, 1, 1).contiguous()
        #agent_bias2 = (self.ha_bias + self.wa_bias).contiguous().reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1).contiguous()
        #agent_bias = agent_bias1 + agent_bias2
        #agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
        #print((q * self.scale) @ agent_tokens.contiguous().transpose(-2, -1).shape)
        if agent_tk1 != None and agent_tk2 != None:
            agent_tokens = (agent_tokens * torch.nn.functional.softmax(agent_tk1 + agent_tk2, -1) + agent_tokens * torch.nn.functional.softmax(agent_tk1 + agent_tk2, 1) + agent_tokens)/3
        q_attn = self.softmax((q * self.scale) @ agent_tokens.contiguous().transpose(-2, -1))
        
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v
        x = x.contiguous().transpose(1, 2).reshape(b, n, c)
        #v_ = v.contiguous().transpose(1, 2).contiguous().reshape(b, h, w, c).contiguous().permute(0, 3, 1, 2).contiguous()
        x = x #+ self.dwc(v_).contiguous().permute(0, 2, 3, 1).contiguous().reshape(b, n - 1, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, agent_rep, agent_tokens


class AgentBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.0, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 agent_num=4, window=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AgentAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   agent_num=agent_num, window=window)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_1=None, attn_2=None, agent_tk1=None, agent_tk2=None):
        att_x, att, at = self.attn(self.norm1(x), attn_1, attn_2, agent_tk1, agent_tk2)
        x = x + self.drop_path(att_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=1e-12, out=None)
        
        return x, att, at 

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


@torch.no_grad()
def gather_together(data): #封装成一个函数，，用于收集各个gpu上的data数据，并返回一个list
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return gather_data
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def gather(keys):
    # ....前面可以忽略，这里的keys可以看作negative_feat()
    keys = keys.detach().clone().cpu() # 先把数据移到cpu上
    gathered_list = gather_together(keys) # 进行汇总，得到一个list
    keys = torch.cat(gathered_list, dim=0).cuda()
    return keys 
    dist.gather(tensor, dst=root, group=group)

def get_similar_indexes(text_embeddings, image_embeddings):
    text_norm = F.normalize(text_embeddings, p=2, dim=2)           # [B, T, D]
    image_norm = F.normalize(image_embeddings, p=2, dim=3)         # [B, Z, I, D]
    B, Z, I, D = image_norm.shape
    # Expand text to align with Z
    text_expanded = text_norm.unsqueeze(2).expand(-1, -1, Z, -1)   # [B, T, Z, D]
    image_expanded = image_norm.permute(0, 2, 1, 3)                # [B, I, Z, D]
    image_expanded = image_expanded.permute(0, 2, 1, 3)            # [B, Z, I, D]

    # Compute cosine similarity per time step: [B, T, Z, I]
    #print(text_expanded.shape)
    #print(image_expanded.shape)
    cos_sim = torch.einsum("btzd,bzid->btzi", text_expanded, image_expanded)

    # For each (B, T, Z), get the best matching image token (max over I)
    max_sim_values, max_sim_indices = torch.max(cos_sim, dim=3)  # [B, T, Z], [B, T, Z]


    """
    for b in range(B):
        print(f"\nBatch {b}:")
        for t in range(T):
            print(f"  Text token {t} → Image token {max_sim_indices[b, t].item()} (sim={max_sim_values[b, t].item():.4f})")
    """

    return max_sim_indices

def get_matching_image_tokens(image_embeddings, max_sim_indices):
    """
    image_embeddings: [B, Z, I, D]
    max_sim_indices: [B, T, Z] — best image token index at each [B, T, Z]
    Returns:
        selected_tokens: [B, T, Z, D] — most similar image token embeddings
    """
    B, Z, I, D = image_embeddings.shape
    _, T, _ = max_sim_indices.shape

    # Expand image_embeddings: [B, 1, Z, I, D] → for broadcasting
    image_exp = image_embeddings.unsqueeze(1).expand(B, T, Z, I, D)  # [B, T, Z, I, D]

    # Expand indices to match shape for gather
    index_exp = max_sim_indices.unsqueeze(-1).unsqueeze(-1)         # [B, T, Z, 1, 1]
    index_exp = index_exp.expand(-1, -1, -1, 1, D)                   # [B, T, Z, 1, D]

    # Gather along dim=3 (I dimension)
    selected_tokens = torch.gather(image_exp, dim=3, index=index_exp)  # [B, T, Z, 1, D]

    # Remove the singleton dimension
    selected_tokens = selected_tokens.squeeze(3)  # [B, T, Z, D]

    return selected_tokens


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_queries: int = 6):
        """
        num_queries: if > 0, create learnable queries of shape [num_queries, D]
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.num_queries = num_queries
        if num_queries > 0:
            # Learned query tokens (broadcasted across the batch)
            self.query_tokens = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
    @torch.no_grad()
    def _make_learned_queries(self, B, device, dtype):
        # [1, NQ, D] -> [B, NQ, D]
        return self.query_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    def forward(self, context, query=None, append_learned: bool = False, return_split: bool = False):
        """
        context: [B, T_kv, D]  (e.g., features_image)
        query:   [B, T_q,  D]  (optional external queries)
        
        Modes:
          - If query is None and num_queries>0: use learned queries only -> returns [B, NQ, D]
          - If query is provided and append_learned=False: standard cross-attn on 'query'
          - If query is provided and append_learned=True and num_queries>0:
                concatenate [query, learned_queries] as the attention queries,
                and (optionally) return the split (external vs learned) if return_split=True.
        """
        B, _, D = context.shape
        device, dtype = context.device, context.dtype

        learned_q = None
        if self.num_queries > 0:
            learned_q = self._make_learned_queries(B, device, dtype)

        if query is None:
            assert learned_q is not None, "No query provided and num_queries=0; nothing to attend with."
            q = learned_q
            split_sizes = None
        elif append_learned and learned_q is not None:
            q = torch.cat([query, learned_q], dim=1)      # [B, T_q + NQ, D]
            split_sizes = (query.size(1), learned_q.size(1))
        else:
            q = query
            split_sizes = None

        # Cross-attention: queries attend to the context (keys/values)
        attn_output, _ = self.attn(query=q, key=context, value=context)

        # Add & norm
        x = self.norm1(q + attn_output)

        # Feedforward with residual
        x = self.norm2(x + self.ff(x))

        if return_split and split_sizes is not None:
            x_query, x_learned = torch.split(x, split_sizes, dim=1)
            return x_query, x_learned   # external-query outputs, learned-query outputs
        return x

class CrossAttentionBlock2(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        """
        query:   [B, T_q, D]  (agg_trajs)
        context: [B, T_kv, D] (features_image)
        returns: [B, T_q, D]
        """
        # Apply multi-head attention
        attn_output, _ = self.attn(query=query, key=context, value=context)

        # Add & norm
        x = self.norm1(query + attn_output)

        # Feedforward with residual
        x = x + self.ff(x)
        x = self.norm2(x)
        return x
def irm_penalty(logits, y):
    # logits from a scalar "dummy" classifier w*phi(x); assume cross-entropy
    import torch
    scale = torch.ones(1, requires_grad=True, device=logits.device)
    loss = focal_binary_cross_entropy(logits*scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad**2).sum()

class ClassCentersEMA(torch.nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9, normalize: bool = True, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.normalize = normalize

        centers = torch.zeros(num_classes, feat_dim, dtype=torch.float32, device=device)
        self.register_buffer("centers", centers)          # [C, D]
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool, device=device))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, targets: torch.Tensor):
        """
        feats:   [B, D] features from the backbone/head (optionally L2-normalized)
        targets: [B, C] multi-hot (0/1) labels
        """
        assert feats.dim() == 2 and targets.dim() == 2
        B, D = feats.shape
        C = targets.shape[1]
        assert C == self.num_classes and D == self.feat_dim

        if self.normalize:
            feats = F.normalize(feats, dim=1)

        # counts per class: [C]
        counts = targets.sum(dim=0)                       # how many positives per class in this batch

        # class sums via matrix multiply: [C, D]
        # (targets^T) @ feats computes sum of features for each class across positives
        class_sums = targets.t().matmul(feats)            # [C, D]

        # indices of classes that appear in the batch
        mask = counts > 0
        if mask.any():
            # compute means only for present classes
            means = torch.zeros_like(self.centers)
            means[mask] = class_sums[mask] / counts[mask].unsqueeze(1)

            # initialize unseen classes on their first appearance to the mean
            new_classes = (~self.initialized) & mask
            self.centers[new_classes] = means[new_classes]
            self.initialized[new_classes] = True

            # EMA update for seen classes
            m = self.momentum
            self.centers[mask] = m * self.centers[mask] + (1 - m) * means[mask]

            if self.normalize:
                self.centers[mask] = F.normalize(self.centers[mask], dim=1)

        # classes with no positives: skip update
        return self.centers
def flatten_ntd(x):
    """
    x: [N,T,D] -> [N*T, D]
    """
    return x.reshape(-1, x.size(-1))

def loss_maximize_mmd_seq(x, y):
    """
    x: [N,T,D], y: [M,T,D] or [M,T',D]
    Compare as distributions (ignoring temporal alignment).
    """
    x_flat = flatten_ntd(x)
    y_flat = flatten_ntd(y)
    return loss_maximize_mmd(x_flat, y_flat)   # from my earlier message

class Blipv2(nn.Module):
    def __init__(self,):
        super(Blipv2, self).__init__()
        self.modelblip, self.vis_processors, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=False)
        '''for block in self.modelblip.visual_encoder.blocks:
            block.side_adapter = SideAdapter(block.mlp.fc1.in_features)
            old_forward = block.forward
            def new_forward(self, x, *args, **kwargs):
                x = old_forward(x, *args, **kwargs)
                return self.side_adapter(x)
            block.forward = new_forward.__get__(block, block.__class__)
        for name, param in self.modelblip.named_parameters():
            if "side_adapter" not in name:
                param.requires_grad = False'''
        self.proj_box = nn.Linear(100,32)
        self.proj_sem = nn.Linear(1, 32)
        self.proj = nn.Linear(772,768)
        self.agent_temporal = AgentBlock(dim=768, num_heads=1, window=8) #AgentAttention()
        self.grounding_dino = load_model("/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/ravar/benchmarks/exp_new_model/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/ravar/benchmarks/exp_new_model/GroundingDINO/weights/groundingdino_swint_ogc.pth")
        '''self.dinov3 =  pipeline(
                model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                task="image-feature-extraction", 
            )'''
        self.llm_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True,
        )
        self.proj_text = nn.Linear(2560,768)
        self.detection_model =  detr_resnet50(pretrained=True).eval()#torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True, force_reload=True)
        for param in self.detection_model.parameters():
            param.requires_grad = False
        #self.bbox_regression_head = nn.Sequential(nn.Linear(768, 4))
        
        
        self.bbox_regression_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128,4), nn.Sigmoid())
        self.classification_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 80))

        self.bbox_regression_head2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128,4), nn.Sigmoid())
        self.classification_head2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 80))




        self.mamba_fusion_temp = nn.Sequential(nn.Linear(768, 256), Mamba(d_model=256), Mamba(d_model=256)) #Mamba(d_model=768)
        self.mamba_fusion = nn.Sequential(Mamba(d_model=768)) #Mamba(d_model=768)
        self.mamba_image = nn.Sequential(nn.Linear(768, 256), Mamba(d_model=256), Mamba(d_model=256))
        self.manifold = geoopt.manifolds.PoincareBall(c=1)

        self.bceloss = torch.nn.BCEWithLogitsLoss()
        self.mseloss = torch.nn.MSELoss()
        self.cross_attention = CrossAttentionBlock(embed_dim=256, num_heads=8)
        self.cross_attention_2 = CrossAttentionBlock(embed_dim=256, num_heads=8)
        self.cross_attention_box = CrossAttentionBlock(embed_dim=256, num_heads=8)

        self.cross_attention_sem = CrossAttentionBlock(embed_dim=256, num_heads=8)
        self.cross_attention_sem2 = CrossAttentionBlock(embed_dim=256, num_heads=8)
        self.cross_attention_box2 = CrossAttentionBlock(embed_dim=256, num_heads=8)
        self.centers = ClassCentersEMA(num_classes=80, feat_dim=768, momentum=0.9, normalize=True)
        self.proj_img = nn.Linear(768, 256)
        self.proj_boxes = nn.Linear(768, 256)
        self.proj_text2 = nn.Linear(768, 256)


        self.latent_proj = nn.Linear(256, 256)
        self.latent_proj2 = nn.Linear(256, 256)



    def center_pull_loss(self, feats, targets, reduce='mean'):
        """
        feats:   [B, D] embeddings
        targets: [B, C] multi-hot {0,1}
        centers: [C, D] prototypes (buffer, no grad)
        """
        centers = self.centers.centers

        feats = F.normalize(feats, dim=1)
        centers_n = F.normalize(centers.detach(), dim=1)

        # cosine similarity: [B, C]
        sim = feats @ centers_n.T

        # (1 - cos) for positives
        pos_mask = targets.bool()
        pos_terms = (1.0 - sim) * targets  # [B, C], zero if not positive

        # per-sample average across positive classes
        counts = targets.sum(dim=1).clamp_min(1)  # [B]
        per_sample = pos_terms.sum(dim=1) / counts

        if reduce == 'mean':
            # only average over samples that had ≥1 positive label
            mask = targets.sum(dim=1) > 0
            return per_sample[mask].mean() if mask.any() else feats.new_tensor(0.)
        else:
            return per_sample

    def forward(self, key_frame, input_ids, token_type_ids, attention_mask, video, video_mask=None, bbox=None, ann=None, training=True):
        worlds = worlds_feature.cuda()
        text_emb_list = []
        video = torch.as_tensor(video).float()
        if len(video.shape) == 5:
            video = video.unsqueeze(1).unsqueeze(1)
        else:
            video = video.unsqueeze(1)
        
        b, pair, bs, ts, channel, h, w = video.shape


        detect_results = self.detection_model(key_frame.squeeze())

        preds = detect_results["pred_logits"] # B,Q,N
        boxes = detect_results["pred_boxes"] # B,Q,N
        B = preds.shape[0]

        categories = torch.argmax(torch.nn.functional.softmax(preds,-1),-1).flatten(0,1)-1

        w_embs = torch.index_select(worlds, 0, categories)
        w_embs = w_embs.contiguous().view(B, 100, -1)
        boxes = torch.stack([boxes[...,0] - 0.5*boxes[...,3], boxes[...,1] - 0.5*boxes[...,2], boxes[...,0] + 0.5*boxes[...,3],boxes[...,1] + 0.5*boxes[...,2]],-1)

        bboxes = self.proj(torch.cat([boxes, w_embs], -1))
        mask = torch.argmax(preds,-1) != 1
        bboxes[mask] = 0.0* bboxes[mask]
        boxes = self.proj_box(bboxes.contiguous().permute(0,2,1)).contiguous().permute(0,2,1)

        #print(boxes.shape)

        video = video.contiguous().view(b * pair * bs* ts, channel, h, w)
        video_frame = bs * ts
        #with torch.no_grad():
        #print(b)
        #print(input_ids)
        with torch.no_grad():
            text = []
            for i in range(b):
                for j in range(ts):
                    #print(i)
                    text.append("Predict the fine-grained action of " + input_ids[i])
            sample = {"image": video.half().cuda(), "text_input": text}
            features_text = self.modelblip.extract_features(sample, mode="text") # batch_size, 12, 768

            features_text = features_text.text_embeds.contiguous().view(b, ts, -1, 768).mean(2)
            features_image = self.modelblip.extract_features(sample, mode="image") # batch_size, 8, 32, 768

        features_image = features_image.image_embeds.contiguous().view(b, ts, -1, 768)
        sem_trajs = []

        for sind, sentence in enumerate(input_ids):
            # 1. Remove stopwords
            
            image_tokens = features_image[sind].unsqueeze(0)

            words = sentence.strip().split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            filtered_sentence = ' '.join(filtered_words)
            tokenized = self.llm_model.tokenizer(
                filtered_sentence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=30,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            tokenized = {k: v.cuda() for k, v in tokenized.items()}
            with torch.no_grad():
                #with torch.no_grad():
                output = self.llm_model.opt_model(**tokenized, output_hidden_states=True, return_dict=True)
            last_hidden = output.hidden_states[-1]  # shape: (1, seq_len, hidden_dim)
            # 4. Decode tokens
            tokens = self.llm_model.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
            # 5. Filter out padding/special tokens
            valid_mask = tokenized["attention_mask"][0].bool()
            valid_tokens = [tok for i, tok in enumerate(tokens) if valid_mask[i]]
            valid_embeddings = last_hidden[0][valid_mask]
            #print(valid_embeddings.shape)
            # 6. Store results
            #results.append(self.proj_text(valid_embeddings))
            desire_text_tokens = self.proj_text(valid_embeddings.unsqueeze(0)) # 1,T,D

            indexes = get_similar_indexes(desire_text_tokens,image_tokens)
            image_trajectories = get_matching_image_tokens(image_tokens, indexes)
            #print(image_trajectories.shape)
            B,S,T,D = image_trajectories.shape
            sem_trajs.append(self.mamba_fusion(image_trajectories.view(B, -1, D)).contiguous().view(B,S,T,D).mean(1))

        sem_trajs = torch.cat(sem_trajs,0)
        agg_trajs = self.mamba_fusion_temp(sem_trajs) # B,T,D
        features_image_bboxes = self.proj_img(features_image.mean(2))
        #print(features_image_bboxes.shape)
        features_image = features_image.mean(1) # B,T,D
        features_image = self.mamba_image(features_image)
        bboxes = self.proj_boxes(boxes)
        features_text = self.proj_text2(features_text)





        output = self.cross_attention_box(bboxes, features_image_bboxes).mean(1) +  self.cross_attention(agg_trajs, features_image).mean(1) + self.cross_attention_sem(features_text, features_image).mean(1) #+ self.proj_feature_text(features_text) # [B, T, D]
        output_2 = self.cross_attention_box2(bboxes, features_image_bboxes).mean(1) + self.cross_attention_2(agg_trajs, features_image).mean(1) + self.cross_attention_sem2(features_text, features_image).mean(1) #+ self.proj_feature_text_2(features_text)

        #output = self.manifold.expmap0(output)
        #output_2 = self.manifold.expmap0(output_2)

        '''with torch.no_grad():
            self.centers.update(output.detach(), ann.float().cuda())'''


        residual_boxes = self.bbox_regression_head(output_2)#pred_boxes.cuda()
        cls_results = self.classification_head(output)




        output_2_2 = self.latent_proj2(output_2)
        residual_boxes2 = self.bbox_regression_head2(output_2_2)
        output_1_2 = self.latent_proj(output)
        cls_results2 = self.classification_head2(output_1_2)

        mmd_loss = 1-mmd_rbf(output_2_2, output_1_2)



        #print(predictions[0])
        #print(ann[0])
        if training:
            #predictions = gather(predictions)
            #ann = gather(ann)
            #print(cls_results.shape)
            #print(ann.shape)
            #ann = ann[:,0]
            loss_cls = focal_binary_cross_entropy_label_smooth(cls_results, ann.float().cuda()) + auc_rank_loss(cls_results, ann.float().cuda()) + irm_penalty(cls_results, ann.float().cuda()) - 0.1*F.binary_cross_entropy_with_logits(cls_results, 1-ann.float().cuda())
            loss_bbox = self.mseloss(residual_boxes, bbox.float().cuda()) + iou_loss(residual_boxes, bbox.float().cuda())

            loss_cls2 = focal_binary_cross_entropy_label_smooth(cls_results2, ann.float().cuda()) + auc_rank_loss(cls_results2, ann.float().cuda()) + irm_penalty(cls_results2, ann.float().cuda()) - 0.1*F.binary_cross_entropy_with_logits(cls_results2, 1-ann.float().cuda()) 
            loss_bbox2 = self.mseloss(residual_boxes2, bbox.float().cuda()) + iou_loss(residual_boxes2, bbox.float().cuda())
            


            loss = 1.5*torch.mean(loss_bbox) + torch.mean(loss_cls) + 2*torch.mean(loss_bbox2) + torch.mean(loss_cls2) + mmd_loss  #+ self.center_pull_loss(output, ann.float().cuda())
            #print('bbox_loss:', torch.mean(loss_bbox))
            #print('cls_loss:', torch.mean(loss_cls))
        else:
            loss = 0.0


        return loss, (cls_results +cls_results2)/2, (residual_boxes + residual_boxes2)/2, ann

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


import math


def compute_loss(target_bboxes, pred_bboxes):
    """Reference: https://arxiv.org/pdf/1911.08287.pdf
    Args:
        target_bboxes (torch.Tensor): Target boundong boxes, [N, H, W, 4]
        pred_bboxes (torch.Tensor): Predicted bounding boxes, [N, H,W, 4]
    """
    # Compute intersections
    x1 = torch.max(target_bboxes[..., 0], pred_bboxes[..., 0])
    y1 = torch.max(target_bboxes[..., 1], pred_bboxes[..., 1])
    x2 = torch.min(target_bboxes[..., 2], pred_bboxes[..., 2])
    y2 = torch.min(target_bboxes[..., 3], pred_bboxes[..., 3])

    intersects = torch.clamp((x2-x1), min=0.0) * torch.clamp((y2-y1), min=0.0)

    # Compute unions
    A = abs((target_bboxes[..., 2]-target_bboxes[..., 0]) * target_bboxes[..., 3]-target_bboxes[..., 1])
    B = abs((pred_bboxes[..., 2]-pred_bboxes[..., 0]) * pred_bboxes[..., 3]-pred_bboxes[..., 1])

    unions = A + B - intersects

    ious = intersects / unions

    cx1 = torch.min(target_bboxes[..., 0], pred_bboxes[..., 0])
    cy1 = torch.min(target_bboxes[..., 1], pred_bboxes[..., 1])
    cx2 = torch.max(target_bboxes[..., 2], pred_bboxes[..., 2])
    cy2 = torch.max(target_bboxes[..., 3], pred_bboxes[..., 3])

    # Compute Euclidean between central points and diagonal lenght
    c_dist = ((target_bboxes[..., 2] + target_bboxes[..., 0] - pred_bboxes[..., 2] - pred_bboxes[..., 0]) ** 2 + \
              (target_bboxes[..., 3] + target_bboxes[..., 1] - pred_bboxes[..., 3] - pred_bboxes[..., 1]) ** 2) / 4
    
    diagonal_l2 = (cx2-cx1) **2 + (cy2-cy1) ** 2

    # Postive trade-off parameter and asspect ratio
    with torch.no_grad():
        v = (4/math.pi**2) * torch.pow((torch.atan((target_bboxes[..., 2]-target_bboxes[..., 0])/(target_bboxes[..., 3]-target_bboxes[..., 1]))- \
            torch.atan((pred_bboxes[..., 2]-pred_bboxes[..., 0])/(pred_bboxes[..., 3]-pred_bboxes[..., 1]))), 2)
        alpha = v / (1 - ious + v)

    cious = ious - (c_dist / diagonal_l2 + alpha * v)
    cious = torch.clamp(cious, min=-1.0, max=1.0)

    return cious