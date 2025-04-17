from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from collections import OrderedDict
from einops import repeat

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x # padding_maks : None(default)
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, fusion_dim=512, num_heads=8, dropout=0.1, cross_attn_depth=1):
        super(CrossAttentionFusion, self).__init__()
        # 캡션 → 시각 Cross Attention
        self.cross_attn_cap = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout)
        # 시각 → 캡션 Cross Attention
        self.cross_attn_vis = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout)
        
        # LayerNorm
        self.norm_cap = nn.LayerNorm(fusion_dim)
        self.norm_vis = nn.LayerNorm(fusion_dim)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                self.cross_attn_cap,
                self.cross_attn_vis,
                self.norm_cap,
                self.norm_vis
            ]))

        self.mlp_head_cap = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim//2)
        )

        self.mlp_head_vis = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim//2)
        )

        self.mlp_head_fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim//2),
            nn.Linear(fusion_dim//2, fusion_dim)
        )

    def forward(self, caption_feat, visual_feat):
        # Shape: (batch, 256, 512) for both caption_feat and visual_feat
        
        for c_to_v, v_to_c, norm_c, norm_v in self.cross_attn_layers:
            # 대표 토큰: 평균값 사용
            cap_class = caption_feat[:, 0]  # (batch, 512)
            vis_class = visual_feat[:, 0]   # (batch, 512)
            cap_patches = caption_feat[:, 1:]  # (batch, 256, 512)
            vis_patches = visual_feat[:, 1:]   # (batch, 256, 512)

            # 캡션 → 시각 Cross Attention
            cap_query = cap_class.unsqueeze(1)  # (batch, 1, 512)
            cap_qkv = norm_c(torch.cat((cap_query, vis_patches), dim=1)) # (batch, 256+1, 512)
            attn_out_cap, _ = c_to_v(cap_query.permute(1, 0, 2), cap_qkv.permute(1, 0, 2), cap_qkv.permute(1, 0, 2))
            cap_out = attn_out_cap.permute(1, 0, 2) + cap_query
            fusion_cap = torch.cat((cap_patches, cap_out), dim=1)

            # 시각 → 캡션 Cross Attention
            vis_query = vis_class.unsqueeze(1)  # (batch, 1, 512)
            vis_qkv = norm_v(torch.cat((vis_query, cap_patches), dim=1)) # (batch, 256+1, 512)
            attn_out_vis, _ = v_to_c(vis_query.permute(1, 0, 2), vis_qkv.permute(1, 0, 2), vis_qkv.permute(1, 0, 2))
            vis_out = attn_out_vis.permute(1, 0, 2) + vis_query
            fusion_vis = torch.cat((vis_patches, vis_out), dim=1)

        caption_features = self.mlp_head_cap(fusion_cap)
        visual_features = self.mlp_head_vis(fusion_vis)

        fusion_features = caption_features + visual_features
        fusion_features = self.mlp_head_fusion(fusion_features)

        return fusion_features

class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 batch_size : int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length  # 256
        self.visual_width = visual_width    # 512
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.batch_size = batch_size # add cls token
        self.device = device

        self.temporal = Transformer(
            width=visual_width, # 512
            layers=visual_layers, # ucf-2
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))

        self.temp = nn.Linear(1024, 512)
        self.classifier = nn.Linear(512, 1)
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length+1, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.caption_embeddings = nn.Embedding(visual_length+1, visual_width) # add idea66-6
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=visual_width, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)

        self.cls_embeddings_caption = nn.Parameter(torch.randn(1, 1, self.visual_width)) # add cls token (batch*2, 1, 512)
        self.cls_embeddings_visual = nn.Parameter(torch.randn(1, 1, self.visual_width)) # add cls token (batch*2, 1, 512)

        self.crossfusion = CrossAttentionFusion(fusion_dim=visual_width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        nn.init.normal_(self.caption_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, cls_token_vis):  # LGT Adapter
        images = images.to(torch.float) # (batch size, 256, 512)
        position_ids = torch.arange(self.visual_length+1, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)    # (batch size,256+1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)    # (batch size, 256+1, 512)
        cls_token_vis = torch.cat((cls_token_vis, images), dim=1)
        image_feat = frame_position_embeddings + cls_token_vis

        return image_feat
    
    def lgt_adapter(self, images, lengths):  # LGT Adapter
        x, _ = self.temporal((images[:, 1:].permute(1, 0, 2), None))    # local module(clip img features)
        x = x.permute(1, 0, 2)

        # global module()
        adj = self.adj4(x, lengths) # H_sim
        disadj = self.disAdj(x.shape[0], x.shape[1])    # H_dis
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)  # X_g -> X

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)   # 클래스 토큰 생성, tokenizer(label), (14,77)
        word_embedding = self.clipmodel.encode_token(word_tokens)   # 클래스 토큰 임베딩, (14,77,512)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])  # (14,77,512)
        text_tokens = torch.zeros(len(text), 77).to(self.device)    # (14, 77)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)  # 제일 큰 값을 가지는 인덱스 추출(보통 EOT값)
            text_embeddings[i, 0] = word_embedding[i, 0]    # 시작 토큰 배치
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]    # 11~10+ind까지는 클래스 임베딩 사용
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind] # 20 + ind에 클래스 임베딩의 max 토큰(보통 EOT) 사용
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]    # max 토큰 이외에는 0으로 지정
            # 논문에서는 20개의 learnable prompt를 사용한다고 했지만 실제로는 77개 사용
            # 아래와 같이 EOT 토큰 이후의 값을 0으로 설정해서 사용하지 않았지만 성능 변화는 없었음
            # text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix + 1:] = 0

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)    # (14,512)

        return text_features

    def encode_caption(self, caption, cls_token_cap):
        caption = caption.to(torch.float) # (batch size, 256, 512)
        position_ids = torch.arange(self.visual_length+1, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(caption.shape[0], -1)    # (batch size, 256+1)
        frame_position_embeddings = self.caption_embeddings(position_ids)    # (batch size, 256+1, 512)
        cls_token_cap = torch.cat((cls_token_cap, caption), dim=1) # add cls token
        caption_feat = frame_position_embeddings + cls_token_cap
        
        # x = self.transformer_encoder(caption_feat)

        return caption_feat
    
    def forward(self, visual, captioning, padding_mask, text, lengths, cap_lengths): 
        cls_token_cap = repeat(self.cls_embeddings_caption, '() n d -> b n d', b = captioning.shape[0]) # (batch, 1, 512)
        cls_token_vis = repeat(self.cls_embeddings_visual, '() n d -> b n d', b = visual.shape[0]) # (batch, 1, 512)
        
        caption_features = self.encode_caption(captioning, cls_token_cap) # batch, 256+1, 512
        visual_features = self.encode_video(visual, cls_token_vis)  # LGT Adapter(clip img features), torch.Size([batch, 256+1, 512])

        fusion_feat = self.crossfusion(caption_features, visual_features)

        fusion_feat = self.lgt_adapter(fusion_feat, lengths)
        logits1 = self.classifier(fusion_feat + self.mlp1(fusion_feat)) # A = Sigmoid(FC(FFN(X) + X)), (batch, 256, 1)

        text_features_ori = self.encode_textprompt(text)    # clip text encoder(learnable prompt + text), (14,77, 512) -> (14, 512)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)  # (batch, 1, 256)
        visual_attn = logits_attn @ fusion_feat # aggregate(visual features, logits1), (batch, 1, 512)
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)  # aggregate(visual features, logits1)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])    # (batch, 7, 512)
        
        text_features = text_features_ori.unsqueeze(0)  # (1, 7, 512)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2]) # (batch, 7, 512)
        text_features = text_features + visual_attn # visual prompt(vision + Text)
        text_features = text_features + self.mlp1(text_features) # label features = visual prompt(ffn(text features) + text features), (batch, 7, 512)
        
        visual_features_norm = fusion_feat / fusion_feat.norm(dim=-1, keepdim=True) # (batch, 256, 512)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)    # (batch, 512, 7)
        
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / self.temperature #(batch, 256, 7)

        return text_features_ori, logits1, logits2, visual_features, caption_features
    