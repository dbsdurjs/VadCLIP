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

class CLIPVAD(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.num_class = args.classes_num
        self.visual_length = args.visual_length  # 256
        self.visual_width = args.visual_width    # 512
        self.visual_layers = args.visual_layers
        self.visual_head = args.visual_head
        self.embed_dim = args.embed_dim
        self.attn_window = args.attn_window
        self.prompt_prefix = args.prompt_prefix
        self.prompt_postfix = args.prompt_postfix
        self.batch_size = args.batch_size # add cls token
        self.lstm_nlayers = args.lstm_layer
        self.cross_attn_head = args.cross_attn_head
        self.text_head = args.text_head
        self.text_dim = args.text_dim
        self.text_layers = args.text_layers
        self.device = device

        self.lstm_h_size = 512

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.visual_width, self.visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.visual_width * 4, self.visual_width))
        ]))

        self.classifier = nn.Linear(512, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.lstm = nn.LSTM(self.visual_width, hidden_size=self.lstm_h_size//2, num_layers=self.lstm_nlayers, bidirectional=True, dropout=0.3) # 단방향 먼저, 양방향(output shape = hidden size *2)
        self.lstmnorms = nn.LayerNorm(self.lstm_h_size)

        self.frame_position_embeddings = nn.Embedding(self.visual_width+1, self.visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.temporal = Transformer(
            width=self.visual_width,
            layers=self.visual_layers,
            heads=self.visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )
        self.cls_embeddings_visual = nn.Parameter(torch.randn(1, 1, self.visual_width)) # add cls token (batch*2, 1, 512)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

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

    def encode_video_lstm(self, images, cls_token_vis):
        images = images.to(torch.float) # (batch size, 256, 512)
        position_ids = torch.arange(self.visual_length+1, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)    # (batch size,256+1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)    # (batch size, 256+1, 512)
        cls_token_vis = cls_token_vis + frame_position_embeddings[:, 0].unsqueeze(1)
        images = images.permute(1, 0, 2) + frame_position_embeddings[:, 1:].permute(1, 0, 2) # (256, batch, 512)

        lstm_output, (_, _) = self.lstm(images)
        out = self.lstmnorms(lstm_output.permute(1, 0, 2)) + images.permute(1, 0, 2) # (batch, 256, 512)

        encoder_out, _ = self.temporal((out.permute(1, 0, 2), None)) # (256, batch, 512)
        output = encoder_out.permute(1, 0, 2) + out
        x = torch.cat((cls_token_vis, output), dim=1)
         
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
    
    def forward(self, visual, padding_mask, text): 
        cls_token_vis = repeat(self.cls_embeddings_visual, '() n d -> b n d', b = visual.shape[0]) # (batch, 1, 512)

        avg_vis = visual.mean(dim=1, keepdim=True)      # (batch, 1, D)

        cls_token_vis = cls_token_vis + avg_vis

        visual_features = self.encode_video_lstm(visual, cls_token_vis)  # LGT Adapter(clip img features), torch.Size([batch, 256+1, 512])

        logits1 = self.classifier(visual_features + self.mlp1(visual_features)) # A = Sigmoid(FC(FFN(X) + X)), (batch, 256, 1)

        text_features_ori = self.encode_textprompt(text)    # clip text encoder(learnable prompt + te                      xt), (14,77, 512) -> (14, 512)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)  # (batch, 1, 256)
        visual_attn = logits_attn @ visual_features # aggregate(visual features, logits1), (batch, 1, 512)
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)  # aggregate(visual features, logits1)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])    # (batch, 7, 512)
        
        text_features = text_features_ori.unsqueeze(0)  # (1, 7, 512)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2]) # (batch, 7, 512)
        text_features = text_features + visual_attn # visual prompt(vision + Text)
        text_features = text_features + self.mlp1(text_features) # label features = visual prompt(ffn(text features) + text features), (batch, 7, 512)
        
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True) # (batch, 256, 512)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)    # (batch, 512, 7)
        
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07 #(batch, 256, 7)

        return text_features_ori, logits1, logits2
    