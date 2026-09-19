import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple

from monai.networks.layers import DropPath
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

class GuideDecoderLayer(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):

        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)


    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        txt = self.text_project(txt)

        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2,_ = self.cross_attn(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis

class GuideDecoder(nn.Module):

    def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:

        super().__init__()

        self.guide_layer = GuideDecoderLayer(in_channels,text_len)   # for skip
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')

    
    def forward(self, vis, skip_vis, txt):

        if txt is not None:
            vis =  self.guide_layer(vis, txt)

        vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)

        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')

        return output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def deconv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))
    
class Bridger(nn.Module):
    def __init__(self,
                 d_img = 512,
                 d_txt = 768,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = 2,
                 num_layers = 12,
                 stage_id = 1
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.stage_id = stage_id

        self.fusion_v = Interactor(d_model=d_model, nhead=nhead)
        self.fusion_t = Interactor(d_model=d_model, nhead=nhead)
        self.zoom_in = nn.Conv2d(d_img, d_model, kernel_size=strides, stride=strides, bias=False)
        if self.stage_id == 4:
            self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=strides, stride=strides, 
                                                padding=0, output_padding=1, bias=False)
        else:
            self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=strides, stride=strides, bias=False)
        self.linear1 = nn.Linear(d_txt, d_model)
        self.linear2 = nn.Linear(d_model, d_txt)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)

        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, vis, txt):
        # residual operation
        txt = txt.permute(1, 0, 2)  # NLD -> LND
        v = vis.clone()
        t = txt.clone()
        t = t.float()
        last_v, last_t = v, t
        # dimension reduction
        v = self.zoom_in(v)
        t = self.linear1(t)
        # multi modal fusion
        B, C, H, W = v.shape
        v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
        v, t = self.ln_v(v), self.ln_t(t)
        v, t = self.fusion_v(v, t), self.fusion_t(t, v)
        v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
        # dimension recovery
        v = self.zoom_out(v)                
        t = self.linear2(t)
        # residual connect
        vis = vis + v
        txt = txt + t   

        # After fusion
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        # txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # state = txt[torch.arange(txt.shape[0]),
                #   text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        return vis, txt


class Interactor(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1,
                 activation="relu", ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout) 

        self.activation = _get_activation_fn(activation)   

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,                
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # self attn
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B, C]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)      

        # cross attn                
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), 
                                            nn.BatchNorm2d(out_channels), 
                                            nn.ReLU(inplace=True))

        #下一段的eca原代码没有,但是论文中有  so新增
        # self.attention = ECA_ChannelAttention(in_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        #下一段的对传入的encoder(x2)  进行eca操作原代码没有,但是论文中有  so新增
        # x2 = self.attention(x2)


        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x  


class FusedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_img = 512, d_txt = 768, d_model = 64, 
                    nhead = 8, num_stages = 3, strides = 2, num_layers = 12):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers

        self.fusion_v = Interactor(d_model=d_model, nhead=nhead)
        self.fusion_t = Interactor(d_model=d_model, nhead=nhead)
        self.zoom_in = nn.Conv2d(d_img, d_model, kernel_size=strides, stride=strides, bias=False)
        self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=strides, stride=strides, bias=False)
        self.linear1 = nn.Linear(d_txt, d_model)
        self.linear2 = nn.Linear(d_model, d_txt)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)

        self.initialize_parameters()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), 
                                            nn.BatchNorm2d(out_channels), 
                                            nn.ReLU(inplace=True))

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, vis, txt, x2):
        # residual operation
        txt = txt.permute(1, 0, 2)  # NLD -> LND
        v = vis.clone()
        t = txt.clone()
        t = t.float()
        last_v, last_t = v, t
        # dimension reduction
        v = self.zoom_in(v)
        t = self.linear1(t)
        # multi modal fusion
        B, C, H, W = v.shape
        v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
        v, t = self.ln_v(v), self.ln_t(t)
        v, t = self.fusion_v(v, t), self.fusion_t(t, v)
        v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
        # dimension recovery
        v = self.zoom_out(v)                
        t = self.linear2(t)
        # residual connect
        vis = vis + v
        txt = txt + t   

        # After fusion
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        # txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # state = txt[torch.arange(txt.shape[0]),
                #   text.argmax(dim=-1)] @ backbone.text_projection

        # forward

        x1 = self.up(vis)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)

        return x, txt


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

            
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,window_size=(7, 7), num_heads=8):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 下一行是新增的CoordAttMeanMax
        self.ca = CoordAttMeanMax(inp=dim, oup=dim)  # 通道数保持 dim

        # 下一行是新增的ECA
        self.eca = ECA_ChannelAttention(dim)

        # 新增：GRSA 窗口注意力
        self.grsa = GRSA(dim, window_size=window_size, num_heads=num_heads)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)



    ###下面一行是新增的GRSA
        #
        # x = self.grsa_forward(x)   # 你已有的窗口切分 + GRSA 逻辑
    ####下一行是新增的CoordAttMeanMax
        # x = self.ca(x)
    ####下一行是新增的ECA
        #x = self.eca(x)


        x = input + self.drop_path(x)
        return x

    #下面是新增的GRSA的方法
    def grsa_forward(self, x):
        N, C, H, W = x.shape
        Wh, Ww = 7, 7
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W
        nH, nW = Hp // Wh, Wp // Ww
        x_win = x.unfold(2, Wh, Wh).unfold(3, Ww, Ww)  # (N,C,nH,nW,Wh,Ww)
        x_win = x_win.permute(0, 2, 3, 1, 4, 5).contiguous()  # (N,nH,nW,C,Wh,Ww)
        x_win = x_win.view(-1, C, Wh * Ww).transpose(1, 2)  # (N_win, Wh*Ww, C)
        x_win = self.grsa(x_win)  # GRSA 计算
        x_win = x_win.transpose(1, 2).view(N, nH, nW, C, Wh, Ww)
        x = x_win.permute(0, 3, 1, 4, 2, 5).contiguous().view(N, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x
    
class ECA_ChannelAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x0*y.expand_as(x0)


class Dec_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Dec_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x0*self.sigmoid(out)


class Dec_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Dec_SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x*self.sigmoid(out)

####下面一个类是新增加的
class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttMeanMax, self).__init__()
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Mean pooling branch
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # Max pooling branch
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # Apply attention
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # Expand to original shape
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # Combine outputs
        attention_mean = identity * x_w_mean * x_h_mean
        attention_max = identity * x_w_max * x_h_max

        # Sum the attention outputs
        return attention_mean + attention_max

class GRSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim**-0.5

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # mlp to generate continuous relative position bias
        self.ESRPB_MLP = nn.Sequential(nn.Linear(2, 128, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(128, num_heads, bias=False))
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_position_bias_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0) # 1, 2*Wh-1, 2*Ww-1, 2
        relative_position_bias_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_position_bias_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_position_bias_table *= 3.2  # normalize to -3.2, 3.2
        relative_position_bias_table = torch.sign(relative_position_bias_table) * (1 - torch.exp(
            -torch.abs(relative_position_bias_table)))
        self.register_buffer("relative_position_bias_table", relative_position_bias_table)

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)
        self.q1, self.q2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.k1, self.k2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.v1, self.v2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj1, self.proj2 = nn.Linear(dim//2, dim//2, bias=True), nn.Linear(dim//2, dim//2, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None):
        b_, n, c = x.shape
        x = x.reshape(x.shape[0], x.shape[1], 2, c // 2).permute(2,0,1,3).contiguous()

        #GRL_k
        k = torch.stack((x[0] + self.k1(x[0]), x[1] + self.k2(x[1])), dim= 0)
        k = k.permute(1, 2,0,3).flatten(2)
        k = k.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # GRL_q
        q = torch.stack((x[0] + self.q1(x[0]), x[1] + self.q2(x[1])), dim= 0)
        q = q.permute(1,2,0,3).flatten(2)
        q = q.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # GRL_v
        v = torch.stack((x[0] + self.v1(x[0]), x[1] + self.v2(x[1])), dim=0)
        v = v.permute(1,2,0,3).flatten(2)
        v = v.reshape(b_, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))


        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        logit_scale = torch.clamp(
                    self.logit_scale,
                    max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))
                    ).exp()

        attn = attn * logit_scale

        relative_position_bias_table = self.ESRPB_MLP(self.relative_position_bias_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = x.reshape(b_, n, 2, c // 2).permute(2,0,1,3).contiguous()
        x = torch.stack((self.proj1(x[0]), self.proj2(x[1])), dim=0).permute(1,2,0,3).reshape(b_, n, c)
        return x

#下面的CrossBridge是新增的
class CrossBridger(nn.Module):
    def __init__(self, d_img, d_txt, heads=1):
        super().__init__()
        self.scale = (d_img // heads) ** -0.5
        self.to_q = nn.Conv2d(d_img, d_img, 1, bias=False)
        self.to_kv = nn.Linear(d_txt, d_img*2, bias=False)
        self.proj = nn.Conv2d(d_img, d_img, 1)

    def forward(self, img, txt):            # img: [B,C,H,W], txt: [B,L,C]
        B, C, H, W = img.shape
        q = self.to_q(img).flatten(2).transpose(1, 2)          # [B,HW,C]
        k, v = self.to_kv(txt).chunk(2, dim=-1)                # [B,L,C]
        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B,HW,L]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                           # [B,HW,C]
        out = out.transpose(1, 2).view(B, C, H, W)             # [B,C,H,W]
        return self.proj(out) + img, txt                       # 残差连接