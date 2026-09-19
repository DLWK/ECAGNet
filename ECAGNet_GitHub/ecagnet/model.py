import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import *
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel


class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}

class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]
        os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    
class MMIUNet_GuideDecoder(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(MMIUNet_GuideDecoder, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]    # 224*224
        feature_dim = [768,384,192,96]

        channels = [96, 192, 384, 768] # ConvNeXt-T
        
        self.fusion1 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
        self.fusion2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
        self.fusion3 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
        self.fusion4 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        encoder_feats = []
        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        vis_feat, lan_feat = self.fusion1(image_features[1], text_embeds[-1])
        encoder_feats.append(vis_feat)
        vis_feat, lan_feat = self.fusion2(image_features[2], text_embeds[-1])
        encoder_feats.append(vis_feat)
        vis_feat, lan_feat = self.fusion3(image_features[3], text_embeds[-1])
        encoder_feats.append(vis_feat)
        vis_feat, lan_feat = self.fusion4(image_features[4], text_embeds[-1])
        encoder_feats.append(vis_feat)

        if len(encoder_feats[0].shape) == 4: 
            # encoder_feats = encoder_feats[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            encoder_feats = [rearrange(item,'b c h w -> b (h w) c') for item in encoder_feats] 

        os32 = encoder_feats[3]
        os16 = self.decoder16(os32, encoder_feats[2], text_embeds[-1])
        os8 = self.decoder8(os16, encoder_feats[1], text_embeds[-1])
        os4 = self.decoder4(os8, encoder_feats[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out

#下面这个类是新增的(解码器的多尺度融合,拼接带卷积,计算量大,精度高)
class MultiScaleFusion(nn.Module):
    """跨尺度特征融合模块 (FPN + Dense Skip)"""
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        self.out_conv = nn.Conv2d(len(in_channels_list) * out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features, target_size):
        # features: List[Tensor], 不同分辨率的特征
        resized = []
        for f, proj in zip(features, self.proj):
            x = proj(f)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            resized.append(x)
        fused = torch.cat(resized, dim=1)
        return self.out_conv(fused)

#下面是新增的(编码器的多尺度融合,直接相加,轻量)
class EncoderMultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.out_channels = out_channels

    def forward(self, feats, target_size):
        """
        feats: list[Tensor], 不同层的特征 [f1, f2, ...]
        target_size: (H, W), 目标分辨率
        """
        outs = []
        for conv, f in zip(self.convs, feats):
            x = conv(f)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)
        return torch.sum(torch.stack(outs, dim=0), dim=0)



#下面这个类是新增的
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_img, d_txt, heads=8):
        super().__init__()
        # ---------- 归一化 ----------
        self.norm_img = nn.LayerNorm(d_img)
        self.norm_txt = nn.LayerNorm(d_txt)

        # ---------- 投影：保证维度一致 ----------
        self.txt_to_img = nn.Linear(d_txt, d_img)   # 文本→图像
        self.img_to_txt = nn.Linear(d_img, d_txt)   # 图像→文本 ⭐

        # ---------- 双向 Cross-Attention ⭐ ----------
        self.attn_text_to_img = nn.MultiheadAttention(d_img, heads, batch_first=True)
        self.attn_img_to_text = nn.MultiheadAttention(d_txt, heads, batch_first=True)  # ⭐ 新增

        # ---------- 双向门控（可选） ----------
        self.gate_img = nn.Sequential(nn.Linear(d_txt, d_img), nn.Sigmoid())
        self.gate_txt = nn.Sequential(nn.Linear(d_img, d_txt), nn.Sigmoid())

    def forward(self, img_feat, txt_feat):
        B, C, H, W = img_feat.shape
        # 1. 展平图像并归一化
        x = img_feat.flatten(2).transpose(1, 2)          # (B, HW, C)
        x = self.norm_img(x)                               # (B, HW, d_img)

        t = txt_feat                                       # (B, L, d_txt)
        t = self.norm_txt(t)                               # (B, L, d_txt)

        # 2. 双向投影
        t_img = self.txt_to_img(t)                         # (B, L, d_img)  文本→图像
        x_txt = self.img_to_txt(x)                         # (B, HW, d_txt) 图像→文本 ⭐

        # 3. 双向 Cross-Attention ⭐
        img_out, _ = self.attn_text_to_img(x, t_img, t_img)      # Text→Image
        txt_out, _ = self.attn_img_to_text(t, x_txt, x_txt)      # Image→Text ⭐

        # 4. 残差 + 门控
        x = x + img_out                                    # 图像分支残差
        t = t + txt_out                                    # 文本分支残差 ⭐

        # 4. 残差 + 削弱门控（系数 0.3） 这一段不是完全体
        scale = 0.3
        gate_i = self.gate_img(t.mean(1)) * scale  # 0~0.3
        gate_t = self.gate_txt(x.mean(1)) * scale  # 0~0.3
        x = x * gate_i.unsqueeze(1) + x * (1 - scale)  # 残差比例混合
        t = t * gate_t.unsqueeze(1) + t * (1 - scale)

      #####下面这段是完全体
        # gate_i = self.gate_img(t.mean(1))  # (B, d_img)  0~1
        # gate_t = self.gate_txt(x.mean(1))  # (B, d_txt)  0~1
        # # 图像分支：互补加权
        # x = x * gate_i.unsqueeze(1) + (1 - gate_i.unsqueeze(1)) * (x + img_out)
        # # 文本分支：互补加权
        # t = t * gate_t.unsqueeze(1) + (1 - gate_t.unsqueeze(1)) * (t + txt_out)

        # 5. 还原图像空间
        img_feat = x.transpose(1, 2).view(B, C, H, W)
        return img_feat, t                                 # t 形状不变 (B, L, d_txt)

#下面一行是新增的 动态 mask refinement 级联
class MaskedRefineBlock(nn.Module):
    """
    轻量版：使用深度可分离卷积 + 局部注意力
    """

    def __init__(self, d_model, nhead=8):
        super().__init__()
        # ===== 改动：使用轻量级卷积代替 attention =====
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, d_model // 4, 1),  # 降维到 1/4
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU()
        )

        # 深度可分离卷积代替 attention
        self.depthwise = nn.Conv2d(d_model + d_model // 4, d_model + d_model // 4,
                                   kernel_size=3, padding=1, groups=d_model + d_model // 4)
        self.pointwise = nn.Conv2d(d_model + d_model // 4, d_model, 1)

        # 简化的 FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 1),  # 从 4 倍降到 2 倍
            nn.GELU(),
            nn.Conv2d(d_model * 2, d_model, 1)
        )

        self.norm = nn.BatchNorm2d(d_model)
        self.out_conv = nn.Conv2d(d_model, 1, 1)
        # ===== 改动结束 =====

    def forward(self, feat, coarse_mask):
        """
        feat:    (B, C, H, W)  当前尺度视觉特征
        mask:    (B, 1, H, W)  上尺度上采样后的粗 mask
        return:  (B, 1, H, W)  同分辨率精细 mask
        """
        B, C, H, W = feat.shape

        # ===== 改动：移除下采样，直接处理 =====
        # 1. 特征融合
        mask_feat = self.mask_proj(coarse_mask)  # (B, C/4, H, W)
        combined = torch.cat([feat, mask_feat], dim=1)  # (B, C+C/4, H, W)

        # 2. 深度可分离卷积
        out = self.depthwise(combined)
        out = self.pointwise(out)

        # 3. 残差连接 + FFN
        out = out + feat
        out = self.norm(out)
        out = out + self.ffn(out)

        # 4. 输出
        mask = self.out_conv(out)
        return mask.sigmoid()

#下一段是为了修改att_map改尺寸的方式太硬新增的   att_56 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA-Net)
    只含一个 1-D 自适应卷积，kernel = k | k = odd, k ≈ |C|/γ + 1
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 计算卷积核大小
        k = int(abs((math.log(channels) / math.log(2) + b) / gamma))
        k = k if k % 2 else k + 1          # 必须奇数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局空间池化  (B,C,1,1)
        y = self.avg_pool(x)
        # 降维→1D 卷积→升维
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


#下面是完整的模型代码(最终结果)
class ECAFUNet_V2(nn.Module):      #class MMIUNet_V2(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(ECAFUNet_V2, self).__init__()      #super(MMIUNet_V2, self).__init__()
        self.mars_routing = "default"
        self.fusion_mode = "ecagnet"

        # self.encoder = VisionModel(vision_type, project_dim)
        in_chans = 3
        depths = [3,3,9,3]
        dims = [96,192,384,768]
        drop_path_rate = 0.
        layer_scale_init_value = 1e-6
        head_init_scale = 1.

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], 1)

        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7, 14, 28, 56]    # 224*224
        feature_dim = [768, 384, 192, 96]
        channels = [96, 192, 384, 768] # ConvNeXt-T
        # Direct-fusion comparator: pooled report tokens are projected and added
        # to each visual encoder stage without an explicit spatial prior.
        self.direct_text_proj = nn.ModuleList([nn.Linear(project_dim, c) for c in channels])

        # # #下面的几行是原本的
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.fusion1 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
        # self.fusion2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
        # self.fusion3 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
        # self.fusion4 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)

        # ===== 新增：DeepSeek LLM Block 相关组件 =====
        # 初始化 DeepSeek 配置
        # self.init_deepseek_llm_block()
        # ===== 新增结束 =====




        self.decode4 = Decoder(channels[3],channels[2])
        self.decode3 = Decoder(channels[2],channels[1])
        self.decode2 = Decoder(channels[1],channels[0])

        # 下面这段是新增的  多尺度 skip 融合器 (FPN/Dense skip)   (解码器多尺度融合,计算量大,精度高)
        # self.ms_fusion3 = MultiScaleFusion([channels[3], channels[2]], channels[2])  # 给 d3 用
        # self.ms_fusion2 = MultiScaleFusion([channels[3], channels[2], channels[1]], channels[1])  # 给 d2 用
        # self.ms_fusion1 = MultiScaleFusion([channels[3], channels[2], channels[1], channels[0]], channels[0])  # 给最终输出用
        # self.norm_f3 = LayerNorm(channels[2], eps=1e-6)  # 通道在最后
        # self.norm_f2 = LayerNorm(channels[1], eps=1e-6)
        # self.norm_f1 = LayerNorm(channels[0], eps=1e-6)

        #下面这段是新增的
        # 编码器 多尺度融合器(直接相加,轻量)
        # self.encoder_fusion2 = EncoderMultiScaleFusion([96, 192], 192)
        # self.encoder_fusion3 = EncoderMultiScaleFusion([96, 192, 384], 384)
        # self.encoder_fusion4 = EncoderMultiScaleFusion([96, 192, 384, 768], 768)


        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        # # ===== (改动) 新增：用于LLM输出后处理的模块 =====
        # # 这个模块用于替代简单的 reshape，实现更丰富的特征融合
        # self.llm_post_process = nn.Sequential(
        #     # 3x3 卷积，用于邻居信息互通
        #     nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, padding=1, bias=False),
        #     # 可以复用模型中已有的 LayerNorm 实现
        #     LayerNorm(768, eps=1e-6, data_format="channels_first"),
        #     nn.GELU()
        # )
        # # ===== 改动结束 =====

        #下面一段是搞llm单项蒸馏新增的
        # 定义一个对齐损失，比如均方误差损失 (Mean Squared Error)
        # self.alignment_loss_fn = nn.MSELoss()
        # 定义一个权重来平衡协同训练损失和主分割损失
        # self.alignment_loss_weight = 0.5  # 这是一个超参数，可以调整
        # print(">>> 协同训练模块已初始化。✔")

        # ===== 初始化 DeepSeek LLM Block（无 fallback） =====
        # def init_deepseek_llm_block(self):
        #     """
        #     仅初始化 DeepSeek LLM block 和相关投影层。
        #     不再使用 fallback。
        #     """
        #
        #     from transformers import AutoModelForCausalLM, AutoConfig
        #     import torch.nn as nn
        #
        #     model_name = "path/to/optional-language-model"
        #     layer_idx = 27  #使用后面的层
        #
        #     # 加载配置
        #     config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        #     self.llm_hidden_dim = config.hidden_size  # 2048
        #
        #     # 投影层：视觉特征 -> LLM 维度
        #     self.visual_to_llm = nn.Sequential(
        #         nn.Linear(768, self.llm_hidden_dim),
        #         nn.LayerNorm(self.llm_hidden_dim),
        #         nn.GELU()
        #     )
        #
        #     # 加载预训练模型，并取中间层 block
        #     print(f">>> Loading DeepSeek-R1-Distill-Qwen-1.5B layer {layer_idx} ...")
        #     full_model = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         trust_remote_code=True,
        #         torch_dtype=torch.float32,
        #         device_map="cpu"
        #     )
        #
        #     # 提取中间 decoder block
        #     if hasattr(full_model, 'model') and hasattr(full_model.model, 'layers'):
        #         self.llm_block = full_model.model.layers[layer_idx]
        #     else:
        #         raise AttributeError("Cannot find layers in DeepSeek model")
        #
        #     # 冻结参数
        #     for param in self.llm_block.parameters():
        #         param.requires_grad = False
        #     self.llm_block.eval()
        #
        #     # 释放内存
        #     del full_model
        #     torch.cuda.empty_cache()
        #
        #     # 投影层：LLM → 视觉特征
        #     self.llm_to_visual = nn.Sequential(
        #         nn.Linear(self.llm_hidden_dim, 768),
        #         nn.LayerNorm(768),
        #         nn.GELU()
        #     )
        #
        #     # 门控参数
        #     self.llm_gate = nn.Parameter(torch.zeros(1))  # sigmoid -> 0.5 起点

        #下面一段因向llm中加入文本而改动序列长度和mask,具体的构建在forward中

        # # ===== 修复：注册 causal attention mask（加性，-inf 屏蔽）=====
        # seq_len = 49  # 7x7
        # # 生成 causal mask: 上三角 -inf，下三角 0
        # # causal_mask = torch.full((seq_len, seq_len), float("-inf"))
        # # causal_mask = torch.triu(causal_mask, diagonal=1)  # 严格上三角 -inf
        # # self.register_buffer('causal_mask', causal_mask.unsqueeze(0).unsqueeze(0))  # (1,1,49,49)
        #
        # #下面两行可替代上面三行,更适用于视觉
        # full_mask = torch.zeros((seq_len, seq_len), dtype=torch.float)  # 全0
        # self.register_buffer('attn_mask', full_mask.unsqueeze(0).unsqueeze(0))  # (1,1,49,49)


        #下一行是新增的显式的图文融合的mask
        # ===== 新增：可训练的 Mask Generator（多任务辅助分支）=====
        # 轻量 CNN + 文本引导
        self.mask_gen_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            #下一行新增(增加图文自注意力时)
        )

        #下一段是新增的给图像和文本做自注意力
        # ===== (新增) 自注意力模块 =====
        # 假设输入图像为 224x224，经过 mask_gen_conv 后特征图大小为 56x56
        new_seq_len = 784     #原本是784(尺度改回56改的)
        # 减小特征维度和头数
        feature_dim_mask_gen = 96  # 从 128 减小到 96
        num_heads = 4  # 注意力头数，可以调整


        # # ===== 新增：交叉注意力模块 =====    修改文本平均池化时新增的   改为图文交叉注意力来生成att_map
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim_mask_gen,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(feature_dim_mask_gen)
        # # ===== 新增结束 =====


        # 图像自注意力 (更新位置编码的尺寸)
        self.img_pos_embed = nn.Parameter(torch.zeros(1, new_seq_len, feature_dim_mask_gen))
        self.img_self_attn = nn.MultiheadAttention(feature_dim_mask_gen, num_heads, batch_first=True)
        self.img_attn_norm = nn.LayerNorm(feature_dim_mask_gen)
        # 文本自注意力
        # self.txt_self_attn = nn.MultiheadAttention(feature_dim_mask_gen, num_heads, batch_first=True)
        # self.txt_attn_norm = nn.LayerNorm(feature_dim_mask_gen)
        # ===== (新增结束) =====



        #下一段是原本的
        # self.text_proj = nn.Sequential(
        #     nn.Linear(768, 128),
        #     nn.ReLU()
        # )

        # ===== (新增) 提前下采样到 28×28，序列长度 = 784 =====   下一段是原本的
        # self.down4attn = nn.Conv2d(128, feature_dim_mask_gen, 3, stride=2, padding=1)  # 56->28

        # 替换原来的单个down4attn(原本的)
        self.down4attn = nn.Sequential(
            # 先保持通道数进行空间下采样
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            LayerNorm(128, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            # 再进行通道降维
            nn.Conv2d(128, 96, 1, stride=1),  # 1x1卷积降维
            LayerNorm(96, eps=1e-6, data_format="channels_first")
        )


        #下一段时尺度由28改回56时更改的
        # self.down4attn = nn.Sequential(
        #     # 不进行空间下采样，只做通道变换
        #     nn.Conv2d(128, 96, 1, stride=1),  # 只降维，不降采样
        #     LayerNorm(96, eps=1e-6, data_format="channels_first"),
        #     nn.GELU()
        # )




        #下一段是给图文新增自注意力时改的
        self.text_proj = nn.Linear(768, feature_dim_mask_gen)

        self.mask_refine = nn.Sequential(
            #下一行是原本的
            # nn.Conv2d(feature_dim_mask_gen + feature_dim_mask_gen, 64, 3, padding=1),    #原本是nn.Conv2d(128 + 128, 64, 3, padding=1),

            #下一行是修改文本平均池化更改的  改为图文交叉注意力来生成att_map
            nn.Conv2d(feature_dim_mask_gen, 64, 3, padding=1),

            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            #下一行是原本的(尺度改回56了)
            # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            # 上采样倍数需要增加，因为输入特征图更小了
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),  # 28 -> 224
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.attention_loss_fn = nn.BCELoss()
        # print(">>> Mask Generator (Multi-task) initialized. ✔")
        # ===== 新增结束 =====












        # print(">>> Successfully loaded DeepSeek-R1-Distill-Qwen-1.5B LLM Block ✔")

    # ===== 结束 =====






    # ##### 第二个 U-Net 的组件
    #     # 1. 下采样层 - 第一个下采样层需要处理4通道输入（3通道图像 + 1通道分割图）
    #     self.downsample_layers2 = nn.ModuleList()
    #     stem2 = nn.Sequential(
    #         nn.Conv2d(4, dims[0], kernel_size=4, stride=4),  # 输入通道改为4
    #         LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
    #     )
    #     self.downsample_layers2.append(stem2)
    #     for i in range(3):
    #         downsample_layer = nn.Sequential(
    #             LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
    #             nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
    #         )
    #         self.downsample_layers2.append(downsample_layer)
    #
    #     # 2. 阶段层 - 结构与第一个 U-Net 相同
    #     self.stages2 = nn.ModuleList()
    #     cur = 0
    #     for i in range(4):
    #         stage = nn.Sequential(
    #             *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
    #                     layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
    #         )
    #         self.stages2.append(stage)
    #         cur += depths[i]
    #
    #     # 3. 融合层 - 结构与第一个 U-Net 相同
    #     self.fusion1_2 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
    #     self.fusion2_2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
    #     self.fusion3_2 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
    #     self.fusion4_2 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)
    #
    #     # 4. 解码层 - 结构与第一个 U-Net 相同
    #     self.decode4_2 = Decoder(channels[3], channels[2])
    #     self.decode3_2 = Decoder(channels[2], channels[1])
    #     self.decode2_2 = Decoder(channels[1], channels[0])
    #
    #     # 5. 最终解码器和输出层
    #     self.decoder1_2 = SubpixelUpsample(2, feature_dim[3], 24, 4)
    #     self.out2 = UnetOutBlock(2, in_channels=24, out_channels=1)





    def forward(self, data):
        encoder_feats = []
        image, text = data
        if image.shape[1] == 1:
            image = repeat(image,'b 1 h w -> b c h w', c=3)

        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, _ = text_output['feature'],text_output['project']
        txt = text_embeds[-1]

        # # 下一段是新增的显式图文融合的mask  (原本的,无图文自注意力)
        # # ===== 新增：Mask Generator 前向 =====
        # # 全局文本特征 下一段是原本的
        # global_txt = txt.mean(dim=1)  # (B, 768)
        # # 图像编码（到 H/4, W/4）
        # mg_feat = self.mask_gen_conv(image)  # (B, 128, H/4, W/4)
        # # 文本投影并扩展为空间维度
        # txt_spatial = self.text_proj(global_txt)  # (B, 128)
        # txt_spatial = txt_spatial.view(txt_spatial.size(0), -1, 1, 1)
        # txt_spatial = txt_spatial.expand(-1, -1, mg_feat.size(2), mg_feat.size(3))
        # # 融合并上采样到原图尺寸
        # fused = torch.cat([mg_feat, txt_spatial], dim=1)
        # attention_map = self.mask_refine(fused)  # (B, 1, H, W)
        # # # ===== 新增结束 =====



####### 下一段是修改后源代码!!!!!
        ######下一段是改的 有图文自注意力的显式图文融合mask   !!!!确定的代码
        # # ===== (改动) Mask Generator 前向传播（集成自注意力） =====
        # # 1. 图像特征提取
        mg_feat = self.mask_gen_conv(image)  # (B, 128, 56, 56)
        # ===== (新增) 下采样到 28×28，通道=96 =====
        mg_feat = self.down4attn(mg_feat)  # (B, 96, 28, 28)
        B, C, H, W = mg_feat.shape
        # 2. 图像自注意力处理
        # 展平为序列: (B, C, H, W) -> (B, H*W, C)
        img_seq = mg_feat.flatten(2).permute(0, 2, 1)
        # 添加位置编码
        img_seq = img_seq + self.img_pos_embed
        # 归一化和自注意力 (遵循 Pre-LN 结构)
        norm_img_seq = self.img_attn_norm(img_seq)
        attn_img_seq, _ = self.img_self_attn(norm_img_seq, norm_img_seq, norm_img_seq)
        # 残差连接
        img_seq = img_seq + attn_img_seq
        # 恢复为图像特征图: (B, H*W, C) -> (B, C, H, W)
        attended_mg_feat = img_seq.permute(0, 2, 1).view(B, C, H, W)
        # 3. 文本自注意力处理
        # 投影到与图像相同的维度
        projected_txt_tokens = self.text_proj(txt)


        #下一段是原本的
        # # 归一化和自注意力
        # norm_txt_tokens = self.txt_attn_norm(projected_txt_tokens)
        # attn_txt_tokens, _ = self.txt_self_attn(norm_txt_tokens, norm_txt_tokens, norm_txt_tokens)
        # # 残差连接
        # projected_txt_tokens = projected_txt_tokens + attn_txt_tokens
        # # 全局平均池化得到最终的文本表示
        # txt_global_attended = projected_txt_tokens.mean(dim=1)  # (B, 128)
        # # 4. 图文融合
        # # 将文本特征扩展为空间维度
        # txt_spatial = txt_global_attended.view(B, C, 1, 1)
        # txt_spatial = txt_spatial.expand(-1, -1, H, W)
        # # 拼接并 refine
        # fused = torch.cat([attended_mg_feat, txt_spatial], dim=1)
        # attention_map = self.mask_refine(fused)  # (B, 1, H, W)
        # ===== (改动结束) =====



####### 下一段是修改后源代码!!!!
        # # #下一段是修改文本平均池化新增的      改为图文交叉注意力来生成att_map
        B, C, H, W = attended_mg_feat.shape
        # 4. (核心修改) 图文交叉注意力
        # 展平图像特征作为 Query
        img_seq_query = attended_mg_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C) -> (B, 784, 96)
        # 文本特征作为 Key 和 Value
        txt_kv = projected_txt_tokens  # (B, text_len, 96)
        # 归一化 Query
        norm_img_seq_query = self.cross_attn_norm(img_seq_query)
        # 执行交叉注意力: 每个图像位置向所有文本 token 查询信息
        # Q: img_seq, K: txt_kv, V: txt_kv
        fused_seq, _ = self.cross_attn(
            query=norm_img_seq_query,
            key=txt_kv,
            value=txt_kv
        )
        # 残差连接：将文本信息融入图像特征
        refined_img_seq = img_seq_query + fused_seq
        # 将序列恢复为图像特征图
        refined_img_feat = refined_img_seq.permute(0, 2, 1).view(B, C, H, W)  # (B, 96, 28, 28)
        # 5. (核心修改) 使用融合后的特征进行 refine
        # 不再需要拼接，直接将融合了文本信息的图像特征传入
        attention_map = self.mask_refine(refined_img_feat)  # (B, 1, H, W)
        # # # ===== (改动结束) =====

        direct_text_add = None
        if self.fusion_mode == "direct_fusion":
            text_global = txt.mean(dim=1)
            direct_text_add = [
                proj(text_global).unsqueeze(-1).unsqueeze(-1)
                for proj in self.direct_text_proj
            ]
            # The comparator has no explicitly supervised spatial map.
            attention_map = torch.zeros_like(attention_map)




        ################下一段是原本的编码器部分
        # x = self.downsample_layers[0](image)
        # x = self.stages[0](x)
        # res = x
        # #下一段是原本的
        # vis_feat, txt_feat = self.fusion1(x, txt)
        # encoder_feats.append(vis_feat)    #原本是encoder_feats.append(vis_feat)    里面的vis-feat改为x是为了删去原论文的图文交叉融合(下面几个同理)
        # x = self.downsample_layers[1](vis_feat + res)     #原本是x = self.downsample_layers[1](vis_feat + res)  里面的vis_feat + res改为x是为了删去原论文的图文交叉融合(下面几个同理)
        # x = self.stages[1](x)
        # res = x
        # # 下一段是原本的
        # vis_feat, txt_feat = self.fusion2(x, txt + txt_feat)
        # encoder_feats.append(vis_feat)    #原本是encoder_feats.append(vis_feat)
        #
        # #下一行新增(解码器多尺度融合,轻量级)
        # #encoder_feats[1] = self.encoder_fusion2(encoder_feats[:2], encoder_feats[1].shape[-2:])
        #
        # x = self.downsample_layers[2](vis_feat + res)   # x = self.downsample_layers[2](vis_feat + res)
        # x = self.stages[2](x)
        # res = x
        # # 下一段是原本的
        # vis_feat, txt_feat = self.fusion3(x, txt + txt_feat)
        # encoder_feats.append(vis_feat)       #原本是encoder_feats.append(vis_feat)
        #
        # # 下一行新增(解码器多尺度融合,轻量级)
        # #encoder_feats[2] = self.encoder_fusion3(encoder_feats[:3], encoder_feats[2].shape[-2:])
        #
        # x = self.downsample_layers[3](vis_feat + res)        # x = self.downsample_layers[3](vis_feat + res)
        # x = self.stages[3](x)
        # # 下一段是原本的
        # vis_feat, txt_feat = self.fusion4(x, txt + txt_feat)
        # encoder_feats.append(vis_feat)        # encoder_feats.append(vis_feat)


        ####下一段是更改的融合残差叠加的显式图文融合的mask的编码器
        # ===== 修改后的编码器部分（引入 Attention-Guided Residual Path）=====
        #####下一段是原本的
        # x = self.downsample_layers[0](image)
        # x = self.stages[0](x)
        # att_56 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
        # x_att = x * att_56
        # x_fused = x + x_att
        # encoder_feats.append(x_fused)
        # # Stage 1 (28x28)
        # x = self.downsample_layers[1](x_fused)
        # x = self.stages[1](x)
        # att_28 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
        # x_att = x * att_28
        # x_fused = x + x_att
        # encoder_feats.append(x_fused)
        # # Stage 2 (14x14)
        # x = self.downsample_layers[2](x_fused)
        # x = self.stages[2](x)
        # att_14 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
        # x_att = x * att_14
        # x_fused = x + x_att
        # encoder_feats.append(x_fused)
        # # Stage 3 (7x7)
        # x = self.downsample_layers[3](x_fused)
        # x = self.stages[3](x)
        # att_7 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
        # x_att = x * att_7
        # x_fused = x + x_att
        # encoder_feats.append(x_fused)


####### 下一段是修改后源代码!!!!!!!
        #下一段是高仿源代码的编码层    编码层放x_att而非x_fused    !!!确定的代码
        x = self.downsample_layers[0](image)
        x = self.stages[0](x)
        if direct_text_add is not None:
            x = x + direct_text_add[0]
            encoder_feats.append(x)
            x = self.downsample_layers[1](x)
            x = self.stages[1](x)
            x = x + direct_text_add[1]
            encoder_feats.append(x)
            x = self.downsample_layers[2](x)
            x = self.stages[2](x)
            x = x + direct_text_add[2]
            encoder_feats.append(x)
            x = self.downsample_layers[3](x)
            x = self.stages[3](x)
            x = x + direct_text_add[3]
            encoder_feats.append(x)
        else:
        #下一行是原本的
            guidance_map = attention_map.mean(dim=(2, 3), keepdim=True) if self.mars_routing == "global_gate" else attention_map
            att_56 = F.interpolate(guidance_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x_att = x * att_56
            x_fused = x + x_att
            encoder_feats.append(x_fused if self.mars_routing in ("all_fused", "reversed") else x_att)
        # Stage 1 (28x28)
            x = self.downsample_layers[1](x_fused)
            x = self.stages[1](x)
        # 下一行是原本的
            att_28 = F.interpolate(guidance_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x_att = x * att_28
            x_fused = x + x_att
            encoder_feats.append(x_fused if self.mars_routing in ("all_fused", "reversed") else x_att)
        # Stage 2 (14x14)
            x = self.downsample_layers[2](x_fused)
            x = self.stages[2](x)
        # 下一行是原本的
            att_14 = F.interpolate(guidance_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x_att = x * att_14
            x_fused = x + x_att
            encoder_feats.append(x_att if self.mars_routing in ("all_att", "reversed") else x_fused)
        # Stage 3 (7x7)
            x = self.downsample_layers[3](x_fused)
            x = self.stages[3](x)
        # 下一行是原本的
            att_7 = F.interpolate(guidance_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
            x_att = x * att_7
            x_fused = x + x_att
            encoder_feats.append(x_att if self.mars_routing in ("all_att", "reversed") else x_fused)


        #下一段时不加显式图文融合mask的最简单纯粹的编码器
        # x = self.downsample_layers[0](image)
        # x = self.stages[0](x)
        # # 下一行是原本的
        # encoder_feats.append(x)  # 原本是 encoder_feats.append(x_att)
        # # Stage 1 (28x28)
        # x = self.downsample_layers[1](x)
        # x = self.stages[1](x)
        # # 下一行是原本的
        # encoder_feats.append(x)
        # # Stage 2 (14x14)
        # x = self.downsample_layers[2](x)
        # x = self.stages[2](x)
        # # 下一行是原本的
        # encoder_feats.append(x)
        # # Stage 3 (7x7)
        # x = self.downsample_layers[3](x)
        # x = self.stages[3](x)
        # # 下一行是原本的
        # encoder_feats.append(x)









        # #下一段是新增的显式图文融合的mask
        # # # ===== 新增：Soft masking 融合 =====
        # att_56 = F.interpolate(attention_map, size=(56, 56), mode='bilinear', align_corners=True)
        # att_28 = F.interpolate(attention_map, size=(28, 28), mode='bilinear', align_corners=True)
        # att_14 = F.interpolate(attention_map, size=(14, 14), mode='bilinear', align_corners=True)
        # att_7 = F.interpolate(attention_map, size=( 7,  7), mode='bilinear', align_corners=True)
        # encoder_feats[0] = encoder_feats[0] * att_56
        # encoder_feats[1] = encoder_feats[1] * att_28
        # encoder_feats[2] = encoder_feats[2] * att_14
        # encoder_feats[3] = encoder_feats[3] * att_7


        # # ===== 新增结束 =====



        # 下一行新增(解码器多尺度融合,轻量级)
        #encoder_feats[3] = self.encoder_fusion4(encoder_feats[:4], encoder_feats[3].shape[-2:])

        # ===== 新增：DeepSeek LLM Block 处理 =====
        # 获取最高层的编码特征
        # high_level_feat = encoder_feats[-1]  # (B, 768, 7, 7)
        # B, C, H, W = high_level_feat.shape
        # num_visual_tokens = H * W  # 49
        #
        # # 1. Flatten 并转置: (B, C, H, W) -> (B, H*W, C)
        # visual_tokens = high_level_feat.flatten(2).permute(0, 2, 1)  # (B, 49, 768)

        #下一段是为了向llm中接入文本新增的
        # 2. 准备文本 Token
        # text_output['feature'][-1] 包含了BERT最后一层的输出, shape: (B, text_len, 768)
        # text_tokens = text_output['feature'][-1]
        # num_text_tokens = text_tokens.shape[1]
        # # 3. (核心) 拼接视觉和文本 Token
        # # 注意：确保 visual_tokens 和 text_tokens 的最后一个维度（特征维度）是相同的 (都是768)
        # # 序列维度 (dim=1) 上拼接: [B, 49, 768] + [B, text_len, 768] -> [B, 49 + text_len, 768]
        # multimodal_tokens = torch.cat([visual_tokens, text_tokens], dim=1)

        #下一段是为了向llm中接入文本新增的
        # 4. 投影到 LLM 维度
        # llm_input = self.visual_to_llm(multimodal_tokens)  # (B, 49 + text_len, llm_hidden_dim)
        #下一段是原本的
        # llm_input = self.visual_to_llm(visual_tokens)  # (B, 49, llm_hidden_dim)

        # 下一段是为了向llm中接入文本新增的
        # 5. 构建动态的 position_ids 和 attention_mask
        # total_seq_len = num_visual_tokens + num_text_tokens

        # 下一段是为了向llm中接入文本新增的
        # 构建 position_ids
        # position_ids = torch.arange(total_seq_len, device=llm_input.device).unsqueeze(0).expand(B, -1)
        #下一段是原本的
        # 3. 构建位置和注意力信息
        # position_ids = torch.arange(49, device=llm_input.device).unsqueeze(0).expand(B, -1)  # (B, 49)
        # attn_mask = self.causal_mask.expand(B, -1, -1, -1).to(llm_input.device)  # (B, 1, 49, 49)

        #下一段是为了向llm中接入文本新增的
        # 构建 attention_mask
        # 视觉部分: 全都互相可见 (全1矩阵)
        # visual_mask = torch.ones(B, num_visual_tokens, device=llm_input.device)
        # # 文本部分: 使用原始的 attention mask (处理padding)
        # text_mask = text['attention_mask']  # (B, text_len), 1 for real tokens, 0 for padding
        # # 拼接成一维的mask
        # combined_1d_mask = torch.cat([visual_mask, text_mask], dim=1)  # (B, total_seq_len)
        # # 扩展成LLM需要的4D attention mask (B, 1, To, From)
        # attn_mask = combined_1d_mask[:, None, None, :].expand(B, 1, total_seq_len, total_seq_len).to(llm_input.device)
        # # 这是最简单的全注意力mask。如果需要更复杂的causal mask，逻辑会更复杂。对于视觉任务，全注意力通常OK。
        # # Hugging Face 的 LLM block 通常期望 additive mask (0 for attend, -inf for ignore)
        # # 我们需要转换一下
        # additive_attn_mask = (1.0 - attn_mask) * torch.finfo(llm_input.dtype).min

        #下面一行在上面的注意力改为更适合视觉的注意力时替换掉上一行
        # 下一段是原本的
        # attn_mask = self.attn_mask.expand(B, -1, -1, -1).to(llm_input.device)  # (B, 1, 49, 49)

        # 4. 调用 DeepSeek LLM Block（无 try-except 简化调用）
        # llm_output = self.llm_block(
        #     hidden_states=llm_input,
        #     attention_mask=additive_attn_mask,   #原本是 attn_mask
        #     position_ids=position_ids,
        #     use_cache=False
        # )
        # if isinstance(llm_output, tuple):
        #     llm_output = llm_output[0]

        # 下一段是为了向llm中接入文本新增的
        # 7. (核心) 从输出中分离出视觉 Token
        # 输出的序列顺序和输入一致，所以前49个是处理后的视觉token
        # processed_visual_tokens_llm_dim = llm_output[:, :num_visual_tokens, :]  # (B, 49, llm_hidden_dim)
        #
        #
        # # 5. 投影回视觉特征维度
        # refined_tokens = self.llm_to_visual(processed_visual_tokens_llm_dim)  # (B, 49, 768)
        #下一行是原本的
        # refined_tokens = self.llm_to_visual(llm_output)  # (B, 49, 768)
        # 6. Reshape 回 feature map: (B, H*W, C) -> (B, C, H, W)
        # refined_feat = refined_tokens.permute(0, 2, 1).view(B, C, H, W)
        #
        #
        # # 下面一段是搞llm单项蒸馏新增的
        # alignment_loss = self.alignment_loss_fn(high_level_feat, refined_feat)
        # 策略A (推荐): 用“学生”自己的特征继续。
        # 这样可以保证解码器是基于可训练的编码器进行优化的，更稳定。
        # 协同训练损失会通过反向传播，间接将LLM的知识“注入”到学生编码器中。
        # final_bottleneck_feat = high_level_feat
        # 策略B: 用融合后的特征继续。
        # 这种方式更直接，但可能因为老师的特征是固定的而导致训练不稳定。
        # gate = 0.5*torch.sigmoid(self.llm_gate)
        # final_bottleneck_feat = high_level_feat + gate * refined_feat
        # # 我们将使用策略A，并更新encoder_feats
        # encoder_feats[-1] = final_bottleneck_feat




        # 7. 门控残差连接(带权重使用llm)(原本的连接文本的门控)
        # gate =  0.3*torch.sigmoid(self.llm_gate) #原本是0.8*
        # encoder_feats[-1] = high_level_feat + gate * refined_feat

        # 完全使用 LLM 的 semantic-enhanced 输出
        # encoder_feats[-1] = refined_feat  # 完全使用 LLM 的 semantic-enhanced 输出
        # ===== 结束 =====

        #下一段是原本的纯粹的解码器(显式图文融合的mask)
        d4 = self.decode4(encoder_feats[3], encoder_feats[2])

        #下面这段是新增的
        # f3 = self.ms_fusion3([encoder_feats[3], encoder_feats[2]], d4.shape[-2:])
        # f3 = self.norm_f3(f3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # ← 新增

        d3 = self.decode3(d4, encoder_feats[1]) #原本是d3 = self.decode3(d4, encoder_feats[1])  d4+f3

        # 下面这段是新增的
        # f2 = self.ms_fusion2([encoder_feats[3], encoder_feats[2], encoder_feats[1]], d3.shape[-2:])
        # f2 = self.norm_f2(f2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # ← 新增

        d2 = self.decode2(d3, encoder_feats[0]) #原本是d2 = self.decode2(d3, encoder_feats[0])  d3+f2

        # 下面这段是新增的
        # f1 = self.ms_fusion1([encoder_feats[3], encoder_feats[2], encoder_feats[1], encoder_feats[0]], d2.shape[-2:])
        # f1 = self.norm_f1(f1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # ← 新增

        os1 = self.decoder1(d2) #原本是os1 = self.decoder1(d2)  d2+f1
        out = self.out(os1).sigmoid()


        #下一段是改的添加显式图文融合mask的 解码器  方式1
        # 首先, 用 attention_map 来精炼 skip-connection 特征
        # guided_skip_feat2 = encoder_feats[2] * att_14  # Soft masking
        # # 或者使用残差方式: guided_skip_feat2 = encoder_feats[2] + encoder_feats[2] * att_14
        # d4 = self.decode4(encoder_feats[3], guided_skip_feat2)
        # # d3: 从 14x14 上采样到 28x28, 与 encoder_feats[1] (28x28) 融合
        # guided_skip_feat1 = encoder_feats[1] * att_28
        # d3 = self.decode3(d4, guided_skip_feat1)
        # # d2: 从 28x28 上采样到 56x56, 与 encoder_feats[0] (56x56) 融合
        # guided_skip_feat0 = encoder_feats[0] * att_56
        # d2 = self.decode2(d3, guided_skip_feat0)
        # # 最后的上采样和输出层
        # os1 = self.decoder1(d2)
        # out = self.out(os1).sigmoid()
        # ===== 解码器结束 =====








        ##### 第二个U-Net（下半部分）- 使用第一个U-Net的输出作为输入
        # encoder_feats2 = []  # 为第二个U-Net创建特征列表
        #
        # # 将第一个U-Net的输出与原始图像拼接作为第二个U-Net的输入
        # # 或者可以直接使用第一个U-Net的输出，这里选择拼接以保留更多信息
        # second_input = torch.cat([image, out1], dim=1)
        #
        # # 第二个U-Net的编码路径
        # x2 = self.downsample_layers2[0](second_input)  # 需要额外的下采样层
        # x2 = self.stages2[0](x2)  # 需要额外的阶段层
        # res2 = x2
        #
        # vis_feat2, txt_feat2 = self.fusion1_2(x2, txt)  # 需要额外的融合层
        # encoder_feats2.append(vis_feat2)
        # x2 = self.downsample_layers2[1](vis_feat2 + res2)
        # x2 = self.stages2[1](x2)
        # res2 = x2
        #
        # vis_feat2, txt_feat2 = self.fusion2_2(x2, txt + txt_feat2)
        # encoder_feats2.append(vis_feat2)
        # x2 = self.downsample_layers2[2](vis_feat2 + res2)
        # x2 = self.stages2[2](x2)
        # res2 = x2
        #
        # vis_feat2, txt_feat2 = self.fusion3_2(x2, txt + txt_feat2)
        # encoder_feats2.append(vis_feat2)
        # x2 = self.downsample_layers2[3](vis_feat2 + res2)
        # x2 = self.stages2[3](x2)
        #
        # vis_feat2, txt_feat2 = self.fusion4_2(x2, txt + txt_feat2)
        # encoder_feats2.append(vis_feat2)
        #
        # ##第二个U-Net的解码路径
        # d4_2 = self.decode4_2(encoder_feats2[3], encoder_feats2[2])  # 需要额外的解码层
        # d3_2 = self.decode3_2(d4_2, encoder_feats2[1])
        # d2_2 = self.decode2_2(d3_2, encoder_feats2[0])
        # os1_2 = self.decoder1_2(d2_2)
        # out2 = self.out2(os1_2).sigmoid()  # 第二个U-Net的输出，也是最终输出




        return  out, attention_map    ##原本是return out      return out, attention_map























#下面是跑消融实验的代码(CAGM)
# class ECAFUNet_V2(nn.Module):      #class MMIUNet_V2(nn.Module):
#
#     def __init__(self, bert_type, vision_type, project_dim=512):
#
#         super(ECAFUNet_V2, self).__init__()      #super(MMIUNet_V2, self).__init__()
#
#         # self.encoder = VisionModel(vision_type, project_dim)
#         in_chans = 3
#         depths = [3,3,9,3]
#         dims = [96,192,384,768]
#         drop_path_rate = 0.
#         layer_scale_init_value = 1e-6
#         head_init_scale = 1.
#
#         self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.downsample_layers.append(stem)
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                 LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                 nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)
#
#         self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
#         dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
#                         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#
#         self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
#         self.head = nn.Linear(dims[-1], 1)
#
#         self.text_encoder = BERTModel(bert_type, project_dim)
#
#         self.spatial_dim = [7, 14, 28, 56]    # 224*224
#         feature_dim = [768, 384, 192, 96]
#         channels = [96, 192, 384, 768] # ConvNeXt-T
#
#         # #下面的几行是原本的
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.fusion1 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
#         self.fusion2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
#         self.fusion3 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
#         self.fusion4 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)
#
#         # ===== 新增：DeepSeek LLM Block 相关组件 =====
#         # 初始化 DeepSeek 配置
#         # self.init_deepseek_llm_block()
#         # ===== 新增结束 =====
#
#
#
#
#         self.decode4 = Decoder(channels[3],channels[2])
#         self.decode3 = Decoder(channels[2],channels[1])
#         self.decode2 = Decoder(channels[1],channels[0])
#
#         # 下面这段是新增的  多尺度 skip 融合器 (FPN/Dense skip)   (解码器多尺度融合,计算量大,精度高)
#         self.ms_fusion3 = MultiScaleFusion([channels[3], channels[2]], channels[2])  # 给 d3 用
#         self.ms_fusion2 = MultiScaleFusion([channels[3], channels[2], channels[1]], channels[1])  # 给 d2 用
#         self.ms_fusion1 = MultiScaleFusion([channels[3], channels[2], channels[1], channels[0]], channels[0])  # 给最终输出用
#         self.norm_f3 = LayerNorm(channels[2], eps=1e-6)  # 通道在最后
#         self.norm_f2 = LayerNorm(channels[1], eps=1e-6)
#         self.norm_f1 = LayerNorm(channels[0], eps=1e-6)
#
#         #下面这段是新增的
#         # 编码器 多尺度融合器(直接相加,轻量)
#         self.encoder_fusion2 = EncoderMultiScaleFusion([96, 192], 192)
#         self.encoder_fusion3 = EncoderMultiScaleFusion([96, 192, 384], 384)
#         self.encoder_fusion4 = EncoderMultiScaleFusion([96, 192, 384, 768], 768)
#
#
#         self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
#         self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
#
#         # # ===== (改动) 新增：用于LLM输出后处理的模块 =====
#         # # 这个模块用于替代简单的 reshape，实现更丰富的特征融合
#         # self.llm_post_process = nn.Sequential(
#         #     # 3x3 卷积，用于邻居信息互通
#         #     nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, padding=1, bias=False),
#         #     # 可以复用模型中已有的 LayerNorm 实现
#         #     LayerNorm(768, eps=1e-6, data_format="channels_first"),
#         #     nn.GELU()
#         # )
#         # # ===== 改动结束 =====
#
#         #下面一段是搞llm单项蒸馏新增的
#         # 定义一个对齐损失，比如均方误差损失 (Mean Squared Error)
#         self.alignment_loss_fn = nn.MSELoss()
#         # 定义一个权重来平衡协同训练损失和主分割损失
#         self.alignment_loss_weight = 0.5  # 这是一个超参数，可以调整
#         # print(">>> 协同训练模块已初始化。✔")
#
#         # ===== 初始化 DeepSeek LLM Block（无 fallback） =====
#         # def init_deepseek_llm_block(self):
#         #     """
#         #     仅初始化 DeepSeek LLM block 和相关投影层。
#         #     不再使用 fallback。
#         #     """
#         #
#         #     from transformers import AutoModelForCausalLM, AutoConfig
#         #     import torch.nn as nn
#         #
#         #     model_name = "path/to/optional-language-model"
#         #     layer_idx = 27  #使用后面的层
#         #
#         #     # 加载配置
#         #     config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
#         #     self.llm_hidden_dim = config.hidden_size  # 2048
#         #
#         #     # 投影层：视觉特征 -> LLM 维度
#         #     self.visual_to_llm = nn.Sequential(
#         #         nn.Linear(768, self.llm_hidden_dim),
#         #         nn.LayerNorm(self.llm_hidden_dim),
#         #         nn.GELU()
#         #     )
#         #
#         #     # 加载预训练模型，并取中间层 block
#         #     print(f">>> Loading DeepSeek-R1-Distill-Qwen-1.5B layer {layer_idx} ...")
#         #     full_model = AutoModelForCausalLM.from_pretrained(
#         #         model_name,
#         #         trust_remote_code=True,
#         #         torch_dtype=torch.float32,
#         #         device_map="cpu"
#         #     )
#         #
#         #     # 提取中间 decoder block
#         #     if hasattr(full_model, 'model') and hasattr(full_model.model, 'layers'):
#         #         self.llm_block = full_model.model.layers[layer_idx]
#         #     else:
#         #         raise AttributeError("Cannot find layers in DeepSeek model")
#         #
#         #     # 冻结参数
#         #     for param in self.llm_block.parameters():
#         #         param.requires_grad = False
#         #     self.llm_block.eval()
#         #
#         #     # 释放内存
#         #     del full_model
#         #     torch.cuda.empty_cache()
#         #
#         #     # 投影层：LLM → 视觉特征
#         #     self.llm_to_visual = nn.Sequential(
#         #         nn.Linear(self.llm_hidden_dim, 768),
#         #         nn.LayerNorm(768),
#         #         nn.GELU()
#         #     )
#         #
#         #     # 门控参数
#         #     self.llm_gate = nn.Parameter(torch.zeros(1))  # sigmoid -> 0.5 起点
#
#         #下面一段因向llm中加入文本而改动序列长度和mask,具体的构建在forward中
#
#         # # ===== 修复：注册 causal attention mask（加性，-inf 屏蔽）=====
#         # seq_len = 49  # 7x7
#         # # 生成 causal mask: 上三角 -inf，下三角 0
#         # # causal_mask = torch.full((seq_len, seq_len), float("-inf"))
#         # # causal_mask = torch.triu(causal_mask, diagonal=1)  # 严格上三角 -inf
#         # # self.register_buffer('causal_mask', causal_mask.unsqueeze(0).unsqueeze(0))  # (1,1,49,49)
#         #
#         # #下面两行可替代上面三行,更适用于视觉
#         # full_mask = torch.zeros((seq_len, seq_len), dtype=torch.float)  # 全0
#         # self.register_buffer('attn_mask', full_mask.unsqueeze(0).unsqueeze(0))  # (1,1,49,49)
#
#
#         #下一行是新增的显式的图文融合的mask
#         # ===== 新增：可训练的 Mask Generator（多任务辅助分支）=====
#         # 轻量 CNN + 文本引导
#         self.mask_gen_conv = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             #下一行新增(增加图文自注意力时)
#         )
#
#         #下一段是新增的给图像和文本做自注意力
#         # ===== (新增) 自注意力模块 =====
#         # 假设输入图像为 224x224，经过 mask_gen_conv 后特征图大小为 56x56
#         new_seq_len = 784     #原本是784(尺度改回56改的)
#         # 减小特征维度和头数
#         feature_dim_mask_gen = 96  # 从 128 减小到 96
#         num_heads = 4  # 注意力头数，可以调整
#
#
#         # # ===== 新增：交叉注意力模块 =====    修改文本平均池化时新增的   改为图文交叉注意力来生成att_map
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=feature_dim_mask_gen,
#             num_heads=num_heads,
#             batch_first=True
#         )
#         self.cross_attn_norm = nn.LayerNorm(feature_dim_mask_gen)
#         # # ===== 新增结束 =====
#
#
#         # 图像自注意力 (更新位置编码的尺寸)
#         self.img_pos_embed = nn.Parameter(torch.zeros(1, new_seq_len, feature_dim_mask_gen))
#         self.img_self_attn = nn.MultiheadAttention(feature_dim_mask_gen, num_heads, batch_first=True)
#         self.img_attn_norm = nn.LayerNorm(feature_dim_mask_gen)
#         # 文本自注意力
#         self.txt_self_attn = nn.MultiheadAttention(feature_dim_mask_gen, num_heads, batch_first=True)
#         self.txt_attn_norm = nn.LayerNorm(feature_dim_mask_gen)
#         # ===== (新增结束) =====
#
#
#
#         #下一段是原本的
#         # self.text_proj = nn.Sequential(
#         #     nn.Linear(768, 128),
#         #     nn.ReLU()
#         # )
#
#         # ===== (新增) 提前下采样到 28×28，序列长度 = 784 =====   下一段是原本的
#         # self.down4attn = nn.Conv2d(128, feature_dim_mask_gen, 3, stride=2, padding=1)  # 56->28
#
#         # 替换原来的单个down4attn(原本的)
#         self.down4attn = nn.Sequential(
#             # 先保持通道数进行空间下采样
#             nn.Conv2d(128, 128, 3, stride=2, padding=1),
#             LayerNorm(128, eps=1e-6, data_format="channels_first"),
#             nn.GELU(),
#             # 再进行通道降维
#             nn.Conv2d(128, 96, 1, stride=1),  # 1x1卷积降维
#             LayerNorm(96, eps=1e-6, data_format="channels_first")
#         )
#
#
#         #下一段时尺度由28改回56时更改的
#         # self.down4attn = nn.Sequential(
#         #     # 不进行空间下采样，只做通道变换
#         #     nn.Conv2d(128, 96, 1, stride=1),  # 只降维，不降采样
#         #     LayerNorm(96, eps=1e-6, data_format="channels_first"),
#         #     nn.GELU()
#         # )
#
#
#
#
#         #下一段是给图文新增自注意力时改的
#         self.text_proj = nn.Linear(768, feature_dim_mask_gen)
#
#         self.mask_refine = nn.Sequential(
#             #下一行是原本的
#             # nn.Conv2d(feature_dim_mask_gen + feature_dim_mask_gen, 64, 3, padding=1),    #原本是nn.Conv2d(128 + 128, 64, 3, padding=1),
#
#             #下一行是修改文本平均池化更改的  改为图文交叉注意力来生成att_map
#             nn.Conv2d(feature_dim_mask_gen, 64, 3, padding=1),
#
#             nn.ReLU(),
#             nn.Conv2d(64, 32, 3, padding=1),
#             nn.ReLU(),
#             #下一行是原本的(尺度改回56了)
#             # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
#             # 上采样倍数需要增加，因为输入特征图更小了
#             nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),  # 28 -> 224
#             nn.Conv2d(32, 1, 1),
#             nn.Sigmoid()
#         )
#         self.attention_loss_fn = nn.BCELoss()
#         # print(">>> Mask Generator (Multi-task) initialized. ✔")
#         # ===== 新增结束 =====
#
#
#
#
#
#
#
#
#
#
#
#
#         # print(">>> Successfully loaded DeepSeek-R1-Distill-Qwen-1.5B LLM Block ✔")
#
#     # ===== 结束 =====
#
#
#
#
#
#
#     # ##### 第二个 U-Net 的组件
#     #     # 1. 下采样层 - 第一个下采样层需要处理4通道输入（3通道图像 + 1通道分割图）
#     #     self.downsample_layers2 = nn.ModuleList()
#     #     stem2 = nn.Sequential(
#     #         nn.Conv2d(4, dims[0], kernel_size=4, stride=4),  # 输入通道改为4
#     #         LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#     #     )
#     #     self.downsample_layers2.append(stem2)
#     #     for i in range(3):
#     #         downsample_layer = nn.Sequential(
#     #             LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#     #             nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
#     #         )
#     #         self.downsample_layers2.append(downsample_layer)
#     #
#     #     # 2. 阶段层 - 结构与第一个 U-Net 相同
#     #     self.stages2 = nn.ModuleList()
#     #     cur = 0
#     #     for i in range(4):
#     #         stage = nn.Sequential(
#     #             *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
#     #                     layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
#     #         )
#     #         self.stages2.append(stage)
#     #         cur += depths[i]
#     #
#     #     # 3. 融合层 - 结构与第一个 U-Net 相同
#     #     self.fusion1_2 = Bridger(d_img=channels[0], d_model=channels[0], stage_id=1)
#     #     self.fusion2_2 = Bridger(d_img=channels[1], d_model=channels[1], stage_id=2)
#     #     self.fusion3_2 = Bridger(d_img=channels[2], d_model=channels[2], stage_id=3)
#     #     self.fusion4_2 = Bridger(d_img=channels[3], d_model=channels[3], stage_id=4)
#     #
#     #     # 4. 解码层 - 结构与第一个 U-Net 相同
#     #     self.decode4_2 = Decoder(channels[3], channels[2])
#     #     self.decode3_2 = Decoder(channels[2], channels[1])
#     #     self.decode2_2 = Decoder(channels[1], channels[0])
#     #
#     #     # 5. 最终解码器和输出层
#     #     self.decoder1_2 = SubpixelUpsample(2, feature_dim[3], 24, 4)
#     #     self.out2 = UnetOutBlock(2, in_channels=24, out_channels=1)
#
#
#
#
#
#     def forward(self, data):
#         encoder_feats = []
#         image, text = data
#         if image.shape[1] == 1:
#             image = repeat(image,'b 1 h w -> b c h w', c=3)
#
#         text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
#         text_embeds, _ = text_output['feature'],text_output['project']
#         txt = text_embeds[-1]
#
#         # # 下一段是新增的显式图文融合的mask  (原本的,无图文自注意力)
#         # # ===== 新增：Mask Generator 前向 =====
#         # # 全局文本特征 下一段是原本的
#         # global_txt = txt.mean(dim=1)  # (B, 768)
#         # # 图像编码（到 H/4, W/4）
#         # mg_feat = self.mask_gen_conv(image)  # (B, 128, H/4, W/4)
#         # # 文本投影并扩展为空间维度
#         # txt_spatial = self.text_proj(global_txt)  # (B, 128)
#         # txt_spatial = txt_spatial.view(txt_spatial.size(0), -1, 1, 1)
#         # txt_spatial = txt_spatial.expand(-1, -1, mg_feat.size(2), mg_feat.size(3))
#         # # 融合并上采样到原图尺寸
#         # fused = torch.cat([mg_feat, txt_spatial], dim=1)
#         # attention_map = self.mask_refine(fused)  # (B, 1, H, W)
#         # # # ===== 新增结束 =====
#
#
#
# ####### 下一段是修改后源代码!!!!!
#         ######下一段是改的 有图文自注意力的显式图文融合mask   !!!!确定的代码
#         # # ===== (改动) Mask Generator 前向传播（集成自注意力） =====
#         # # 1. 图像特征提取
#         mg_feat = self.mask_gen_conv(image)  # (B, 128, 56, 56)
#         # ===== (新增) 下采样到 28×28，通道=96 =====
#         mg_feat = self.down4attn(mg_feat)  # (B, 96, 28, 28)
#         B, C, H, W = mg_feat.shape
#         # 2. 图像自注意力处理
#         # 展平为序列: (B, C, H, W) -> (B, H*W, C)
#         img_seq = mg_feat.flatten(2).permute(0, 2, 1)
#         # 添加位置编码
#         img_seq = img_seq + self.img_pos_embed
#         # 归一化和自注意力 (遵循 Pre-LN 结构)
#         norm_img_seq = self.img_attn_norm(img_seq)
#         attn_img_seq, _ = self.img_self_attn(norm_img_seq, norm_img_seq, norm_img_seq)
#         # 残差连接
#         img_seq = img_seq + attn_img_seq
#         # 恢复为图像特征图: (B, H*W, C) -> (B, C, H, W)
#         attended_mg_feat = img_seq.permute(0, 2, 1).view(B, C, H, W)
#         # 3. 文本自注意力处理
#         # 投影到与图像相同的维度
#         projected_txt_tokens = self.text_proj(txt)
#
#
#         #下一段是原本的
#         # # 归一化和自注意力
#         # norm_txt_tokens = self.txt_attn_norm(projected_txt_tokens)
#         # attn_txt_tokens, _ = self.txt_self_attn(norm_txt_tokens, norm_txt_tokens, norm_txt_tokens)
#         # # 残差连接
#         # projected_txt_tokens = projected_txt_tokens + attn_txt_tokens
#         # # 全局平均池化得到最终的文本表示
#         # txt_global_attended = projected_txt_tokens.mean(dim=1)  # (B, 128)
#         # # 4. 图文融合
#         # # 将文本特征扩展为空间维度
#         # txt_spatial = txt_global_attended.view(B, C, 1, 1)
#         # txt_spatial = txt_spatial.expand(-1, -1, H, W)
#         # # 拼接并 refine
#         # fused = torch.cat([attended_mg_feat, txt_spatial], dim=1)
#         # attention_map = self.mask_refine(fused)  # (B, 1, H, W)
#         # ===== (改动结束) =====
#
#
#
# ####### 下一段是修改后源代码!!!!
#         # # #下一段是修改文本平均池化新增的      改为图文交叉注意力来生成att_map
#         B, C, H, W = attended_mg_feat.shape
#         # 4. (核心修改) 图文交叉注意力
#         # 展平图像特征作为 Query
#         img_seq_query = attended_mg_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C) -> (B, 784, 96)
#         # 文本特征作为 Key 和 Value
#         txt_kv = projected_txt_tokens  # (B, text_len, 96)
#         # 归一化 Query
#         norm_img_seq_query = self.cross_attn_norm(img_seq_query)
#         # 执行交叉注意力: 每个图像位置向所有文本 token 查询信息
#         # Q: img_seq, K: txt_kv, V: txt_kv
#         fused_seq, _ = self.cross_attn(
#             query=norm_img_seq_query,
#             key=txt_kv,
#             value=txt_kv
#         )
#         # 残差连接：将文本信息融入图像特征
#         refined_img_seq = img_seq_query + fused_seq
#         # 将序列恢复为图像特征图
#         refined_img_feat = refined_img_seq.permute(0, 2, 1).view(B, C, H, W)  # (B, 96, 28, 28)
#         # 5. (核心修改) 使用融合后的特征进行 refine
#         # 不再需要拼接，直接将融合了文本信息的图像特征传入
#         attention_map = self.mask_refine(refined_img_feat)  # (B, 1, H, W)
#         # # # ===== (改动结束) =====
#
#
#
#
#         ################下一段是原本的编码器部分
#         # x = self.downsample_layers[0](image)
#         # x = self.stages[0](x)
#         # res = x
#         # #下一段是原本的
#         # vis_feat, txt_feat = self.fusion1(x, txt)
#         # encoder_feats.append(vis_feat)    #原本是encoder_feats.append(vis_feat)    里面的vis-feat改为x是为了删去原论文的图文交叉融合(下面几个同理)
#         # x = self.downsample_layers[1](vis_feat + res)     #原本是x = self.downsample_layers[1](vis_feat + res)  里面的vis_feat + res改为x是为了删去原论文的图文交叉融合(下面几个同理)
#         # x = self.stages[1](x)
#         # res = x
#         # # 下一段是原本的
#         # vis_feat, txt_feat = self.fusion2(x, txt + txt_feat)
#         # encoder_feats.append(vis_feat)    #原本是encoder_feats.append(vis_feat)
#         #
#         # #下一行新增(解码器多尺度融合,轻量级)
#         # #encoder_feats[1] = self.encoder_fusion2(encoder_feats[:2], encoder_feats[1].shape[-2:])
#         #
#         # x = self.downsample_layers[2](vis_feat + res)   # x = self.downsample_layers[2](vis_feat + res)
#         # x = self.stages[2](x)
#         # res = x
#         # # 下一段是原本的
#         # vis_feat, txt_feat = self.fusion3(x, txt + txt_feat)
#         # encoder_feats.append(vis_feat)       #原本是encoder_feats.append(vis_feat)
#         #
#         # # 下一行新增(解码器多尺度融合,轻量级)
#         # #encoder_feats[2] = self.encoder_fusion3(encoder_feats[:3], encoder_feats[2].shape[-2:])
#         #
#         # x = self.downsample_layers[3](vis_feat + res)        # x = self.downsample_layers[3](vis_feat + res)
#         # x = self.stages[3](x)
#         # # 下一段是原本的
#         # vis_feat, txt_feat = self.fusion4(x, txt + txt_feat)
#         # encoder_feats.append(vis_feat)        # encoder_feats.append(vis_feat)
#
#
#         ####下一段是更改的融合残差叠加的显式图文融合的mask的编码器
#         # ===== 修改后的编码器部分（引入 Attention-Guided Residual Path）=====
#         #####下一段是原本的
#         # x = self.downsample_layers[0](image)
#         # x = self.stages[0](x)
#         # att_56 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         # x_att = x * att_56
#         # x_fused = x + x_att
#         # encoder_feats.append(x_fused)
#         # # Stage 1 (28x28)
#         # x = self.downsample_layers[1](x_fused)
#         # x = self.stages[1](x)
#         # att_28 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         # x_att = x * att_28
#         # x_fused = x + x_att
#         # encoder_feats.append(x_fused)
#         # # Stage 2 (14x14)
#         # x = self.downsample_layers[2](x_fused)
#         # x = self.stages[2](x)
#         # att_14 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         # x_att = x * att_14
#         # x_fused = x + x_att
#         # encoder_feats.append(x_fused)
#         # # Stage 3 (7x7)
#         # x = self.downsample_layers[3](x_fused)
#         # x = self.stages[3](x)
#         # att_7 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         # x_att = x * att_7
#         # x_fused = x + x_att
#         # encoder_feats.append(x_fused)
#
#
# ####### 下一段是修改后源代码!!!!!!!
#         #下一段是高仿源代码的编码层    编码层放x_att而非x_fused    !!!确定的代码
#         x = self.downsample_layers[0](image)
#         x = self.stages[0](x)
#         #下一行是原本的
#         att_56 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         x_att = x * att_56
#         # x_fused = x + x_att
#         encoder_feats.append(x_att)    # x_att
#         # Stage 1 (28x28)
#         x = self.downsample_layers[1](x_att)
#         x = self.stages[1](x)
#         # 下一行是原本的
#         att_28 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         x_att = x * att_28
#         # x_fused = x + x_att
#         encoder_feats.append(x_att)    # x_att
#         # Stage 2 (14x14)
#         x = self.downsample_layers[2](x_att)
#         x = self.stages[2](x)
#         # 下一行是原本的
#         att_14 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         x_att = x * att_14
#         # x_fused = x + x_att
#         encoder_feats.append(x_att)    # x_fused
#         # Stage 3 (7x7)
#         x = self.downsample_layers[3](x_att)
#         x = self.stages[3](x)
#         # 下一行是原本的
#         att_7 = F.interpolate(attention_map, size=x.shape[-2:], mode='bilinear', align_corners=True)
#         x_att = x * att_7
#         # x_fused = x + x_att
#         encoder_feats.append(x_att)    # x_fused
#
#
#         #下一段时不加显式图文融合mask的最简单纯粹的编码器
#         # x = self.downsample_layers[0](image)
#         # x = self.stages[0](x)
#         # # 下一行是原本的
#         # encoder_feats.append(x)  # 原本是 encoder_feats.append(x_att)
#         # # Stage 1 (28x28)
#         # x = self.downsample_layers[1](x)
#         # x = self.stages[1](x)
#         # # 下一行是原本的
#         # encoder_feats.append(x)
#         # # Stage 2 (14x14)
#         # x = self.downsample_layers[2](x)
#         # x = self.stages[2](x)
#         # # 下一行是原本的
#         # encoder_feats.append(x)
#         # # Stage 3 (7x7)
#         # x = self.downsample_layers[3](x)
#         # x = self.stages[3](x)
#         # # 下一行是原本的
#         # encoder_feats.append(x)
#
#
#
#
#
#
#
#
#
#         # #下一段是新增的显式图文融合的mask
#         # # # ===== 新增：Soft masking 融合 =====
#         # att_56 = F.interpolate(attention_map, size=(56, 56), mode='bilinear', align_corners=True)
#         # att_28 = F.interpolate(attention_map, size=(28, 28), mode='bilinear', align_corners=True)
#         # att_14 = F.interpolate(attention_map, size=(14, 14), mode='bilinear', align_corners=True)
#         # att_7 = F.interpolate(attention_map, size=( 7,  7), mode='bilinear', align_corners=True)
#         # encoder_feats[0] = encoder_feats[0] * att_56
#         # encoder_feats[1] = encoder_feats[1] * att_28
#         # encoder_feats[2] = encoder_feats[2] * att_14
#         # encoder_feats[3] = encoder_feats[3] * att_7
#
#
#         # # ===== 新增结束 =====
#
#
#
#         # 下一行新增(解码器多尺度融合,轻量级)
#         #encoder_feats[3] = self.encoder_fusion4(encoder_feats[:4], encoder_feats[3].shape[-2:])
#
#         # ===== 新增：DeepSeek LLM Block 处理 =====
#         # 获取最高层的编码特征
#         # high_level_feat = encoder_feats[-1]  # (B, 768, 7, 7)
#         # B, C, H, W = high_level_feat.shape
#         # num_visual_tokens = H * W  # 49
#         #
#         # # 1. Flatten 并转置: (B, C, H, W) -> (B, H*W, C)
#         # visual_tokens = high_level_feat.flatten(2).permute(0, 2, 1)  # (B, 49, 768)
#
#         #下一段是为了向llm中接入文本新增的
#         # 2. 准备文本 Token
#         # text_output['feature'][-1] 包含了BERT最后一层的输出, shape: (B, text_len, 768)
#         # text_tokens = text_output['feature'][-1]
#         # num_text_tokens = text_tokens.shape[1]
#         # # 3. (核心) 拼接视觉和文本 Token
#         # # 注意：确保 visual_tokens 和 text_tokens 的最后一个维度（特征维度）是相同的 (都是768)
#         # # 序列维度 (dim=1) 上拼接: [B, 49, 768] + [B, text_len, 768] -> [B, 49 + text_len, 768]
#         # multimodal_tokens = torch.cat([visual_tokens, text_tokens], dim=1)
#
#         #下一段是为了向llm中接入文本新增的
#         # 4. 投影到 LLM 维度
#         # llm_input = self.visual_to_llm(multimodal_tokens)  # (B, 49 + text_len, llm_hidden_dim)
#         #下一段是原本的
#         # llm_input = self.visual_to_llm(visual_tokens)  # (B, 49, llm_hidden_dim)
#
#         # 下一段是为了向llm中接入文本新增的
#         # 5. 构建动态的 position_ids 和 attention_mask
#         # total_seq_len = num_visual_tokens + num_text_tokens
#
#         # 下一段是为了向llm中接入文本新增的
#         # 构建 position_ids
#         # position_ids = torch.arange(total_seq_len, device=llm_input.device).unsqueeze(0).expand(B, -1)
#         #下一段是原本的
#         # 3. 构建位置和注意力信息
#         # position_ids = torch.arange(49, device=llm_input.device).unsqueeze(0).expand(B, -1)  # (B, 49)
#         # attn_mask = self.causal_mask.expand(B, -1, -1, -1).to(llm_input.device)  # (B, 1, 49, 49)
#
#         #下一段是为了向llm中接入文本新增的
#         # 构建 attention_mask
#         # 视觉部分: 全都互相可见 (全1矩阵)
#         # visual_mask = torch.ones(B, num_visual_tokens, device=llm_input.device)
#         # # 文本部分: 使用原始的 attention mask (处理padding)
#         # text_mask = text['attention_mask']  # (B, text_len), 1 for real tokens, 0 for padding
#         # # 拼接成一维的mask
#         # combined_1d_mask = torch.cat([visual_mask, text_mask], dim=1)  # (B, total_seq_len)
#         # # 扩展成LLM需要的4D attention mask (B, 1, To, From)
#         # attn_mask = combined_1d_mask[:, None, None, :].expand(B, 1, total_seq_len, total_seq_len).to(llm_input.device)
#         # # 这是最简单的全注意力mask。如果需要更复杂的causal mask，逻辑会更复杂。对于视觉任务，全注意力通常OK。
#         # # Hugging Face 的 LLM block 通常期望 additive mask (0 for attend, -inf for ignore)
#         # # 我们需要转换一下
#         # additive_attn_mask = (1.0 - attn_mask) * torch.finfo(llm_input.dtype).min
#
#         #下面一行在上面的注意力改为更适合视觉的注意力时替换掉上一行
#         # 下一段是原本的
#         # attn_mask = self.attn_mask.expand(B, -1, -1, -1).to(llm_input.device)  # (B, 1, 49, 49)
#
#         # 4. 调用 DeepSeek LLM Block（无 try-except 简化调用）
#         # llm_output = self.llm_block(
#         #     hidden_states=llm_input,
#         #     attention_mask=additive_attn_mask,   #原本是 attn_mask
#         #     position_ids=position_ids,
#         #     use_cache=False
#         # )
#         # if isinstance(llm_output, tuple):
#         #     llm_output = llm_output[0]
#
#         # 下一段是为了向llm中接入文本新增的
#         # 7. (核心) 从输出中分离出视觉 Token
#         # 输出的序列顺序和输入一致，所以前49个是处理后的视觉token
#         # processed_visual_tokens_llm_dim = llm_output[:, :num_visual_tokens, :]  # (B, 49, llm_hidden_dim)
#         #
#         #
#         # # 5. 投影回视觉特征维度
#         # refined_tokens = self.llm_to_visual(processed_visual_tokens_llm_dim)  # (B, 49, 768)
#         #下一行是原本的
#         # refined_tokens = self.llm_to_visual(llm_output)  # (B, 49, 768)
#         # 6. Reshape 回 feature map: (B, H*W, C) -> (B, C, H, W)
#         # refined_feat = refined_tokens.permute(0, 2, 1).view(B, C, H, W)
#         #
#         #
#         # # 下面一段是搞llm单项蒸馏新增的
#         # alignment_loss = self.alignment_loss_fn(high_level_feat, refined_feat)
#         # 策略A (推荐): 用“学生”自己的特征继续。
#         # 这样可以保证解码器是基于可训练的编码器进行优化的，更稳定。
#         # 协同训练损失会通过反向传播，间接将LLM的知识“注入”到学生编码器中。
#         # final_bottleneck_feat = high_level_feat
#         # 策略B: 用融合后的特征继续。
#         # 这种方式更直接，但可能因为老师的特征是固定的而导致训练不稳定。
#         # gate = 0.5*torch.sigmoid(self.llm_gate)
#         # final_bottleneck_feat = high_level_feat + gate * refined_feat
#         # # 我们将使用策略A，并更新encoder_feats
#         # encoder_feats[-1] = final_bottleneck_feat
#
#
#
#
#         # 7. 门控残差连接(带权重使用llm)(原本的连接文本的门控)
#         # gate =  0.3*torch.sigmoid(self.llm_gate) #原本是0.8*
#         # encoder_feats[-1] = high_level_feat + gate * refined_feat
#
#         # 完全使用 LLM 的 semantic-enhanced 输出
#         # encoder_feats[-1] = refined_feat  # 完全使用 LLM 的 semantic-enhanced 输出
#         # ===== 结束 =====
#
#         #下一段是原本的纯粹的解码器(显式图文融合的mask)
#         d4 = self.decode4(encoder_feats[3], encoder_feats[2])
#
#         #下面这段是新增的
#         # f3 = self.ms_fusion3([encoder_feats[3], encoder_feats[2]], d4.shape[-2:])
#         # f3 = self.norm_f3(f3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # ← 新增
#
#         d3 = self.decode3(d4, encoder_feats[1]) #原本是d3 = self.decode3(d4, encoder_feats[1])  d4+f3
#
#         # 下面这段是新增的
#         # f2 = self.ms_fusion2([encoder_feats[3], encoder_feats[2], encoder_feats[1]], d3.shape[-2:])
#         # f2 = self.norm_f2(f2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # ← 新增
#
#         d2 = self.decode2(d3, encoder_feats[0]) #原本是d2 = self.decode2(d3, encoder_feats[0])  d3+f2
#
#         # 下面这段是新增的
#         # f1 = self.ms_fusion1([encoder_feats[3], encoder_feats[2], encoder_feats[1], encoder_feats[0]], d2.shape[-2:])
#         # f1 = self.norm_f1(f1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # ← 新增
#
#         os1 = self.decoder1(d2) #原本是os1 = self.decoder1(d2)  d2+f1
#         out = self.out(os1).sigmoid()
#
#
#         #下一段是改的添加显式图文融合mask的 解码器  方式1
#         # 首先, 用 attention_map 来精炼 skip-connection 特征
#         # guided_skip_feat2 = encoder_feats[2] * att_14  # Soft masking
#         # # 或者使用残差方式: guided_skip_feat2 = encoder_feats[2] + encoder_feats[2] * att_14
#         # d4 = self.decode4(encoder_feats[3], guided_skip_feat2)
#         # # d3: 从 14x14 上采样到 28x28, 与 encoder_feats[1] (28x28) 融合
#         # guided_skip_feat1 = encoder_feats[1] * att_28
#         # d3 = self.decode3(d4, guided_skip_feat1)
#         # # d2: 从 28x28 上采样到 56x56, 与 encoder_feats[0] (56x56) 融合
#         # guided_skip_feat0 = encoder_feats[0] * att_56
#         # d2 = self.decode2(d3, guided_skip_feat0)
#         # # 最后的上采样和输出层
#         # os1 = self.decoder1(d2)
#         # out = self.out(os1).sigmoid()
#         # ===== 解码器结束 =====
#
#
#
#
#
#
#
#
#         ##### 第二个U-Net（下半部分）- 使用第一个U-Net的输出作为输入
#         # encoder_feats2 = []  # 为第二个U-Net创建特征列表
#         #
#         # # 将第一个U-Net的输出与原始图像拼接作为第二个U-Net的输入
#         # # 或者可以直接使用第一个U-Net的输出，这里选择拼接以保留更多信息
#         # second_input = torch.cat([image, out1], dim=1)
#         #
#         # # 第二个U-Net的编码路径
#         # x2 = self.downsample_layers2[0](second_input)  # 需要额外的下采样层
#         # x2 = self.stages2[0](x2)  # 需要额外的阶段层
#         # res2 = x2
#         #
#         # vis_feat2, txt_feat2 = self.fusion1_2(x2, txt)  # 需要额外的融合层
#         # encoder_feats2.append(vis_feat2)
#         # x2 = self.downsample_layers2[1](vis_feat2 + res2)
#         # x2 = self.stages2[1](x2)
#         # res2 = x2
#         #
#         # vis_feat2, txt_feat2 = self.fusion2_2(x2, txt + txt_feat2)
#         # encoder_feats2.append(vis_feat2)
#         # x2 = self.downsample_layers2[2](vis_feat2 + res2)
#         # x2 = self.stages2[2](x2)
#         # res2 = x2
#         #
#         # vis_feat2, txt_feat2 = self.fusion3_2(x2, txt + txt_feat2)
#         # encoder_feats2.append(vis_feat2)
#         # x2 = self.downsample_layers2[3](vis_feat2 + res2)
#         # x2 = self.stages2[3](x2)
#         #
#         # vis_feat2, txt_feat2 = self.fusion4_2(x2, txt + txt_feat2)
#         # encoder_feats2.append(vis_feat2)
#         #
#         # ##第二个U-Net的解码路径
#         # d4_2 = self.decode4_2(encoder_feats2[3], encoder_feats2[2])  # 需要额外的解码层
#         # d3_2 = self.decode3_2(d4_2, encoder_feats2[1])
#         # d2_2 = self.decode2_2(d3_2, encoder_feats2[0])
#         # os1_2 = self.decoder1_2(d2_2)
#         # out2 = self.out2(os1_2).sigmoid()  # 第二个U-Net的输出，也是最终输出
#
#
#
#
#         return  out, attention_map    ##原本是return out      return out, attention_map












#下面是跑baseline的参数量的时候的代码
