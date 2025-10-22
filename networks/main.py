from typing import Tuple, Union

import torch
import torch.nn as nn

from unetr_block_modify_V3 import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from unetr_block_modify_V3 import UnetrUpBlock1, UnetrUpBlock2, UnetrUpBlock3, UnetrUpBlock4
from dynunet_block import UnetOutBlock
from nets import ViT


class convblock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(convblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Tuple[int, int],  # 2D图像尺寸 (H, W)
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # 单通道输入、4通道输出，图像尺寸(96,96)，特征尺寸32，使用批归一化
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(224,224), feature_size=32, norm_name='batch')

            # 4通道输入、3通道输出，图像尺寸(128,128)，卷积位置嵌入，使用实例归一化
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(224,224), pos_embed='conv', norm_name='instance')

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate应在0到1之间")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden_size必须能被num_heads整除")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"不支持的位置嵌入类型: {pos_embed}")

        self.num_layers = 12
        self.patch_size = (16, 16)  # 2D patch尺寸
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )  # 2D特征图尺寸
        self.hidden_size = hidden_size
        self.classification = False
        ####################################################111111111###########################################################

        self.vit1 = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=2,
        )

        self.encoder11 = UnetrBasicBlock(
            spatial_dims=2,  # 2D空间维度
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder12 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder13 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder14 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )

        # 解码器模块（全部改为2D操作）
        self.decoder15 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder14 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder13 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder12 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # 输出层（2D）
        self.out1 = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)

        ####################################################22222222222#################################################
        # self.vit2 = self.vit1
        # self.encoder21 = self.encoder11
        # self.encoder22 = self.encoder12
        # self.encoder23 = self.encoder13
        # self.encoder24 = self.encoder14
        # self.decoder25 = self.decoder15
        # self.decoder24 = self.decoder14
        # self.decoder23 = self.decoder13
        # self.decoder22 = self.decoder12
        # self.out2 = self.out1
        #####################################################333333333333################################################
        # self.vit3 = self.vit1
        # self.encoder31 = self.encoder11
        # self.encoder32 = self.encoder12
        # self.encoder33 = self.encoder13
        # self.encoder34 = self.encoder14
        # self.decoder35 = self.decoder15
        # self.decoder34 = self.decoder14
        # self.decoder33 = self.decoder13
        # self.decoder32 = self.decoder12
        # self.out3 = self.out1
        ####################################################4444444444###########################################
        # self.vit4 = ViT(
        #     in_channels=1,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=2,
        # )
        self.encoder41 = UnetrBasicBlock(
            spatial_dims=2,  # 2D空间维度
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder42 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder43 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder44 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder45 = UnetrUpBlock1(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder44 = UnetrUpBlock2(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder43 = UnetrUpBlock3(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder42 = UnetrUpBlock4(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out4 = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)
        ######################################Conv-Conv-Conv-Conv-Conv-Conv-Conv-###############################################
        self.Conv5 = convblock(3, 1)

    ######################################End-End-End-End-End-End-End-End-End-End-##########################################

    def proj_feat(self, x, hidden_size, feat_size):
        """将Transformer输出的扁平特征重塑为2D特征图"""
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)  # 移除3D维度
        x = x.permute(0, 3, 1, 2).contiguous()  # 调整维度顺序为 (B, C, H, W)
        return x

    def load_from(self, weights):
        """加载预训练权重（适配2D位置嵌入）"""
        with torch.no_grad():
            # 加载patch嵌入权重
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_2d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # 加载Transformer块权重
            for bname, block in self.vit.blocks.named_children():
                block.loadFrom(weights, n_block=bname)

            # 加载Transformer最终归一化层权重
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, image_batch_a, image_batch_v, image_batch_d):
        ###############################111111111111################################################################
        x1, hidden_states_out1 = self.vit1(image_batch_a)
        x2, hidden_states_out2 = self.vit1(image_batch_v)
        x3, hidden_states_out3 = self.vit1(image_batch_d)

        enc11 = self.encoder11(image_batch_a)
        enc21 = self.encoder11(image_batch_v)
        enc31 = self.encoder11(image_batch_d)

        x12 = hidden_states_out1[3]
        x22 = hidden_states_out2[3]
        x32 = hidden_states_out3[3]

        enc12 = self.encoder12(self.proj_feat(x12, self.hidden_size, self.feat_size))
        enc22 = self.encoder12(self.proj_feat(x22, self.hidden_size, self.feat_size))
        enc32 = self.encoder12(self.proj_feat(x32, self.hidden_size, self.feat_size))

        x13 = hidden_states_out1[6]
        x23 = hidden_states_out2[6]
        x33 = hidden_states_out3[6]

        enc13 = self.encoder13(self.proj_feat(x13, self.hidden_size, self.feat_size))
        enc23 = self.encoder13(self.proj_feat(x23, self.hidden_size, self.feat_size))
        enc33 = self.encoder13(self.proj_feat(x33, self.hidden_size, self.feat_size))

        x14 = hidden_states_out1[9]
        x24 = hidden_states_out2[9]
        x34 = hidden_states_out3[9]

        enc14 = self.encoder14(self.proj_feat(x14, self.hidden_size, self.feat_size))
        enc24 = self.encoder14(self.proj_feat(x24, self.hidden_size, self.feat_size))
        enc34 = self.encoder14(self.proj_feat(x34, self.hidden_size, self.feat_size))

        dec14 = self.proj_feat(x1, self.hidden_size, self.feat_size)
        dec24 = self.proj_feat(x2, self.hidden_size, self.feat_size)
        dec34 = self.proj_feat(x3, self.hidden_size, self.feat_size)

        dec13 = self.decoder15(dec14, enc14)
        dec23 = self.decoder15(dec24, enc24)
        dec33 = self.decoder15(dec34, enc34)

        dec12 = self.decoder14(dec13, enc13)
        dec22 = self.decoder14(dec23, enc23)
        dec32 = self.decoder14(dec33, enc33)

        dec11 = self.decoder13(dec12, enc12)
        dec21 = self.decoder13(dec22, enc22)
        dec31 = self.decoder13(dec32, enc32)

        out1 = self.decoder12(dec11, enc11)
        out2 = self.decoder12(dec21, enc21)
        out3 = self.decoder12(dec31, enc31)

        ###############################4444444444444################################################################
        image_batch_adv = torch.cat((image_batch_a, image_batch_v, image_batch_d), dim=1)
        image_batch_adv = self.Conv5(image_batch_adv)
        # x4, hidden_states_out4 = self.vit4(image_batch_adv)  # ViT输出
        x4, hidden_states_out4 = self.vit1(image_batch_adv)  # ViT输出
        enc41 = self.encoder41(image_batch_adv)

        x42 = hidden_states_out4[3]
        enc42 = self.encoder42(self.proj_feat(x42, self.hidden_size, self.feat_size))

        x43 = hidden_states_out4[6]
        enc43 = self.encoder43(self.proj_feat(x43, self.hidden_size, self.feat_size))

        x44 = hidden_states_out4[9]
        enc44 = self.encoder44(self.proj_feat(x44, self.hidden_size, self.feat_size))

        dec44 = self.proj_feat(x4, self.hidden_size, self.feat_size)
        dec43 = self.decoder45(dec44, enc44, dec13, dec23, dec33)
        dec42 = self.decoder44(dec43, enc43, dec12, dec22, dec32)
        dec41 = self.decoder43(dec42, enc42, dec11, dec21, dec31)
        out4 = self.decoder42(dec41, enc41, out1, out2, out3)
        logits4 = self.out4(out4)
        return logits4
###############################-end-end-end-end-end-end-################################################################

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device -> {device}")

    model = UNETR(in_channels=1, out_channels=3, img_size=(224, 224), feature_size=16).to(device)
    model.eval()

    x = torch.randn((1, 1, 224, 224), device=device)
    inputs = (x, x.clone(), x.clone())

    # -----------------------------
    # Params 统计
    # -----------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params/1e6:.4f} M")

    # -----------------------------
    # FLOPs 统计 (含 Self-Attention)
    # -----------------------------
    flops_dict = {}


    def conv_hook(self, input, output):
        batch_size, Cin, H, W = input[0].shape
        Cout, _, kH, kW = self.weight.shape
        Hout, Wout = output.shape[2:]
        flops = batch_size * Cout * Hout * Wout * (Cin * kH * kW // self.groups * 2)  # 乘加
        flops_dict[self] = flops


    def linear_hook(self, input, output):
        batch_size = input[0].shape[0]
        in_features = self.in_features
        out_features = self.out_features
        flops = batch_size * in_features * out_features * 2  # 乘+加
        flops_dict[self] = flops


    def bn_hook(self, input, output):
        flops = input[0].numel()
        flops_dict[self] = flops


    def ln_hook(self, input, output):
        # LayerNorm ~ 5 * num_features (减均值, 方差, 除法, 乘缩放, 加偏置)
        flops = input[0].numel() * 5
        flops_dict[self] = flops


    def relu_hook(self, input, output):
        flops = input[0].numel()
        flops_dict[self] = flops


    def gelu_hook(self, input, output):
        # GELU 近似按 4 ops / element
        flops = input[0].numel() * 4
        flops_dict[self] = flops


    def pool_hook(self, input, output):
        flops = output.numel()
        flops_dict[self] = flops


    def convtranspose_hook(self, input, output):
        batch_size, Cin, H, W = input[0].shape
        Cout, _, kH, kW = self.weight.shape
        Hout, Wout = output.shape[2:]
        flops = batch_size * Cout * Hout * Wout * (Cin * kH * kW // self.groups * 2)
        flops_dict[self] = flops


    # -----------------------------
    # 专门的 Attention Hook
    # -----------------------------
    def attention_hook(self, input, output):
        # 假设 input[0]: (B, N, D)
        B, N, D = input[0].shape
        num_heads = self.num_heads
        head_dim = D // num_heads

        # Q,K,V projections: 3 * (B * N * D * D)
        qkv_flops = 3 * B * N * D * D

        # Attention(QK^T): B * num_heads * N * N * head_dim
        attn_flops = B * num_heads * (N * N * head_dim * 2)  # 乘+加

        # Attention * V: B * num_heads * N * N * head_dim
        attn_v_flops = B * num_heads * (N * N * head_dim * 2)

        # Output projection: B * N * D * D
        proj_flops = B * N * D * D * 2

        flops = qkv_flops + attn_flops + attn_v_flops + proj_flops
        flops_dict[self] = flops


    # -----------------------------
    # 注册 hooks
    # -----------------------------
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        elif isinstance(m, nn.ConvTranspose2d):
            m.register_forward_hook(convtranspose_hook)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(linear_hook)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            m.register_forward_hook(bn_hook)
        elif isinstance(m, nn.LayerNorm):
            m.register_forward_hook(ln_hook)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(relu_hook)
        elif isinstance(m, nn.GELU):
            m.register_forward_hook(gelu_hook)
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            m.register_forward_hook(pool_hook)

        # 针对 Attention (假设 UNETR 用 nn.MultiheadAttention 或自定义 AttentionBlock)
        if m.__class__.__name__ in ["MultiheadAttention", "Attention", "SelfAttention", "MHSA"]:
            m.register_forward_hook(attention_hook)
    # 前向推理一次，触发 hooks
    with torch.no_grad():
        model(*inputs)

    import numpy as np

    total_flops = np.sum(list(flops_dict.values()), dtype=np.float64)
    print(f"总 FLOPs: {total_flops / 1e9:.4f} GFLOPs")
    # print(f"总 FLOPs: {total_flops/1e9:.4f} GFLOPs")

