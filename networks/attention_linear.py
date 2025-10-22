import torch
import torch.nn as nn


class MultiSinglePeriodCrossAttention(nn.Module):
    """
    三个单期两两先做交叉注意力实现局部信息交互，
    再进行 单期↔多期(融合特征) 交互；两部分结果均作残差回连。
    输入:
        t1, t2, t3: [B, C, H, W]  # 单期特征
        fusion_tensor: [B, C, H, W]  # 多期/融合特征
    输出:
        fused_out: [B, C, H, W]
    """
    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # 延后按channels初始化
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.back_proj = None
        self.out_proj = None
        self.channels = None

        self.attn_dropout = nn.Dropout(dropout)
        self.res_dropout = nn.Dropout(dropout)

    # ---------- 工具函数 ----------
    def _init_projections(self, channels, device):
        if self.channels != channels or self.q_proj is None:
            self.q_proj = nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False).to(device)
            self.k_proj = nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False).to(device)
            self.v_proj = nn.Conv2d(channels, self.d_model, kernel_size=1, bias=False).to(device)
            self.back_proj = nn.Conv2d(self.d_model, channels, kernel_size=1, bias=False).to(device)
            # 拼接三个单期后的还原
            self.out_proj = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False).to(device)
            self.channels = channels

    @staticmethod
    def _to_seq(x):
        # [B, C, H, W] -> [B, N, C]
        B, C, H, W = x.shape
        N = H * W
        return x.view(B, C, N).permute(0, 2, 1), H, W

    @staticmethod
    def _from_seq(x_seq, H, W):
        # [B, N, C] -> [B, C, H, W]
        B, N, C = x_seq.shape
        return x_seq.permute(0, 2, 1).contiguous().view(B, C, H, W)

    # ---------- 线性注意力（避免 N×N） ----------
    def cross_attn_linear(self, q_feat, kv_feat):
        """
        对 q_feat 以 kv_feat 为 K/V 做跨注意力（线性注意力近似）
        输入输出均为 [B, C, H, W]，内部投影到 d_model 后再回投到 C。
        """
        q = self.q_proj(q_feat)         # [B, D, H, W]
        k = self.k_proj(kv_feat)        # [B, D, H, W]
        v = self.v_proj(kv_feat)        # [B, D, H, W]

        q_seq, H, W = self._to_seq(q)   # [B, N, D]
        k_seq, _, _ = self._to_seq(k)
        v_seq, _, _ = self._to_seq(v)

        B, N, D = q_seq.shape
        Nh, Hd = self.num_heads, self.head_dim

        qh = q_seq.view(B, N, Nh, Hd)   # [B, N, Nh, Hd]
        kh = k_seq.view(B, N, Nh, Hd)
        vh = v_seq.view(B, N, Nh, Hd)

        # 正值映射近似 softmax（Performer/linear attention 思路）
        qh = torch.relu(qh) + 1e-6
        kh = torch.relu(kh) + 1e-6

        # 计算 K^T V： [B, Nh, Hd, Hd]
        kv = torch.einsum("bnhd,bnhm->bhdm", kh, vh)

        # 归一化分母 Z： [B, N, Nh]
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", qh, kh.sum(dim=1)) + 1e-6)

        # 输出： [B, N, Nh, Hd]
        out = torch.einsum("bnhd,bhdm,bnh->bnhm", qh, kv, z)
        out = out.reshape(B, N, D)  # [B, N, D]
        out_spatial = self._from_seq(out, H, W)  # [B, D, H, W]
        out_spatial = self.attn_dropout(out_spatial)
        return self.back_proj(out_spatial)       # [B, C, H, W]

    # ---------- 前向 ----------
    def forward(self, t1, t2, t3, fusion_tensor):
        """
        Stage-A：单期两两交叉注意力（t1↔t2, t1↔t3, t2↔t3 的定向交互，聚合为对每个单期的更新）
        Stage-B：单期与多期（fusion）交叉注意力
        Residual：将 Stage-A 与 Stage-B 的结果分别残差回连到对应输入；最终输出再与 fusion_tensor 残差。
        """
        device = t1.device
        self._init_projections(t1.shape[1], device)

        # -------- Stage-A：单期↔单期（定向：a<-b） --------
        # t1 从 t2、t3 接收信息
        t1_from_t2 = self.cross_attn_linear(t1, t2)
        t1_from_t3 = self.cross_attn_linear(t1, t3)
        t1_pair = self.res_dropout(t1_from_t2 + t1_from_t3)

        # t2 从 t1、t3 接收信息
        t2_from_t1 = self.cross_attn_linear(t2, t1)
        t2_from_t3 = self.cross_attn_linear(t2, t3)
        t2_pair = self.res_dropout(t2_from_t1 + t2_from_t3)

        # t3 从 t1、t2 接收信息
        t3_from_t1 = self.cross_attn_linear(t3, t1)
        t3_from_t2 = self.cross_attn_linear(t3, t2)
        t3_pair = self.res_dropout(t3_from_t1 + t3_from_t2)

        # 残差回连到各自单期
        t1_after_pair = t1 + t1_pair
        t2_after_pair = t2 + t2_pair
        t3_after_pair = t3 + t3_pair

        # -------- Stage-B：单期↔多期（fusion） --------
        t1_fuse = self.cross_attn_linear(t1_after_pair, fusion_tensor)
        t2_fuse = self.cross_attn_linear(t2_after_pair, fusion_tensor)
        t3_fuse = self.cross_attn_linear(t3_after_pair, fusion_tensor)

        # 再次残差回连到各自单期
        t1_final = t1_after_pair + self.res_dropout(t1_fuse)
        t2_final = t2_after_pair + self.res_dropout(t2_fuse)
        t3_final = t3_after_pair + self.res_dropout(t3_fuse)

        # -------- 融合输出 & 融合端残差 --------
        # 将更新后的三路单期拼接并还原到 C 通道
        concat_out = torch.cat([t1_final, t2_final, t3_final], dim=1)  # [B, 3C, H, W]
        fused = self.out_proj(concat_out)                              # [B, C, H, W]

        # 对融合端也做残差回连
        fused_out = fused + fusion_tensor

        return fused_out
