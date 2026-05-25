import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================================= #
#  0. Environment & Backbone Setup (MiT-b1)
# ================================================================================= #
try:
    from mmseg.models.backbones import MixVisionTransformer
    MMSEG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import MixVisionTransformer: {e}")
    print("Please install: pip install ftfy mmsegmentation mmcv")
    MixVisionTransformer = None
    MMSEG_AVAILABLE = False

class MiT_b1_Backbone(nn.Module):
    def __init__(self, pretrained=None):
        super(MiT_b1_Backbone, self).__init__()
        
        if not MMSEG_AVAILABLE or MixVisionTransformer is None:
            raise ImportError("MixVisionTransformer not available. Please install mmsegmentation.")
        
        self.backbone = MixVisionTransformer(
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )
        
        if pretrained is not None:
            self._load_pretrained(pretrained)
    
    def _load_pretrained(self, pretrained):
        try:
            if pretrained.startswith('http'):
                import torch.utils.model_zoo as model_zoo
                state_dict = model_zoo.load_url(pretrained)
            else:
                state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained weights from {pretrained}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
    
    def forward(self, x):
        features = self.backbone(x)
        return features


# ================================================================================= #
#  1. Helper Modules (ConvBlocks, SimAM)
# ================================================================================= #
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu2(self.bn2(self.pointwise(self.relu1(self.bn1(self.depthwise(x))))))

class ConvModule(nn.Module):
    def __init__(self, in_channels, use_dsconv=True):
        super(ConvModule, self).__init__()
        if use_dsconv:
            self.conv = DepthwiseSeparableConv(in_channels, in_channels)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return self.activaton(y)


# ================================================================================= #
#  2. MSTIEM
# ================================================================================= #
class EMSA(nn.Module):
    def __init__(self, channels, groups=16):
        super(EMSA, self).__init__()
        self.groups = groups
        self.step = channels // groups
        assert channels % groups == 0
        assert self.step > 0

        self.softmax = nn.Softmax(-1)
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        
        self.conv1x1 = nn.Conv2d(self.step, self.step, kernel_size=1, stride=1, padding=0)
        self.gn_shared = nn.GroupNorm(1, self.step)
        self.act_shared = nn.GELU()
        
        self.large_kernel_branch = nn.Sequential(
            nn.Conv2d(self.step, self.step, kernel_size=5, stride=1, padding=2, groups=self.step, bias=False),
            nn.Conv2d(self.step, self.step, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, self.step),
            nn.GELU()
        )
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        
        # Branch A
        x_h = self.pool_h_avg(group_x) + self.pool_h_max(group_x)
        x_w = self.pool_w_avg(group_x) + self.pool_w_max(group_x)
        x_w = x_w.permute(0, 1, 3, 2)
        
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        hw = self.act_shared(self.gn_shared(hw))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        
        attn_h = x_h.sigmoid()
        attn_w = x_w.permute(0, 1, 3, 2).sigmoid()
        f_a = group_x * attn_h * attn_w

        # Branch B
        f_b = self.large_kernel_branch(group_x)
        
        # Cross-Spatial Fusion
        global_a = self.softmax(self.agp(f_a).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        global_b = self.softmax(self.agp(f_b).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        
        flat_a = f_a.reshape(b * self.groups, self.step, -1)
        flat_b = f_b.reshape(b * self.groups, self.step, -1)
        
        map_a = torch.matmul(global_a, flat_b).reshape(b * self.groups, 1, h, w)
        map_b = torch.matmul(global_b, flat_a).reshape(b * self.groups, 1, h, w)
        
        y_sub = group_x * (map_a + map_b).sigmoid()
        
        # Channel Shuffle & Reshape
        y = y_sub.view(b, self.groups, self.step, h, w)
        y = torch.transpose(y, 1, 2).contiguous()
        y = y.view(b, c, h, w)
        return y

class MC3DDEM(nn.Module):
    def __init__(self, in_channels, factor=16):
        super(MC3DDEM, self).__init__()
        self.ema = EMSA(in_channels, groups=factor)
        
        self.conv_main1 = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1)
        self.bn_main1 = nn.BatchNorm2d(in_channels)
        self.relu_main = nn.ReLU(inplace=True)
        self.conv_main2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn_main2 = nn.BatchNorm2d(in_channels)
        
        self.conv_res = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)
        self.bn_res = nn.BatchNorm2d(in_channels)
        self.simam = SimAM()

    def forward(self, x1, x2):
        F_diff = torch.abs(x1 - x2)
        F_diff_att = self.ema(F_diff)
        
        F_cat = torch.cat([x1, x2], dim=1)
        F_main = self.bn_main2(self.conv_main2(self.relu_main(self.bn_main1(self.conv_main1(F_cat)))))
        F_res = self.bn_res(self.conv_res(F_cat))
        
        F_ms = F_main + F_res
        W_simam = self.simam(F_ms)
        F_conn = F_ms * W_simam
        
        return F_diff_att + F_conn


# ================================================================================= #
#  3. Dual-Domain Edge Guidance Module，DDEGM
# ================================================================================= #

class D2_LDEE(nn.Module):
    def __init__(self, in_channels):
        super(D2_LDEE, self).__init__()
        
        self.h_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.o_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.anti_alias = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, 
                                    groups=in_channels, bias=False)
        self.freq_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gamma = nn.Parameter(torch.zeros(1)) 

    def haar_wavelet(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        
        LL = (x00 + x01 + x10 + x11) / 4.0
        LH = (x00 + x10) - (x01 + x11)
        HL = (x00 + x01) - (x10 + x11)
        HH = (x00 + x11) - (x01 + x10)
        
        return LL, LH / 2.0, HL / 2.0, HH / 2.0

    def forward(self, x):
        s_h = self.h_conv(x)
        s_v = self.v_conv(x)
        s_o = self.o_conv(x)
        F_spatial = self.spatial_fuse(torch.cat([s_h, s_v, s_o], dim=1))
        
        LL, LH, HL, HH = self.haar_wavelet(x)
        F_low = self.low_freq_conv(LL)
        F_high = self.high_freq_conv(torch.cat([LH, HL, HH], dim=1))
        
        size = x.shape[2:]
        F_low_up = F.interpolate(F_low, size=size, mode='bilinear', align_corners=True)
        F_high_up = F.interpolate(F_high, size=size, mode='bilinear', align_corners=True)
        F_high_up = self.anti_alias(F_high_up)
        
        F_freq = self.freq_fuse(torch.cat([F_low_up, F_high_up], dim=1))
        
        F_edge = F_spatial + self.gamma * F_freq
        return F_edge

class EGFI(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super(EGFI, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, edge_feature):
        B, C, H, W = x.size()
        _, C_r, _, _ = edge_feature.size()
        Q = self.q_conv(x).view(B, C_r, H * W)
        K = self.k_conv(edge_feature).view(B, C_r, H * W)
        V = self.v_conv(edge_feature).view(B, C, H * W)
        
        Energy = torch.bmm(Q.transpose(1, 2), K)
        Attention = F.softmax(Energy, dim=-1)
        
        F_inject = torch.bmm(V, Attention.transpose(1, 2)).view(B, C, H, W)
        return x + self.alpha * self.out_proj(F_inject)

class MDAEG(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(MDAEG, self).__init__()
        reduced_channels = in_channels // reduction_ratio
        
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        self.ldee = D2_LDEE(reduced_channels)
        self.egfi = EGFI(in_channels, reduced_channels)
        self.post_process = DepthwiseSeparableConv(in_channels, in_channels)
    
    def forward(self, x):
        F_in = self.channel_reduction(x)
        F_edge = self.ldee(F_in)
        X_refined = self.egfi(x, F_edge)
        return self.post_process(X_refined)


# ================================================================================= #
#  4. Bidirectional Guided Multi-Granularity Aggregation Module，BGMGA
# ================================================================================= #
class BG_CAA(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, mid_channels=64):
        super(BG_CAA, self).__init__()
        self.align_high = nn.Sequential(
            nn.Conv2d(in_channels_high, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.align_low = nn.Sequential(
            nn.Conv2d(in_channels_low, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.cgp_gap = nn.AdaptiveAvgPool2d(1)
        self.cgp_gmp = nn.AdaptiveMaxPool2d(1)
        self.cgp_mlp = nn.Sequential(
            nn.Linear(mid_channels, mid_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels // 4, mid_channels),
            nn.Sigmoid()
        )
        self.sgp_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.aggregation = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, F_high, F_low):
        F_h = self.align_high(F_high)
        F_l = self.align_low(F_low)
        H, W = F_low.size()[2:]
        F_h_up = F.interpolate(F_h, size=(H, W), mode='bilinear', align_corners=True)
        
        v_semantic = (self.cgp_gap(F_h) + self.cgp_gmp(F_h)).view(F_h.size(0), -1)
        W_channel = self.cgp_mlp(v_semantic).view(F_h.size(0), -1, 1, 1)
        F_l_refined = F_l * W_channel + F_l
        
        F_spatial = torch.cat([torch.max(F_l, 1, keepdim=True)[0], torch.mean(F_l, 1, keepdim=True)], dim=1)
        F_h_refined = F_h_up * self.sgp_conv(F_spatial) + F_h_up
        
        return self.aggregation(torch.cat([F_h_refined, F_l_refined], dim=1))


# ================================================================================= #
#  5. Main Network: STDDF_MiT
# ================================================================================= #
class STDDF_MiT(nn.Module):
    def __init__(self, pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'):
        super(STDDF_MiT, self).__init__()
        
        self.backbone = MiT_b1_Backbone(pretrained=pretrained)
        
        # Difference Enhancement Modules
        self.dem1 = MC3DDEM(64, factor=16)
        self.dem2 = MC3DDEM(128, factor=16)
        self.dem3 = MC3DDEM(320, factor=16)
        self.dem4 = MC3DDEM(512, factor=16)
        
        # Edge Guidance Modules (Using updated MDAEG with D2_LDEE)
        self.egam1 = MDAEG(64)
        self.egam2 = MDAEG(128)
        self.egam3 = MDAEG(320)
        self.egam4 = MDAEG(512)
        
        # Convolutions
        self.conv1 = ConvModule(64, use_dsconv=True)
        self.conv2 = ConvModule(128, use_dsconv=True)
        self.conv3 = ConvModule(320, use_dsconv=True)
        self.conv4 = ConvModule(512, use_dsconv=True)
        
        # Aggregation Modules
        self.bgcaa43 = BG_CAA(512, 320, 320)
        self.bgcaa32 = BG_CAA(320, 128, 128)
        self.bgcaa21 = BG_CAA(128, 64, 64)
        self.bgcaa31 = BG_CAA(320, 64, 64)
        self.bgcaa41 = BG_CAA(512, 64, 64)
        
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, t1_input, t2_input):
        # Extract Features
        t1_list = self.backbone(t1_input)
        t2_list = self.backbone(t2_input)
        t1_x1, t1_x2, t1_x3, t1_x4 = t1_list[0], t1_list[1], t1_list[2], t1_list[3]
        t2_x1, t2_x2, t2_x3, t2_x4 = t2_list[0], t2_list[1], t2_list[2], t2_list[3]
        
        # Difference Enhancement
        f1 = self.dem1(t1_x1, t2_x1)
        f2 = self.dem2(t1_x2, t2_x2)
        f3 = self.dem3(t1_x3, t2_x3)
        f4 = self.dem4(t1_x4, t2_x4)
        
        # Edge Guidance
        fe1 = self.egam1(f1)
        fe2 = self.egam2(f2)
        fe3 = self.egam3(f3)
        fe4 = self.egam4(f4)
        
        # Decoding
        ff4 = self.conv4(fe4)
        
        ff3 = self.bgcaa43(ff4, fe3)
        ff3 = self.conv3(ff3)
        
        ff2 = self.bgcaa32(ff3, fe2)
        ff2 = self.conv2(ff2)
        
        ff1 = self.bgcaa21(ff2, fe1)
        ff1 = self.conv1(ff1)
        
        # Deep Supervision Paths
        ff2_up = self.bgcaa21(ff2, ff1)
        ff3_up = self.bgcaa31(ff3, ff1)
        ff4_up = self.bgcaa41(ff4, ff1)
        
        ff = ff1 + ff2_up + ff3_up + ff4_up
        
        # Final Output (Upsample to original size)
        # MiT Stage 1 is 1/4 resolution, so we need 4x upsampling
        ff_out = F.interpolate(self.out(ff), scale_factor=4, mode='bilinear', align_corners=True)
        ff_out = torch.sigmoid(ff_out)
        
        # Auxiliary Outputs
        ff2_out = F.interpolate(self.out(ff2_up), scale_factor=4, mode='bilinear', align_corners=True)
        ff2_out = torch.sigmoid(ff2_out)
        
        ff3_out = F.interpolate(self.out(ff3_up), scale_factor=4, mode='bilinear', align_corners=True)
        ff3_out = torch.sigmoid(ff3_out)
        
        ff4_out = F.interpolate(self.out(ff4_up), scale_factor=4, mode='bilinear', align_corners=True)
        ff4_out = torch.sigmoid(ff4_out)
        
        return ff_out, ff2_out, ff3_out, ff4_out


if __name__ == '__main__':
    print("Initializing Model...")
    try:
        model = STDDF_MiT(pretrained=None)
        t1 = torch.randn(2, 3, 256, 256)
        t2 = torch.randn(2, 3, 256, 256)
        
        print("Forward Pass...")
        outputs = model(t1, t2)
        
        print("--- Output Shapes ---")
        for i, out in enumerate(outputs):
            print(f"Output {i}: {out.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")
        
    except ImportError as e:
        print(f"Skipping test due to missing dependencies: {e}")
    except Exception as e:
        print(f"Runtime Error: {e}")