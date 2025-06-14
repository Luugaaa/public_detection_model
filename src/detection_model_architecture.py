import torch
import torch.nn as nn
import math
from collections import OrderedDict
from torchvision.ops import DeformConv2d

# ==============================================================================
# Helper Modules (DropPath, ConvBNAct, SeparableConvBlock)
# ==============================================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f'drop_prob={round(self.drop_prob,3)}'

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=nn.SiLU(inplace=True)):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.SiLU(inplace=True)):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, kernel_size, stride, groups=in_channels, activation=activation)
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, groups=1, activation=activation)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        x = self.dcn(x, offset, mask)
        return x

# ==============================================================================
# C2f and GELAN Blocks
# ==============================================================================
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class GELANBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_blocks=2, use_separable_conv=True, dropout_rate=0.0):
        super().__init__()
        Conv = SeparableConvBlock if use_separable_conv else ConvBNAct
        
        self.conv_in = ConvBNAct(in_channels, mid_channels, kernel_size=1)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    Conv(mid_channels, mid_channels, kernel_size=3),
                    Conv(mid_channels, mid_channels, kernel_size=3)
                )
            )

        self.conv_out = ConvBNAct(mid_channels * (num_blocks + 1), out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x_in = self.conv_in(x)
        features = [x_in]
        current_feature = x_in
        for block in self.blocks:
            current_feature = block(current_feature)
            features.append(current_feature)
        
        out = torch.cat(features, dim=1)
        out = self.conv_out(out)
        out = self.dropout(out)
        return out

# ==============================================================================
# CSP_GELAN_Block
# ==============================================================================
class CSP_GELAN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_gelan_internal_blocks=2, use_separable_conv=True, dropout_rate=0.0, drop_path_rate=0.0, use_dcn=False):
        super().__init__()
        mid_channels = out_channels // 2

        self.downsample = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2) 
        self.branch1_conv = ConvBNAct(out_channels, mid_channels, kernel_size=1)
        self.branch2_conv = ConvBNAct(out_channels, mid_channels, kernel_size=1)

        if use_dcn:
            self.gelan_block = nn.Sequential(
                GELANBlock(
                    mid_channels, mid_channels, mid_channels,
                    num_blocks=num_gelan_internal_blocks,
                    use_separable_conv=use_separable_conv,
                    dropout_rate=dropout_rate
                ),
                DCNv2(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
             self.gelan_block = GELANBlock(
                mid_channels, mid_channels, mid_channels,
                num_blocks=num_gelan_internal_blocks,
                use_separable_conv=use_separable_conv,
                dropout_rate=dropout_rate
            )

        self.fusion_conv = ConvBNAct(mid_channels * 2, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        branch1_in = self.branch1_conv(x)
        branch2 = self.branch2_conv(x) 
        branch1_out = self.gelan_block(branch1_in) 
        out = torch.cat([branch1_out, branch2], dim=1)
        out = self.fusion_conv(out)
        out = self.dropout(out) 
        return out

# ==============================================================================
# SPPF
# ==============================================================================
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, kernel_size=1)
        self.cv2 = ConvBNAct(c_ * 4, c2, kernel_size=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# ==============================================================================
# Backbone
# ==============================================================================
class EnhancedBackbone(nn.Module):
    def __init__(self, base_channels=64, num_csp_gelan_blocks=[3,3,3,3], 
                 feat_channels_proj=256,
                 use_separable_conv_backbone=True,
                 dropout_rate=0.0,
                 drop_path_rate=0.0,
                 use_dcn_backbone=True):
        super().__init__()
        print(f"Backbone: SeparableConvs: {use_separable_conv_backbone}, Dropout: {dropout_rate}, DropPath: {drop_path_rate}, ProjChannels: {feat_channels_proj}, DCN: {use_dcn_backbone}")

        self.stem = nn.Sequential(
             ConvBNAct(3, base_channels // 2, kernel_size=3, stride=2),
             ConvBNAct(base_channels // 2, base_channels, kernel_size=3, stride=1),
             ConvBNAct(base_channels, base_channels, kernel_size=3, stride=1)
        )

        c1_channels = base_channels
        c2_channels = base_channels * 2
        c3_channels = base_channels * 4
        c4_channels = base_channels * 8
        c5_channels = base_channels * 16

        self.c2f = C2f(c1_channels, c1_channels, n=2, shortcut=True)
        
        self.csp_gelan1 = CSP_GELAN_Block(c1_channels, c2_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[0], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
        self.csp_gelan2 = CSP_GELAN_Block(c2_channels, c3_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[1], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
        self.csp_gelan3 = CSP_GELAN_Block(c3_channels, c4_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[2], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
        self.csp_gelan4 = CSP_GELAN_Block(c4_channels, c5_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[3], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
        self.sppf = SPPF(c5_channels, c5_channels)

        self.proj_p3 = ConvBNAct(c3_channels, feat_channels_proj, kernel_size=1)
        self.proj_p4 = ConvBNAct(c4_channels, feat_channels_proj, kernel_size=1)
        self.proj_p5 = ConvBNAct(c5_channels, feat_channels_proj, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.c2f(x)
        x = self.csp_gelan1(x)
        p3_feat = self.csp_gelan2(x)
        p4_feat = self.csp_gelan3(p3_feat)
        p5_feat = self.csp_gelan4(p4_feat)
        p5_sppf = self.sppf(p5_feat)

        p3 = self.proj_p3(p3_feat)
        p4 = self.proj_p4(p4_feat)
        p5 = self.proj_p5(p5_sppf)

        features = OrderedDict([('p3', p3), ('p4', p4), ('p5', p5)])
        return features

# ==============================================================================
# BiFPN Neck
# ==============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_separable_conv=True, drop_path_rate=0.0, use_dcn=False):
        super().__init__()
        if use_dcn:
            self.conv1 = DCNv2(channels, channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = DCNv2(channels, channels, kernel_size=3, stride=1, padding=1)
        else:
            Conv = SeparableConvBlock if use_separable_conv else ConvBNAct
            self.conv1 = Conv(channels, channels, kernel_size=3)
            self.conv2 = Conv(channels, channels, kernel_size=3)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop_path(out)
        return out + residual

class WeightedFeatureFusion(nn.Module):
    def __init__(self, num_inputs, feat_channels, epsilon=1e-6):
        super().__init__()
        self.num_inputs = num_inputs
        self.feat_channels = feat_channels
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.weight_act = nn.ReLU(inplace=False)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs, f"Incorrect number of inputs {len(inputs)}, expected {self.num_inputs}"
        normalized_weights = self.weight_act(self.weights)
        normalized_weights = normalized_weights / (torch.sum(normalized_weights, dim=0) + self.epsilon)
        dtype = inputs[0].dtype
        fused_output = torch.zeros_like(inputs[0], dtype=dtype)
        for i in range(self.num_inputs):
            w = normalized_weights[i].view(1, -1, 1, 1) if inputs[i].ndim == 4 else normalized_weights[i]
            fused_output += inputs[i] * w.to(dtype)
        return fused_output

class EnhancedBiFPNBlock(nn.Module):
    def __init__(self, feat_channels, num_levels=3, use_separable_conv_neck=True, epsilon=1e-4, dropout_rate=0.0, drop_path_rate=0.0, use_dcn_neck=False):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_levels = num_levels
        
        self.td_process = nn.ModuleList([ResidualBlock(feat_channels, use_separable_conv_neck, drop_path_rate, use_dcn_neck) for _ in range(num_levels -1)])
        self.td_fusion = nn.ModuleList([WeightedFeatureFusion(2, feat_channels, epsilon) for _ in range(num_levels - 1)])

        self.bu_process = nn.ModuleList([ResidualBlock(feat_channels, use_separable_conv_neck, drop_path_rate, use_dcn_neck) for _ in range(num_levels -1)])
        self.bu_fusion = nn.ModuleList([WeightedFeatureFusion(3, feat_channels, epsilon) for _ in range(num_levels - 1)])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.downsample_ops = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.downsample_ops.append(
                ConvBNAct(feat_channels, feat_channels, kernel_size=3, stride=2)
            )

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, features):
        p_in = list(features)
        assert len(p_in) == self.num_levels

        p_td = [None] * self.num_levels
        p_td[self.num_levels-1] = p_in[self.num_levels-1]

        for i in range(self.num_levels - 2, -1, -1):
            fused_node = self.td_fusion[i]([p_in[i], self.upsample(p_td[i+1])])
            p_td[i] = self.td_process[i](fused_node)

        p_out = [None] * self.num_levels
        p_out[0] = p_td[0]

        for i in range(self.num_levels - 1):
            fused_node = self.bu_fusion[i]([ p_in[i+1], p_td[i+1], self.downsample_ops[i](p_out[i]) ])
            p_out[i+1] = self.bu_process[i](fused_node)
        return p_out


class EnhancedBiFPNNeck(nn.Module):
    def __init__(self, feat_channels=256, num_bifpn_blocks=3, num_levels=3,
                 use_separable_conv_neck=True, dropout_rate=0.0, drop_path_rate=0.0, use_dcn_neck=False):
        super().__init__()
        self.num_bifpn_blocks = num_bifpn_blocks
        print(f"Neck: FeatChannels: {feat_channels}, BiFPNBlocks: {num_bifpn_blocks}, SeparableConvs: {use_separable_conv_neck}, Dropout: {dropout_rate}, DropPath: {drop_path_rate}, DCN: {use_dcn_neck}")

        self.bifpn_blocks = nn.ModuleList([
            EnhancedBiFPNBlock(feat_channels, num_levels, use_separable_conv_neck,
                              dropout_rate=dropout_rate, drop_path_rate=drop_path_rate, use_dcn_neck=use_dcn_neck)
            for _ in range(num_bifpn_blocks)
        ])

    def forward(self, backbone_features):
        keys = [f'p{i+3}' for i in range(self.bifpn_blocks[0].num_levels)]
        features = [backbone_features[k] for k in keys]

        for block in self.bifpn_blocks:
            features = block(features)
        return features

# ==============================================================================
# PGI Module
# ==============================================================================
class PGIBlock(nn.Module):
    def __init__(self, feat_channels=256, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.feat_channels = feat_channels

        self.downsample_ops = nn.ModuleList([
            ConvBNAct(feat_channels, feat_channels, kernel_size=3, stride=2) for _ in range(num_levels - 1)
        ])
        
        self.upsample_ops = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest') for _ in range(num_levels - 1)
        ])
        
        self.fusion_convs = nn.ModuleList([
            ConvBNAct(feat_channels * num_levels, feat_channels, kernel_size=1) for _ in range(num_levels)
        ])

        self.guidance_mask_gen = nn.Sequential(
            ConvBNAct(feat_channels, feat_channels // 4, kernel_size=1),
            ConvBNAct(feat_channels // 4, feat_channels, kernel_size=3),
            nn.Conv2d(feat_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        fused_features = []
        for i in range(self.num_levels):
            # Upsample smaller features
            upsampled = []
            for j in range(i + 1, self.num_levels):
                feat = features[j]
                for k in range(j-i):
                    feat = self.upsample_ops[k](feat)
                upsampled.append(feat)

            # Downsample larger features
            downsampled = []
            for j in range(i):
                feat = features[j]
                for k in range(i-j):
                    feat = self.downsample_ops[k](feat)
                downsampled.append(feat)
            
            fused = torch.cat([features[i]] + upsampled + downsampled, dim=1)
            fused_features.append(self.fusion_convs[i](fused))
        
        guidance_mask = self.guidance_mask_gen(fused_features[0])

        guided_features = []
        for i, feat in enumerate(fused_features):
            mask = nn.functional.adaptive_avg_pool2d(guidance_mask, feat.shape[2:])
            guided_features.append(feat * mask)

        return guided_features
        
# ==============================================================================
# Detection Head (Updated for DFL)
# ==============================================================================
class DetectionHead(nn.Module):
    def __init__(self, num_classes=80, max_objects_per_pixel=1, reg_max=16,
                 in_channels=256, mid_channels=256,
                 use_separable_conv_head=True, head_depth=3, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.max_objects_per_pixel = max_objects_per_pixel
        self.reg_max = reg_max
        print(f"Head: Classes: {num_classes}, K: {max_objects_per_pixel}, RegMax: {reg_max}, InChannels: {in_channels}, MidChannels: {mid_channels}, Depth: {head_depth}, Separable: {use_separable_conv_head}, Dropout: {dropout_rate}")

        ConvBlock = SeparableConvBlock if use_separable_conv_head else ConvBNAct

        self.stem = ConvBNAct(in_channels, mid_channels, kernel_size=1)

        cls_layers = []
        for _ in range(head_depth):
            cls_layers.append(ConvBlock(mid_channels, mid_channels, kernel_size=3))
            if dropout_rate > 0: cls_layers.append(nn.Dropout2d(dropout_rate))
        self.cls_convs = nn.Sequential(*cls_layers)
        self.cls_pred = nn.Conv2d(mid_channels, num_classes * max_objects_per_pixel, 1)

        reg_layers = []
        for _ in range(head_depth):
            reg_layers.append(ConvBlock(mid_channels, mid_channels, kernel_size=3))
            if dropout_rate > 0: reg_layers.append(nn.Dropout2d(dropout_rate))
        self.reg_convs = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(mid_channels, 4 * self.reg_max * max_objects_per_pixel, 1)
        self.obj_pred = nn.Conv2d(mid_channels, max_objects_per_pixel, 1)

        self._initialize_biases()

    def _initialize_biases(self):
        prior_prob = 0.001
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if hasattr(self.cls_pred, 'bias') and self.cls_pred.bias is not None:
             nn.init.constant_(self.cls_pred.bias, bias_value)
        if hasattr(self.obj_pred, 'bias') and self.obj_pred.bias is not None:
             nn.init.constant_(self.obj_pred.bias, bias_value)

    def forward(self, x):
        B, _, H, W = x.shape
        stem_features = self.stem(x)

        cls_feat = self.cls_convs(stem_features)
        cls_pred_raw = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(stem_features)
        box_pred_raw = self.reg_pred(reg_feat)
        obj_pred_raw = self.obj_pred(reg_feat)

        box_pred_reshaped = box_pred_raw.view(B, self.max_objects_per_pixel, 4 * self.reg_max, H, W)
        box_pred_final = box_pred_reshaped.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4 * self.reg_max)

        cls_pred_reshaped = cls_pred_raw.view(B, self.max_objects_per_pixel, self.num_classes, H, W)
        cls_pred_final = cls_pred_reshaped.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, self.num_classes)

        obj_pred_reshaped = obj_pred_raw.view(B, self.max_objects_per_pixel, 1, H, W)
        obj_pred_final = obj_pred_reshaped.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 1)

        return {"boxes_dfl": box_pred_final, "classes": cls_pred_final, "objectness": obj_pred_final}

# ==============================================================================
# Complete Detector Model
# ==============================================================================
class DetectionModel(nn.Module):
    def __init__(self,
                 num_classes=80,
                 max_objects_per_pixel=1,
                 reg_max=16,
                 backbone_base_channels=78,
                 backbone_num_csp_gelan_blocks=[3,3,3,3],
                 fpn_feat_channels=128,
                 num_bifpn_blocks=4,
                 head_mid_channels=128,
                 head_depth=4,
                 use_separable_conv_backbone=True,
                 use_separable_conv_neck=True,
                 use_separable_conv_head=True,
                 dropout_rate=0.1,
                 drop_path_rate=0.07,
                 use_dcn_backbone=True,
                 use_dcn_neck=True,
                 use_pgi=True):
        super().__init__()
        self.num_classes = num_classes
        self.max_objects_per_pixel = max_objects_per_pixel
        self.reg_max = reg_max
        self.strides = [8, 16, 32]
        self.use_pgi = use_pgi

        print(f"--- Creating DetectionModel (DFL Ready) ---")
        print(f"Classes: {num_classes}, K: {max_objects_per_pixel}, RegMax: {reg_max}")
        print(f"FPN Channels: {fpn_feat_channels}, BiFPN Blocks: {num_bifpn_blocks}")
        print(f"Overall Dropout: {dropout_rate}, DropPath: {drop_path_rate}")
        print(f"PGI Enabled: {self.use_pgi}")

        self.backbone = EnhancedBackbone(
            base_channels=backbone_base_channels,
            num_csp_gelan_blocks=backbone_num_csp_gelan_blocks,
            feat_channels_proj=fpn_feat_channels,
            use_separable_conv_backbone=use_separable_conv_backbone,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            use_dcn_backbone=use_dcn_backbone
        )

        self.neck = EnhancedBiFPNNeck(
            feat_channels=fpn_feat_channels,
            num_bifpn_blocks=num_bifpn_blocks,
            num_levels=3,
            use_separable_conv_neck=use_separable_conv_neck,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            use_dcn_neck=use_dcn_neck
        )
        
        if self.use_pgi:
            self.pgi = PGIBlock(feat_channels=fpn_feat_channels, num_levels=3)

        self.heads = nn.ModuleList()
        for _ in self.strides:
            self.heads.append(DetectionHead(
                num_classes=num_classes, max_objects_per_pixel=max_objects_per_pixel,
                reg_max=reg_max,
                in_channels=fpn_feat_channels, mid_channels=head_mid_channels,
                use_separable_conv_head=use_separable_conv_head, head_depth=head_depth,
                dropout_rate=dropout_rate
            ))
        print(f"--- Model Creation Complete ---")

    def forward(self, x):
        backbone_features = self.backbone(x)
        fpn_features = self.neck(backbone_features)

        if self.use_pgi:
            fpn_features = self.pgi(fpn_features)

        outputs_per_scale = []
        for head, feature_map in zip(self.heads, fpn_features):
            outputs_per_scale.append(head(feature_map))

        combined_outputs = {}
        keys = outputs_per_scale[0].keys()
        for key in keys:
            tensors_to_concat = [out_scale[key] for out_scale in outputs_per_scale]
            combined_outputs[key] = torch.cat(tensors_to_concat, dim=1)
        
        return combined_outputs

# import torch
# import torch.nn as nn
# import math
# from collections import OrderedDict
# from torchvision.ops import DeformConv2d

# # ==============================================================================
# # Helper Modules (DropPath, ConvBNAct, SeparableConvBlock)
# # ==============================================================================
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#         random_tensor.floor_()
#         output = x.div(keep_prob) * random_tensor
#         return output

#     def extra_repr(self) -> str:
#         return f'drop_prob={round(self.drop_prob,3)}'

# class ConvBNAct(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=nn.SiLU(inplace=True)):
#         super().__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = activation if activation is not None else nn.Identity()

#     def forward(self, x):
#         return self.activation(self.bn(self.conv(x)))

# class SeparableConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.SiLU(inplace=True)):
#         super().__init__()
#         self.depthwise = ConvBNAct(in_channels, in_channels, kernel_size, stride, groups=in_channels, activation=activation)
#         self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1, stride=1, groups=1, activation=activation)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x

# class DCNv2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
#         super(DCNv2, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
        
#         self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

#     def forward(self, x):
#         offset = self.offset_conv(x)
#         mask = torch.sigmoid(self.mask_conv(x))
#         x = self.dcn(x, offset, mask)
#         return x

# # ==============================================================================
# # C2f and GELAN Blocks
# # ==============================================================================
# class Bottleneck(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = ConvBNAct(c1, c_, 1, 1)
#         self.cv2 = ConvBNAct(c_, c2, 3, 1, groups=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# class C2f(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)
#         self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1)
#         self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

# class GELANBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, num_blocks=2, use_separable_conv=True, dropout_rate=0.0):
#         super().__init__()
#         Conv = SeparableConvBlock if use_separable_conv else ConvBNAct
        
#         self.conv_in = ConvBNAct(in_channels, mid_channels, kernel_size=1)
        
#         self.blocks = nn.ModuleList()
#         for _ in range(num_blocks):
#             self.blocks.append(
#                 nn.Sequential(
#                     Conv(mid_channels, mid_channels, kernel_size=3),
#                     Conv(mid_channels, mid_channels, kernel_size=3)
#                 )
#             )

#         self.conv_out = ConvBNAct(mid_channels * (num_blocks + 1), out_channels, kernel_size=1)
#         self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

#     def forward(self, x):
#         x_in = self.conv_in(x)
#         features = [x_in]
#         current_feature = x_in
#         for block in self.blocks:
#             current_feature = block(current_feature)
#             features.append(current_feature)
        
#         out = torch.cat(features, dim=1)
#         out = self.conv_out(out)
#         out = self.dropout(out)
#         return out

# # ==============================================================================
# # CSP_GELAN_Block
# # ==============================================================================
# class CSP_GELAN_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, num_gelan_internal_blocks=2, use_separable_conv=True, dropout_rate=0.0, drop_path_rate=0.0, use_dcn=False):
#         super().__init__()
#         mid_channels = out_channels // 2

#         self.downsample = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2) 
#         self.branch1_conv = ConvBNAct(out_channels, mid_channels, kernel_size=1)
#         self.branch2_conv = ConvBNAct(out_channels, mid_channels, kernel_size=1)

#         if use_dcn:
#             self.gelan_block = nn.Sequential(
#                 GELANBlock(
#                     mid_channels, mid_channels, mid_channels,
#                     num_blocks=num_gelan_internal_blocks,
#                     use_separable_conv=use_separable_conv,
#                     dropout_rate=dropout_rate
#                 ),
#                 DCNv2(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
#             )
#         else:
#              self.gelan_block = GELANBlock(
#                 mid_channels, mid_channels, mid_channels,
#                 num_blocks=num_gelan_internal_blocks,
#                 use_separable_conv=use_separable_conv,
#                 dropout_rate=dropout_rate
#             )

#         self.fusion_conv = ConvBNAct(mid_channels * 2, out_channels, kernel_size=1)
#         self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

#     def forward(self, x):
#         x = self.downsample(x)
#         branch1_in = self.branch1_conv(x)
#         branch2 = self.branch2_conv(x) 
#         branch1_out = self.gelan_block(branch1_in) 
#         out = torch.cat([branch1_out, branch2], dim=1)
#         out = self.fusion_conv(out)
#         out = self.dropout(out) 
#         return out

# # ==============================================================================
# # SPPF
# # ==============================================================================
# class SPPF(nn.Module):
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_ = c1 // 2
#         self.cv1 = ConvBNAct(c1, c_, kernel_size=1)
#         self.cv2 = ConvBNAct(c_ * 4, c2, kernel_size=1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# # ==============================================================================
# # Backbone
# # ==============================================================================
# class EnhancedBackbone(nn.Module):
#     def __init__(self, base_channels=64, num_csp_gelan_blocks=[3,3,3,3], 
#                  feat_channels_proj=256,
#                  use_separable_conv_backbone=True,
#                  dropout_rate=0.0,
#                  drop_path_rate=0.0,
#                  use_dcn_backbone=True):
#         super().__init__()
#         print(f"Backbone: SeparableConvs: {use_separable_conv_backbone}, Dropout: {dropout_rate}, DropPath: {drop_path_rate}, ProjChannels: {feat_channels_proj}, DCN: {use_dcn_backbone}")

#         self.stem = nn.Sequential(
#              ConvBNAct(3, base_channels // 2, kernel_size=3, stride=2),
#              ConvBNAct(base_channels // 2, base_channels, kernel_size=3, stride=1),
#              ConvBNAct(base_channels, base_channels, kernel_size=3, stride=1)
#         )

#         c1_channels = base_channels
#         c2_channels = base_channels * 2
#         c3_channels = base_channels * 4
#         c4_channels = base_channels * 8
#         c5_channels = base_channels * 16

#         self.c2f = C2f(c1_channels, c1_channels, n=2, shortcut=True)
        
#         self.csp_gelan1 = CSP_GELAN_Block(c1_channels, c2_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[0], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
#         self.csp_gelan2 = CSP_GELAN_Block(c2_channels, c3_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[1], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
#         self.csp_gelan3 = CSP_GELAN_Block(c3_channels, c4_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[2], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
#         self.csp_gelan4 = CSP_GELAN_Block(c4_channels, c5_channels, num_gelan_internal_blocks=num_csp_gelan_blocks[3], use_separable_conv=use_separable_conv_backbone, dropout_rate=dropout_rate, use_dcn=use_dcn_backbone)
#         self.sppf = SPPF(c5_channels, c5_channels)

#         self.proj_p3 = ConvBNAct(c3_channels, feat_channels_proj, kernel_size=1)
#         self.proj_p4 = ConvBNAct(c4_channels, feat_channels_proj, kernel_size=1)
#         self.proj_p5 = ConvBNAct(c5_channels, feat_channels_proj, kernel_size=1)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.c2f(x)
#         x = self.csp_gelan1(x)
#         p3_feat = self.csp_gelan2(x)
#         p4_feat = self.csp_gelan3(p3_feat)
#         p5_feat = self.csp_gelan4(p4_feat)
#         p5_sppf = self.sppf(p5_feat)

#         p3 = self.proj_p3(p3_feat)
#         p4 = self.proj_p4(p4_feat)
#         p5 = self.proj_p5(p5_sppf)

#         features = OrderedDict([('p3', p3), ('p4', p4), ('p5', p5)])
#         return features

# # ==============================================================================
# # BiFPN Neck
# # ==============================================================================
# class ResidualBlock(nn.Module):
#     def __init__(self, channels, use_separable_conv=True, drop_path_rate=0.0):
#         super().__init__()
#         Conv = SeparableConvBlock if use_separable_conv else ConvBNAct
#         self.conv1 = Conv(channels, channels, kernel_size=3)
#         self.conv2 = Conv(channels, channels, kernel_size=3)
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.drop_path(out)
#         return out + residual

# class WeightedFeatureFusion(nn.Module):
#     def __init__(self, num_inputs, feat_channels, epsilon=1e-6):
#         super().__init__()
#         self.num_inputs = num_inputs
#         self.feat_channels = feat_channels
#         self.epsilon = epsilon
#         self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
#         self.weight_act = nn.ReLU(inplace=False)

#     def forward(self, inputs):
#         assert len(inputs) == self.num_inputs, f"Incorrect number of inputs {len(inputs)}, expected {self.num_inputs}"
#         normalized_weights = self.weight_act(self.weights)
#         normalized_weights = normalized_weights / (torch.sum(normalized_weights, dim=0) + self.epsilon)
#         dtype = inputs[0].dtype
#         fused_output = torch.zeros_like(inputs[0], dtype=dtype)
#         for i in range(self.num_inputs):
#             w = normalized_weights[i].view(1, -1, 1, 1) if inputs[i].ndim == 4 else normalized_weights[i]
#             fused_output += inputs[i] * w.to(dtype)
#         return fused_output

# class EnhancedBiFPNBlock(nn.Module):
#     def __init__(self, feat_channels, num_levels=3, use_separable_conv_neck=True, epsilon=1e-4, dropout_rate=0.0, drop_path_rate=0.0):
#         super().__init__()
#         self.feat_channels = feat_channels
#         self.num_levels = num_levels
#         Conv = SeparableConvBlock if use_separable_conv_neck else ConvBNAct

#         self.td_process = nn.ModuleList([ResidualBlock(feat_channels, use_separable_conv_neck, drop_path_rate) for _ in range(num_levels -1)])
#         self.td_fusion = nn.ModuleList([WeightedFeatureFusion(2, feat_channels, epsilon) for _ in range(num_levels - 1)])

#         self.bu_process = nn.ModuleList([ResidualBlock(feat_channels, use_separable_conv_neck, drop_path_rate) for _ in range(num_levels -1)])
#         self.bu_fusion = nn.ModuleList([WeightedFeatureFusion(3, feat_channels, epsilon) for _ in range(num_levels - 1)])

#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#         self.downsample_ops = nn.ModuleList()
#         for _ in range(num_levels - 1):
#             self.downsample_ops.append(
#                 ConvBNAct(feat_channels, feat_channels, kernel_size=3, stride=2)
#             )

#         self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

#     def forward(self, features):
#         p_in = list(features)
#         assert len(p_in) == self.num_levels

#         p_td = [None] * self.num_levels
#         p_td[self.num_levels-1] = p_in[self.num_levels-1]

#         for i in range(self.num_levels - 2, -1, -1):
#             fused_node = self.td_fusion[i]([p_in[i], self.upsample(p_td[i+1])])
#             p_td[i] = self.td_process[i](fused_node)

#         p_out = [None] * self.num_levels
#         p_out[0] = p_td[0]

#         for i in range(self.num_levels - 1):
#             fused_node = self.bu_fusion[i]([ p_in[i+1], p_td[i+1], self.downsample_ops[i](p_out[i]) ])
#             p_out[i+1] = self.bu_process[i](fused_node)
#         return p_out


# class EnhancedBiFPNNeck(nn.Module):
#     def __init__(self, feat_channels=256, num_bifpn_blocks=3, num_levels=3,
#                  use_separable_conv_neck=True, dropout_rate=0.0, drop_path_rate=0.0):
#         super().__init__()
#         self.num_bifpn_blocks = num_bifpn_blocks
#         print(f"Neck: FeatChannels: {feat_channels}, BiFPNBlocks: {num_bifpn_blocks}, SeparableConvs: {use_separable_conv_neck}, Dropout: {dropout_rate}, DropPath: {drop_path_rate}")

#         self.bifpn_blocks = nn.ModuleList([
#             EnhancedBiFPNBlock(feat_channels, num_levels, use_separable_conv_neck,
#                               dropout_rate=dropout_rate, drop_path_rate=drop_path_rate)
#             for _ in range(num_bifpn_blocks)
#         ])

#     def forward(self, backbone_features):
#         keys = [f'p{i+3}' for i in range(self.bifpn_blocks[0].num_levels)]
#         features = [backbone_features[k] for k in keys]

#         for block in self.bifpn_blocks:
#             features = block(features)
#         return features

# # ==============================================================================
# # Detection Head (Updated for DFL)
# # ==============================================================================
# class DetectionHead(nn.Module):
#     def __init__(self, num_classes=80, max_objects_per_pixel=1, reg_max=16,
#                  in_channels=256, mid_channels=256,
#                  use_separable_conv_head=True, head_depth=3, dropout_rate=0.1):
#         super().__init__()
#         self.num_classes = num_classes
#         self.max_objects_per_pixel = max_objects_per_pixel
#         self.reg_max = reg_max
#         print(f"Head: Classes: {num_classes}, K: {max_objects_per_pixel}, RegMax: {reg_max}, InChannels: {in_channels}, MidChannels: {mid_channels}, Depth: {head_depth}, Separable: {use_separable_conv_head}, Dropout: {dropout_rate}")

#         ConvBlock = SeparableConvBlock if use_separable_conv_head else ConvBNAct

#         self.stem = ConvBNAct(in_channels, mid_channels, kernel_size=1)

#         cls_layers = []
#         for _ in range(head_depth):
#             cls_layers.append(ConvBlock(mid_channels, mid_channels, kernel_size=3))
#             if dropout_rate > 0: cls_layers.append(nn.Dropout2d(dropout_rate))
#         self.cls_convs = nn.Sequential(*cls_layers)
#         self.cls_pred = nn.Conv2d(mid_channels, num_classes * max_objects_per_pixel, 1)

#         reg_layers = []
#         for _ in range(head_depth):
#             reg_layers.append(ConvBlock(mid_channels, mid_channels, kernel_size=3))
#             if dropout_rate > 0: reg_layers.append(nn.Dropout2d(dropout_rate))
#         self.reg_convs = nn.Sequential(*reg_layers)
#         self.reg_pred = nn.Conv2d(mid_channels, 4 * self.reg_max * max_objects_per_pixel, 1)
#         self.obj_pred = nn.Conv2d(mid_channels, max_objects_per_pixel, 1)

#         self._initialize_biases()

#     def _initialize_biases(self):
#         prior_prob = 0.001
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         if hasattr(self.cls_pred, 'bias') and self.cls_pred.bias is not None:
#              nn.init.constant_(self.cls_pred.bias, bias_value)
#         if hasattr(self.obj_pred, 'bias') and self.obj_pred.bias is not None:
#              nn.init.constant_(self.obj_pred.bias, bias_value)

#     def forward(self, x):
#         B, _, H, W = x.shape
#         stem_features = self.stem(x)

#         cls_feat = self.cls_convs(stem_features)
#         cls_pred_raw = self.cls_pred(cls_feat)

#         reg_feat = self.reg_convs(stem_features)
#         box_pred_raw = self.reg_pred(reg_feat)
#         obj_pred_raw = self.obj_pred(reg_feat)

#         box_pred_reshaped = box_pred_raw.view(B, self.max_objects_per_pixel, 4 * self.reg_max, H, W)
#         box_pred_final = box_pred_reshaped.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4 * self.reg_max)

#         cls_pred_reshaped = cls_pred_raw.view(B, self.max_objects_per_pixel, self.num_classes, H, W)
#         cls_pred_final = cls_pred_reshaped.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, self.num_classes)

#         obj_pred_reshaped = obj_pred_raw.view(B, self.max_objects_per_pixel, 1, H, W)
#         obj_pred_final = obj_pred_reshaped.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 1)

#         return {"boxes_dfl": box_pred_final, "classes": cls_pred_final, "objectness": obj_pred_final}

# # ==============================================================================
# # Complete Detector Model
# # ==============================================================================
# class DetectionModel(nn.Module):
#     def __init__(self,
#                  num_classes=80,
#                  max_objects_per_pixel=1,
#                  reg_max=16,
#                  backbone_base_channels=78,
#                  backbone_num_csp_gelan_blocks=[3,3,3,3],
#                  fpn_feat_channels=128,
#                  num_bifpn_blocks=4,
#                  head_mid_channels=128,
#                  head_depth=4,
#                  use_separable_conv_backbone=True,
#                  use_separable_conv_neck=True,
#                  use_separable_conv_head=True,
#                  dropout_rate=0.1,
#                  drop_path_rate=0.07,
#                  use_dcn_backbone=True):
#         super().__init__()
#         self.num_classes = num_classes
#         self.max_objects_per_pixel = max_objects_per_pixel
#         self.reg_max = reg_max
#         self.strides = [8, 16, 32]

#         print(f"--- Creating DetectionModel (DFL Ready) ---")
#         print(f"Classes: {num_classes}, K: {max_objects_per_pixel}, RegMax: {reg_max}")
#         print(f"FPN Channels: {fpn_feat_channels}, BiFPN Blocks: {num_bifpn_blocks}")
#         print(f"Overall Dropout: {dropout_rate}, DropPath: {drop_path_rate}")

#         self.backbone = EnhancedBackbone(
#             base_channels=backbone_base_channels,
#             num_csp_gelan_blocks=backbone_num_csp_gelan_blocks,
#             feat_channels_proj=fpn_feat_channels,
#             use_separable_conv_backbone=use_separable_conv_backbone,
#             dropout_rate=dropout_rate,
#             drop_path_rate=drop_path_rate,
#             use_dcn_backbone=use_dcn_backbone
#         )

#         self.neck = EnhancedBiFPNNeck(
#             feat_channels=fpn_feat_channels,
#             num_bifpn_blocks=num_bifpn_blocks,
#             num_levels=3,
#             use_separable_conv_neck=use_separable_conv_neck,
#             dropout_rate=dropout_rate,
#             drop_path_rate=drop_path_rate
#         )

#         self.heads = nn.ModuleList()
#         for _ in self.strides:
#             self.heads.append(DetectionHead(
#                 num_classes=num_classes, max_objects_per_pixel=max_objects_per_pixel,
#                 reg_max=reg_max,
#                 in_channels=fpn_feat_channels, mid_channels=head_mid_channels,
#                 use_separable_conv_head=use_separable_conv_head, head_depth=head_depth,
#                 dropout_rate=dropout_rate
#             ))
#         print(f"--- Model Creation Complete ---")

#     def forward(self, x):
#         backbone_features = self.backbone(x)
#         fpn_features = self.neck(backbone_features)

#         outputs_per_scale = []
#         for head, feature_map in zip(self.heads, fpn_features):
#             outputs_per_scale.append(head(feature_map))

#         combined_outputs = {}
#         keys = outputs_per_scale[0].keys()
#         for key in keys:
#             tensors_to_concat = [out_scale[key] for out_scale in outputs_per_scale]
#             combined_outputs[key] = torch.cat(tensors_to_concat, dim=1)
        
#         return combined_outputs
