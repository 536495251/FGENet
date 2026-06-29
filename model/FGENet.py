import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from einops import rearrange
import typing as t

class SCSA(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class SemiLearnableDWConv(nn.Module):
    def __init__(self, channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.padding = padding
        self.dilation = dilation
        self.groups = channels

        self.weight = nn.Parameter(
            torch.zeros(channels, 1, kernel_size, kernel_size)
        )

        self._init_high_pass(kernel_size)

    def _init_high_pass(self, k):
        center = k // 2
        kernel = -torch.ones((1, 1, k, k))
        kernel[0, 0, center, center] = k * k - 1
        kernel /= (k * k - 1)
        self.weight.data = kernel.repeat(self.weight.shape[0], 1, 1, 1)

    def forward(self, x):
        w = self.weight
        w = w - w.mean(dim=[2, 3], keepdim=True)
        return F.conv2d(
            x, w,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
class MDWC(nn.Module):
    def __init__(self, inplanes, outplanes, one, two, three, scales=4, conv_type='semi'):
        super(MDWC, self).__init__()
        if outplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        self.scales = scales
        self.relu = nn.ReLU(inplace=True)
        self.spx = outplanes // scales

        # Choose conv class
        if conv_type == 'semi':
            ConvClass = SemiLearnableDWConv

        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0),
            nn.BatchNorm2d(outplanes)
        )
        self.conv1 = nn.Sequential(
            ConvClass(self.spx, one, one // 2),
            nn.BatchNorm2d(self.spx)
        )
        self.conv2 = nn.Sequential(
            ConvClass(self.spx, two, padding=2, dilation=2),
            nn.BatchNorm2d(self.spx)
        )
        self.conv3 = ConvClass(self.spx, three, padding=1)
        self.conv4 = ConvClass(self.spx, three, padding=2, dilation=2)
        self.conv5 = nn.BatchNorm2d(self.spx)
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.scsa = SCSA(
            dim=outplanes,
            head_num=4,
            window_size=7,
            down_sample_mode='avg_pool',
            gate_layer='sigmoid'
        )
    def forward(self, x):
        x = self.inconv(x)
        inputt = x

        xs = torch.chunk(x, self.scales, dim=1)
        ys = []

        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))

        temp = xs[3] + ys[2]
        temp = self.conv3(temp) + self.conv4(temp)
        temp = self.conv5(temp)
        ys.append(self.relu(temp))

        y = torch.cat(ys, dim=1)
        y = self.outconv(y)
        y = self.scsa(y)
        return self.relu(y + inputt)

class FFT(nn.Module):
    def __init__(self, energy_ratio=0.2, min_cutoff=3, max_search=None):
        super().__init__()
        self.energy_ratio = energy_ratio
        self.min_cutoff = min_cutoff
        self.max_search = max_search

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        x = x.float()

        f = torch.fft.fft2(x, dim=(-2, -1))
        fshift = torch.fft.fftshift(f, dim=(-2, -1))

        crow, ccol = H // 2, W // 2
        y, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        distance = torch.sqrt((y - crow) ** 2 + (x_grid - ccol) ** 2)  # (H,W)
        distance = distance.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)  # (B,C,H,W)

        magnitude = torch.abs(fshift)
        total_energy = torch.sum(magnitude ** 2, dim=(-2, -1), keepdim=True)  # (B,C,1,1)
        target_low_energy = total_energy * self.energy_ratio  # (B,C,1,1)

        max_radius = self.max_search if self.max_search is not None else np.sqrt(crow ** 2 + ccol ** 2)
        radius_list = torch.arange(self.min_cutoff, int(max_radius) + 1, device=device)  # (R,)

        R = radius_list.shape[0]
        dist_expand = distance.unsqueeze(0).expand(R, B, C, H, W)  # (R,B,C,H,W)
        radius_expand = radius_list[:, None, None, None, None].expand(R, B, C, H, W)  # (R,B,C,H,W)

        low_mask = (dist_expand <= radius_expand).float()
        mag_expand = magnitude.unsqueeze(0).expand(R, B, C, H, W)
        low_energy = torch.sum(low_mask * (mag_expand ** 2), dim=(-2, -1))  # (R,B,C)

        target_energy = target_low_energy.squeeze(-1).squeeze(-1)  # (B,C)
        mask_valid = low_energy >= target_energy[None, :, :]  # (R,B,C)
        cutoff_idx = mask_valid.float().argmax(dim=0)  # (B,C)
        cutoff_idx[~mask_valid.any(dim=0)] = self.min_cutoff
        cutoff_radius = cutoff_idx.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        highpass_mask = (distance >= cutoff_radius).float()  # (B,C,H,W)
        fshift = fshift * highpass_mask
        f_ishift = torch.fft.ifftshift(fshift, dim=(-2, -1))
        high_freq = torch.fft.ifft2(f_ishift, dim=(-2, -1)).real
        return high_freq

class MFENet(nn.Module):
    def __init__(self, input_channels=1, block=ResNet, Train=False,
                 energy_ratios=None, conv_type='semi'):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.Train = Train
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.py_init = self._make_layer1(input_channels, 1, block)
        self.encoder_0 = self._make_layer2(param_channels[0], param_channels[0], block, conv_type=conv_type)
        self.encoder_1 = self._make_layer2(param_channels[0], param_channels[1], block, param_blocks[0], conv_type=conv_type)
        self.encoder_2 = self._make_layer2(param_channels[1], param_channels[2], block, param_blocks[1], conv_type=conv_type)
        self.encoder_3 = self._make_layer2(param_channels[2], param_channels[3], block, param_blocks[2], conv_type=conv_type)

        self.middle_layer = self._make_layer2(param_channels[3], param_channels[4], block, param_blocks[3], conv_type=conv_type)

        self.decoder_3 = self._make_layer1(param_channels[3] + param_channels[4], param_channels[3], block,
                                           param_blocks[2])
        self.decoder_2 = self._make_layer1(param_channels[2] + param_channels[3], param_channels[2], block,
                                           param_blocks[1])
        self.decoder_1 = self._make_layer1(param_channels[1] + param_channels[2], param_channels[1], block,
                                           param_blocks[0])
        self.decoder_0 = self._make_layer1(param_channels[0] + param_channels[1], param_channels[0], block)

        if energy_ratios is None:
            energy_ratios = [0.8, 0.6, 0.4, 0.1]
        self.py3 = FFT(energy_ratio=energy_ratios[0])
        self.py2 = FFT(energy_ratio=energy_ratios[1])
        self.py1 = FFT(energy_ratio=energy_ratios[2])
        self.py0 = FFT(energy_ratio=energy_ratios[3])
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)
    def _make_layer1(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)
    def _make_layer2(self, in_channels, out_channels, block, block_num=1, conv_type='semi'):
        layer = []
        layer.append(MDWC(in_channels, out_channels, 3, 3, 3, conv_type=conv_type))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)
    def forward(self, x):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        x_py_init = self.py_init(x)
        x_py_v3 = x_py_init * self.sigmoid(self.up_8(mask3)) + x_py_init
        x_py_v3 = self.py3(x_py_v3)
        x_py_v2 = x_py_v3 * self.sigmoid(self.up_4(mask2)) + x_py_v3
        x_py_v2 = self.py2(x_py_v2)
        x_py_v1 = x_py_v2 * self.sigmoid(self.up(mask1)) + x_py_v2
        x_py_v1 = self.py1(x_py_v1)
        x_py_v0 = x_py_v1 * self.sigmoid(mask0) + x_py_v1
        x_py_v0 = self.sigmoid(self.py0(x_py_v0))
        output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
        output = output * x_py_v0 + output
        mask1 = F.interpolate(mask1, scale_factor=2, mode='bilinear', align_corners=True)
        mask2 = F.interpolate(mask2, scale_factor=4, mode='bilinear', align_corners=True)
        mask3 = F.interpolate(mask3, scale_factor=8, mode='bilinear', align_corners=True)
        if self.Train:
            return [torch.sigmoid(output), torch.sigmoid(mask0), torch.sigmoid(mask1), torch.sigmoid(mask2),
                    torch.sigmoid(mask3)]
        else:
            return torch.sigmoid(output)


if __name__ == '__main__':
    model = MFENet(Train=True)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    flops, params = profile(model, (x,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

    if len(output) > 1:
        print("Output shape:", output[0].shape, output[1].shape, output[2].shape, output[3].shape)
    else:
        print("Output shape:", output.shape)
