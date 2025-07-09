import functools
import torch
import torch.nn.functional as F
from torch import nn

def create_sequential_layers(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class PT_ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int=64, growth_channels: int=32, bias: bool=True):
        super(PT_ResidualDenseBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv2d_2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv2d_3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv2d_4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1, bias=bias)
        self.conv2d_5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1, bias=bias)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.leakyReLU(self.conv2d_1(x))
        x2 = self.leakyReLU(self.conv2d_2(torch.cat((x, x1), 1)))
        x3 = self.leakyReLU(self.conv2d_3(torch.cat((x, x1, x2), 1)))
        x4 = self.leakyReLU(self.conv2d_4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv2d_5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class PT_RRDB(nn.Module):
    def __init__(self, channels, growth_channels=32):
        super(PT_RRDB, self).__init__()
        self.ResidualDenseBlock1 = PT_ResidualDenseBlock(channels, growth_channels)
        self.ResidualDenseBlock2 = PT_ResidualDenseBlock(channels, growth_channels)
        self.ResidualDenseBlock3 = PT_ResidualDenseBlock(channels, growth_channels)

    def forward(self, x):
        out = self.ResidualDenseBlock1(x)
        out = self.ResidualDenseBlock2(out)
        out = self.ResidualDenseBlock3(out)
        return out * 0.2 + x

class PT_RRDB_Net_16x(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int, num_rrdb: int, growth_channels: int=32):
        super(PT_RRDB_Net_16x, self).__init__()

        # use when added to preloaded model and stacked
        self.skiplast = False

        pt_rrdb_block = functools.partial(PT_RRDB, channels=channels, growth_channels=growth_channels)

        self.main_initial_conv2d = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=True)

        # 4x upsampling section
        self.Sequential_RRDB_block_1 = create_sequential_layers(pt_rrdb_block, num_rrdb)
        self.trunk_Conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.end_upsampling_conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        # 4x upsampling section
        self.Sequential_RRDB_trunk_2 = create_sequential_layers(pt_rrdb_block, num_rrdb)
        self.trunk_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_3 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_4 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.end_upsampling_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        self.end_conv2d = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        forward_result = self.main_initial_conv2d(x)

        # 4x upsampling section 1
        block1 = self.trunk_Conv2d_1(self.Sequential_RRDB_block_1(forward_result))
        forward_result = forward_result + block1

        forward_result = self.leakyReLU(self.upsample_conv2d_1(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.leakyReLU(self.upsample_conv2d_2(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.end_upsampling_conv2d_1(forward_result)

        # 4x upsampling section 2
        block2 = self.trunk_conv2d_2(self.Sequential_RRDB_trunk_2(forward_result))
        forward_result = forward_result + block2

        forward_result = self.leakyReLU(self.upsample_conv2d_3(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.leakyReLU(self.upsample_conv2d_4(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.end_upsampling_conv2d_2(forward_result)

        if not self.skiplast:
            forward_result = self.end_conv2d(self.leakyReLU(forward_result))

        return forward_result

    def skip_last(self):
        self.skiplast = True

class PT_RRDB_Net_2x(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int, num_rrdb: int, growth_channels: int=32):
        super(PT_RRDB_Net_2x, self).__init__()

        # use when added to preloaded model and stacked
        self.skiplast = False

        pt_rrdb_block = functools.partial(PT_RRDB, channels=channels, growth_channels=growth_channels)

        self.main_initial_conv2d = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=True)

        # 2x upsampling section
        self.Sequential_RRDB_block_1 = create_sequential_layers(pt_rrdb_block, num_rrdb)
        self.trunk_Conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.end_upsampling_conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        self.end_conv2d = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        forward_result = self.main_initial_conv2d(x)

        block1 = self.trunk_Conv2d_1(self.Sequential_RRDB_block_1(forward_result))
        forward_result = forward_result + block1

        forward_result = self.leakyReLU(self.upsample_conv2d_1(F.interpolate(forward_result, scale_factor=2, mode='nearest-exact')))

        forward_result = self.upsample_conv2d_2(forward_result)
        forward_result = self.leakyReLU(forward_result)
        forward_result = self.end_upsampling_conv2d_1(forward_result)

        if not self.skiplast:
            forward_result = self.end_conv2d(self.leakyReLU(forward_result))

        return forward_result

    def skip_last(self):
        self.skiplast = True

class PT_RRDB_Net_8x(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int, num_rrdb: int, growth_channels: int=32):
        super(PT_RRDB_Net_8x, self).__init__()

        pt_rrdb_block = functools.partial(PT_RRDB, channels=channels, growth_channels=growth_channels)

        self.main_initial_conv2d = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=True)

        # 4x upsampling section
        self.Sequential_RRDB_block_1 = create_sequential_layers(pt_rrdb_block, num_rrdb)
        self.trunk_Conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.end_upsampling_conv2d_1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        # 2x upsampling section
        self.Sequential_RRDB_trunk_2 = create_sequential_layers(pt_rrdb_block, num_rrdb)
        self.trunk_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.upsample_conv2d_3 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.end_upsampling_conv2d_2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        self.end_conv2d = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        forward_result = self.main_initial_conv2d(x)

        # 4x upsampling section 1
        block1 = self.trunk_Conv2d_1(self.Sequential_RRDB_block_1(forward_result))
        forward_result = forward_result + block1

        forward_result = self.leakyReLU(self.upsample_conv2d_1(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.leakyReLU(self.upsample_conv2d_2(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.end_upsampling_conv2d_1(forward_result)

        # 2x upsampling section 2
        block2 = self.trunk_conv2d_2(self.Sequential_RRDB_trunk_2(forward_result))
        forward_result = forward_result + block2

        forward_result = self.leakyReLU(self.upsample_conv2d_3(F.interpolate(forward_result, scale_factor=2, mode='nearest')))
        forward_result = self.end_upsampling_conv2d_2(forward_result)

        forward_result = self.end_conv2d(self.leakyReLU(forward_result))

        return forward_result




