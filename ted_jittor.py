import jittor
import jittor.nn as nn

from utils.AF.fsmish_jittor import smish as Fsmish
from utils.AF.xsmish_jittor import Smish

jittor.flags.use_cuda = 1


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        nn.init.xavier_gauss_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zero_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        nn.init.xavier_gauss_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zero_(m.bias)


class CoFusion(nn.Module):
    # from LDC

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1)  # before 64
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)  # before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32)  # before 64

    def execute(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = nn.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class CoFusion2(nn.Module):
    # TEDv14-3
    def __init__(self, in_ch, out_ch):
        super(CoFusion2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1)  # before 64
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)  # before 64  instead of 32
        self.smish = Smish()  # nn.ReLU(inplace=True)

    def execute(self, x):
        attn = self.conv1(self.smish(x))
        attn = self.conv3(self.smish(attn))  # before , )dim=1)

        return ((x * attn).sum(1)).unsqueeze(1)


class DoubleFusion(nn.Module):
    # TED fusion before the final edge map prediction
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3,
                               stride=1, padding=1)  # before 64

        self.conv2 = nn.Conv2d(24, 24 * 1, kernel_size=3,
                               stride=1, padding=1)  # before 64  instead of 32

        self.AF = Smish()  # Smish()

    def execute(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.conv1(self.AF(x))  # #TEED best res TEDv14 [8, 32, 352, 352]

        attn2 = self.conv2(self.AF(attn))  # #TEED best res TEDv14[8, 3, 352, 352]

        return Fsmish(((attn2 + attn).sum(1)).unsqueeze(1))  # TED best res


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('smish1', Smish()),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))

    def execute(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).execute(Fsmish(x1))  # F.relu()

        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def execute(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        # self.use_bn = use_bs
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = Smish()

    def execute(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        else:
            return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = Smish()  # nn.ReLU(inplace=True)

    def execute(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class TED(nn.Module):
    """
    Definition of Tiny and Efficient Edge Detector model
    """

    def __init__(self):
        super(TED, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2, )
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)  # [32,48,100,100] before (2, 32, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # skip1 connection, see fig. 2
        self.side_1 = SingleConvBlock(16, 32, 2)

        # skip2 connection, see fig. 2
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)  # before (32, 64, 1)

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        self.block_cat = DoubleFusion(3, 3)  # TEED: DoubleFusion

        self.apply(weight_init)

    def execute(self, x, single_test=False):
        assert x.ndim == 4, x.shape
        # suppose the image is 3x352x352

        # Block 1
        block_1 = self.block_1(x)  # [B,16,176,176]
        block_1_side = self.side_1(block_1)  # [B,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1)  # [B,32,176,176]
        block_2_down = self.maxpool(block_2)  # [B,32,88,88]
        block_2_add = block_2_down + block_1_side  # [B,32,88,88]

        block_3_pre_dense = self.pre_dense_3(block_2_down)  # [B,48,88,88] block 3 L connection
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])  # [B,48,88,88]

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        block_cat = jittor.concat([out_1, out_2, out_3], dim=1)  # [B,3,352,352]
        block_cat = self.block_cat(block_cat)  # [B,1,352,352]

        return [out_1, out_2, out_3, block_cat]
