"""PyTorch port of `mod_resnet` from argosfeddeep/models.py (TensorFlow/Keras).

ResNet-style encoder with a PSP-style multi-scale-fusion decoder (resize + 1x1
conv + concat), rather than a symmetric U-Net decoder. Ported 1:1 from the
Keras functional model, including its asymmetries (see ConvResBlock).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# The original only sets kernel_initializer='he_normal' on IdentityBlock's and
# ConvResBlock's conv1/conv2 — every other conv (stem, ConvResBlock's shortcut,
# all decoder convs, the output conv) is left at Keras Conv2D's own default,
# which is glorot_uniform (Xavier uniform), not PyTorch's own Conv2d default.
def _init_conv_he(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _init_conv_default(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def conv2d_he(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return _init_conv_he(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))


def conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return _init_conv_default(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))


class IdentityBlock(nn.Module):
    """Two 3x3 conv+BN layers, added back to the (shape-preserving) input, then ReLU.

    conv1/conv2 are he_normal-initialized and L2-regularized (see
    ModResNet.l2_regularization_loss), matching the original exactly.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = conv2d_he(channels, channels, kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv2d_he(channels, channels, kernel_size, stride=1, padding=pad)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)


class ConvResBlock(nn.Module):
    """Strided residual block that also projects/downsamples the shortcut.

    Note (kept faithful to the original): the main path's second conv (y_2)
    has no BatchNorm before the residual add, unlike the shortcut path (y_3).

    conv1/conv2 are he_normal-initialized and L2-regularized, same as
    IdentityBlock. shortcut_conv is neither — it's left at Keras' default
    (glorot_uniform) with no regularization, matching the original exactly.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = conv2d_he(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv2d_he(out_channels, out_channels, kernel_size, stride=1, padding=pad)

        self.shortcut_conv = conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = self.conv2(y)

        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        return self.relu(y + shortcut)


class DecoderBranch(nn.Module):
    """Resize an encoder feature map to a target spatial size, then 1x1 conv + BN.

    The original resizes via chained 2x bilinear hops (e.g. 16x16 -> 32x32 ->
    64x64 -> 128x128), not a single direct resize to the final size — chained
    bilinear upsampling is not numerically the same as one bilinear resize to
    the target size, so the hop count is preserved here rather than collapsed.
    """

    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.conv = conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, size):
        target_h, target_w = size
        h, w = x.shape[-2], x.shape[-1]
        while (h, w) != (target_h, target_w):
            h, w = min(h * 2, target_h), min(w * 2, target_w)
            # align_corners=False matches TF2's tf.image.resize default
            # (half-pixel-center sampling) for method='bilinear'.
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.bn(self.conv(x))


class ModResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()

        # --- Encoder (ResNet18-ish stem + 4 stages) ---
        self.stem_conv = conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage1 = nn.Sequential(IdentityBlock(64), IdentityBlock(64))

        self.stage2_down = ConvResBlock(64, 128, stride=2)
        self.stage2 = IdentityBlock(128)

        self.stage3_down = ConvResBlock(128, 256, stride=2)
        self.stage3 = IdentityBlock(256)

        self.stage4_down = ConvResBlock(256, 512, stride=2)
        self.stage4 = IdentityBlock(512)

        # --- Decoder: multi-scale fusion (PSP-like) ---
        self.branch_e3 = DecoderBranch(128, 128)
        self.branch_e4 = DecoderBranch(256, 128)
        self.branch_e5 = DecoderBranch(512, 128)

        self.fuse_conv1 = conv2d(128 * 3, 64, kernel_size=1)
        self.fuse_bn1 = nn.BatchNorm2d(64)
        self.fuse_conv2 = conv2d(64, 64, kernel_size=3, padding=1)
        self.fuse_bn2 = nn.BatchNorm2d(64)

        self.skip_conv = conv2d(64, 64, kernel_size=3, padding=1)
        self.skip_bn = nn.BatchNorm2d(64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_bn = nn.BatchNorm2d(64)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_bn = nn.BatchNorm2d(64)

        self.out_conv = conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # No activation here on purpose: the original is Conv->BN->MaxPool,
        # with no ReLU before the pool.
        e1 = self.stem_pool(self.stem_bn(self.stem_conv(x)))

        e2 = self.stage1(e1)

        e3 = self.stage2(self.stage2_down(e2))
        e4 = self.stage3(self.stage3_down(e3))
        e5 = self.stage4(self.stage4_down(e4))

        # Fuse encoder stages at e2's spatial resolution (input // 4).
        target_size = e2.shape[-2:]
        up1 = self.branch_e3(e3, target_size)
        up2 = self.branch_e4(e4, target_size)
        up3 = self.branch_e5(e5, target_size)

        d1 = torch.cat([up3, up2, up1], dim=1)
        d1 = self.fuse_bn1(self.fuse_conv1(d1))
        d1 = self.fuse_bn2(self.fuse_conv2(d1))

        d2 = e2 + d1
        d2 = self.skip_bn(self.skip_conv(d2))

        # No activation here on purpose: the original is ConvTranspose->BN
        # twice with no ReLU in between or after, before the final 1x1 conv.
        d3 = self.up1_bn(self.up1(d2))
        d3 = self.up2_bn(self.up2(d3))

        logits = self.out_conv(d3)
        return F.softmax(logits, dim=1)

    def l2_regularization_loss(self, l2_lambda: float) -> torch.Tensor:
        """Matches the original's selective kernel_regularizer=l2(l2_lambda):
        only IdentityBlock/ConvResBlock's conv1+conv2 kernels are penalized —
        not the stem, decoder convs, output conv, or ConvResBlock's shortcut
        conv. Add this to the task loss before backward(), same as
        `total_loss = regularization_loss + loss_value` in the original's
        train_on_batch/validate_on_batch.
        """
        total = torch.zeros((), device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, (IdentityBlock, ConvResBlock)):
                total = total + torch.sum(m.conv1.weight**2) + torch.sum(m.conv2.weight**2)
        return l2_lambda * total


if __name__ == "__main__":
    model = ModResNet(in_channels=3, num_classes=2)
    dummy = torch.randn(1, 3, 512, 512)
    out = model(dummy)
    print(out.shape)  # expected: torch.Size([1, 2, 512, 512])
