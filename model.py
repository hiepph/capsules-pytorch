import torch
import torch.nn as nn
from torch.autograd import Variable


class CapsulesLayer(nn.Module):
    def __init__(self, in_unit, in_channels, n_unit, unit_size,
                 use_routing, n_routing,
                 use_cuda):
        super(CapsulesLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channels = in_channels
        self.n_unit = n_unit
        self.use_routing = use_routing
        self.n_routing = n_routing
        self.use_cuda = use_cuda

        if self.use_routing:
            pass
        else:
            """
            No routing between Conv1 and PrimaryCapsules

            Paper: PrimaryCapsules is a convolutional  layer with 32 channels of
            convolutional 8D capsules (i.e. each primary capsule contains
            8 convolutional units with a 9 × 9 kernel and a stride of 2)
            """
            self.conv_units = nn.ModuleList([
                nn.Conv2d(in_channels=self.in_channels,
                          out_chanels=32,
                          kernel_size=9,
                          stride=2)
                for _ in range(self.n_unit)
            ])

    def forward(self, x):
        if self.use_routing:
            pass
        else:
            self.no_routing(x)

    def no_routing(self, x):
        """Get output for each unit

        Args:
        x: shape (batch_size, C, H, W)

        Returns: vector output of capsule j
        """
        # Create 8 convolutional units
        units = [self.conv_units[i](x) for i, _ in enumerate(self.conv_units)]

        # Stack all 8 unit outputs
        # output shape: [128, 8, 32, 6, 6]
        units = torch.stack(units, dim=1)

        # Flatten the 32 of 6x6 grid into 1152
        # shape: [128, 8, 1152]
        units = units.view(x.size(0), self.n_unit, -1)

        # Add non-linearity
        # returns squashed output of shape: [128, 8, 1152]
        # dim=2 is the third dim 1152D
        return self.squash(units, dim=2)

    def squash(self, sj, dim=2):
        """Make short vectors get shunk to 0
        and long vectors get shrunk to near 1
        """
        # ||sj||^2
        sj_mag_sq = torch.sum(sj**2, dim, keepdim=True)
        # ||sj||
        sj_mag = torch.sqrt(torch.sum)

        return (sj_mag_sq / (1. + sj_mag_sq)) * (sj / sj_mag_sq)


class CapsulesNet(nn.Module):
    """Capsules Net with 3 routing & reconstruction loss
    """
    def __init__(self, n_conv_in_channel, n_conv_out_channel,
                 n_primary_unit, primary_unit_size,
                 n_classes, output_unit_size,
                 n_routing, regularization_scale,
                 input_weight, input_height,
                 use_cuda):
        super(CapsulesNet, self).__init__()

        self.use_cuda = use_cuda

        # Image setting
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = n_conv_in_channel

        # known as lambda reconstruction
        # use sum of squared errors (SSE)
        self.regularization_scale = regularization_scale

        # Layer 1: Conventional Conv2D layer
        self.conv1 = nn.Sequential(
            # input shape (batch_size, C, H, W): [128, 1, 28, 28]
            # (CONV) -> [128, 256, 20, 20]
            nn.Conv2d(in_channels=n_conv_in_channel,
                      out_chanels=n_conv_out_channel,
                      kernel_size=9, stride=1),
            # (ReLU) -> [128, 256, 20, 20]
            nn.ReLU(inplace=True)
        )

        # Primary Caps
        # Layer 2: Conv2D layer with squash activation
        self.primary = CapsulesLayer(in_unit=0,
                                     in_channels=n_conv_out_channel,
                                     n_unit=n_primary_unit,
                                     unit_size=primary_unit_size,
                                     use_routing=False,
                                     n_routing=n_routing,
                                     use_cuda=use_cuda)
