import torch
import torch.nn as nn
import torch.nn.functional as F
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
            """Weight matrix used by routing algorithm
            https://cdn-images-1.medium.com/max/1000/1*GbmQ2X9NQoGuJ1M-EOD67g.png

            shape: [1 x primary_unit_size x n_classes x output_unit_size x n_primary_unit]
            or specifically: [1 x 1152 x 10 x 16 x 8]
            """
            self.weight = nn.Parameter(torch.randn(1, in_channels, n_unit,
                                                   unit_size, in_unit))
        else:
            """ No routing between Conv1 and PrimaryCapsules

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
            return self.routing(x)
        else:
            return self.no_routing(x)

    def routing(self, x):
        """
        Args:
        x: Tensor of shape [128, 8, 1152]

        Returns:
        Vector output of capsule j
        """
        batch_size = x.size(0)

        # swap dim1 and dim 2,
        # -> shape: [128, 1152, 8]
        x = x.tranpose(1, 2)

        # Stack and add a dimension to a tensor
        # -> [128, 1152, 10, 8]
        # unsqueeze -> [128, 1152, 10, 8, 1]
        x = torch.stack([x] * self.n_unit, dim=2).unsqueeze(4)

        # Convert single weight to batch weight
        # [1 x 1152 x 10 x 16 x 8] -> [128 x 1152 x 10 x 16 x 8]
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)

        # `u_hat` is "prediction vectors" from lower capsules
        # Transform inputs by weight matrix
        # Matrix product of 2 tensors with shape: [128,1152,10,16,8] x [128,1152,10,8,1]
        # u_hat shape: [128,1152,10,16,1]
        u_hat = torch.matmul(batch_weight, x)

        """Implementation of (Procedure 1: Routing algorithm) in Paper
        """
        # (2) At start of training the value of b_ij is initialized at zero
        # b_ji shape: [1,1152,10,1]
        b_ij = Variable(torch.zeros(1, self.in_channels, # primary_unit_size (32 * 6 * 6 = 1152
                                    self.n_unit, # = n_classes = 10
                                    1))
        if self.use_cuda:
            b_ij = b_ij.cuda()

        # number of iterations = number of routing
        for iteration in range(self.n_routing):
            # (4) Calculate routing or coupling coefficients (c_ij)
            # c_ij shape: [1,1152,10,1]
            c_ij = F.softmax(b_ij, dim=2)
            # stack and convert c_ij shape from [128,1152,10,1] to [128,1152,10,1,1]
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # (5) s_j is total input to a capsule, is a weighted sum over all "prediction vectors"
            # u_hat is weighted inputs, prediction u_hat(j|i) made by capsule i
            # c_ij * u_hat shape: [128,1152,10,16,1]
            # s_j output shape: [128,1,10,16,1] (dim 1: 1152D -> 1D)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (6) squash the vector output of capsule j
            # v_j shape: [batch_size, weighted_sum of PrimaryCaps output,
            #             num_classes, ouput_unit_size from u_hat, 1]
            #         == [128, 1, 10, 16, 1]
            # So, length of output vector of a capsule is 16 (dim 3)
            v_j = self.squash(s_j, dim=3)

            # in_chanels = 1152
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # The agreement
            # Tranpose u_hat with shape [128,1152,10,16,1] to [128,1152,10,,16]
            # so we can do matrix product u_hat and v_i1
            # u_vj1 shape: [1,1152,10,1]
            u_vj1 = torch.matmul(u_hat.tranpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij) by adding the agreement to initial logit
            b_ij = b_ij + u_vj1

        # shape: [128,10,16,1]
        return v_j.squeeze(1)

    def no_routing(self, x):
        """Get output for each unit

        Args:
        x: Shape (batch_size, C, H, W)

        Returns:
        Vector output of capsule j
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

        # Digit Caps
        # Last layer: routing between Primary Capsules and DigitsCaps
        self.digits = CapsulesLayer(in_unit=n_primary_unit,
                                    in_channels=primary_unit_size,
                                    n_unit=n_classes,
                                    unit_size=output_unit_size,
                                    use_routing=True,
                                    n_routing=n_routing,
                                    use_cuda=use_cuda)
