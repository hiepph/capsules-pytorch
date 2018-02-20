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
                          out_channels=32,
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
        x = x.transpose(1, 2)

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
            # Transpose u_hat with shape [128,1152,10,16,1] to [128,1152,10,,16]
            # so we can do matrix product u_hat and v_i1
            # u_vj1 shape: [1,1152,10,1]
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

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
        units = [conv_unit(x) for conv_unit in self.conv_units]

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
        sj_mag = torch.sqrt(sj_mag_sq)

        return (sj_mag_sq / (1. + sj_mag_sq)) * (sj / sj_mag_sq)


class Decoder(nn.Module):
    """ The decoder consists of 3 fully-connected layers.
    For each [10,16] output, mask out the incorrect predictions,
    and send [16,] vector to the decoder network to reconstruct
    a [784,] size image.

    This network is used both in training and testing.
    """
    def __init__(self, n_classes, output_unit_size,
                 input_height, input_width,
                 n_conv_in_channel,
                 use_cuda):
        super(Decoder, self).__init__()

        self.use_cuda = use_cuda

        # 3 FC layers
        self.fc1 = nn.Linear(n_classes * output_unit_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, input_height * input_width * n_conv_in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target):
        """Send output of `DigitCaps` layer (shape [batch_size,10,16]) to Decoder network,
        and reconstruct a [batch_size, fc3's output size] tensor representing batch images

        Args:
            x: [batch_size,10,16] Output of digit capsule
            target: [batch_size,10] One-hot MNIST dataset label

        Returns:
            reconstruction: [batch_size, fc3's output size] Tensor of reconstructed image
        """
        # Mask with y
        # masked cap shape: [batch_size,10,16,1]
        masked_caps = self.mask(x, self.use_cuda)

        # Reconstruct image with 3 FC layers
        # vector_j shape: [batch_size,16*10]
        vector_j = masked_caps.view(x.size(0), -1)

        # Forward
        fc1_out = self.relu(self.fc1(vector_j))
        fc2_out = self.relu(self.fc2(fc1_out))
        reconstruction = self.sigmoid(self.fc3(fc2_out))

        return reconstruction

    def mask(self, out_digit_caps, use_cuda):
        """Mask out all but the activity vector of the correct digit capsule
        a) during training, mask all but the capsule (1,16) vector which match ground truth
        b) during testing, mask all but longest capsule ((1,16) vector)

        Args:
            out_digit_caps: [batch_size,10,16] Tensor output of DigitCaps layer

        Returns:
            masked: [batch_size,10,16,1] Masked capsule tensors
        """
        # get capsule output length, ||v_c||
        v_length = torch.sqrt((out_digit_caps**2).sum(dim=2))

        # pick out the index of longest capsule output, v_length by
        # masking the tensor by the max value in dim=1
        _, max_index = v_length.max(dim=1)
        max_index = max_index.data

        # Masking with y
        # In all batches, get the most active capsule
        batch_size = out_digit_caps.size(0)
        masked_v = [None] * batch_size
        for i in range(batch_size):
            sample = out_digit_caps[i]

            # Mask out other capsules
            v = Variable(torch.zeros(sample.size()))
            if use_cuda:
                v = v.cuda()

            # Get maximum capsule index
            max_caps_index = max_index[i]
            v[max_caps_index] = sample[max_caps_index]
            masked_v[i] = v

        # Concatenate sequence of masked capsules tensors along the batch dimension
        masked = torch.stack(masked_v, dim=0)

        return masked


class CapsulesNet(nn.Module):
    def __init__(self, n_conv_in_channel, n_conv_out_channel,
                 n_primary_unit, primary_unit_size,
                 n_classes, output_unit_size,
                 n_routing, regularization_scale,
                 input_width, input_height,
                 use_cuda):
        super(CapsulesNet, self).__init__()

        self.use_cuda = use_cuda

        # Image setting
        self.image_width = input_width
        self.image_height = input_height
        self.image_channel = n_conv_in_channel

        # known as lambda reconstruction
        # use sum of squared errors (SSE)
        self.regularization_scale = regularization_scale

        # Layer 1: Conventional Conv2D layer
        self.conv1 = nn.Sequential(
            # input shape (batch_size, C, H, W): [128, 1, 28, 28]
            # (CONV) -> [128, 256, 20, 20]
            nn.Conv2d(in_channels=n_conv_in_channel,
                      out_channels=n_conv_out_channel,
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

        # Reconstruction loss
        self.decoder = Decoder(n_classes, output_unit_size,
                               input_height, input_width, n_conv_in_channel,
                               use_cuda)

    def forward(self, x):
        # x shape: [128,1,28,28]
        # out conv1 shape: [128,256,20,20]
        out_conv1 = self.conv1(x)
        # out primary caps shape: [128,8,1152]
        out_primary_caps = self.primary(out_conv1)
        # out digit_caps shape: [128,10,16,1]
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, image, out_digit_caps, target, size_average=True):
        """
        Args:
            image: [batch_size,1,28,28] MNIST samples
            out_digit_caps: [batch_size,10,16,1] Output from DigitCaps layer
            target: [batch_size,10] One-hot MNIST dataset labels
            size_average: Enable mean loss (average loss over batch_size) if True

        Returns:
            total_loss: Scalar Variable of total loss
            m_loss: Scalar of margin loss
            recon_loss: Scalar of reconstruction loss
        """
        # Margin loss
        m_loss = self.margin_loss(out_digit_caps, target)
        if size_average:
            m_loss = m_loss.mean()

        # Reconstruction loss
        # Reconstruct image from Decoder network
        reconstruction = self.decoder(out_digit_caps, target)
        recon_loss = self.reconstruction_loss(reconstruction, image)
        # MSE
        if size_average:
            recon_loss = recon_loss.mean()
        # Scale down recon_loss by 0.0005 so that it does not domnitate margin loss during training
        recon_loss *= self.regularization_scale

        return m_loss + recon_loss, m_loss, recon_loss

    def margin_loss(self, input, target):
        """
        Args:
            input: [batch_size,10,16,1] Output of DigitCaps layer
            target: [batch_size,10] One-hot MNIST dataset labels

        Returns:
            l_c: Scalar of class loss (aka margin loss)
        """
        batch_size = input.size(0)

        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max terms
        zero = Variable(torch.zeros(1))
        if self.use_cuda:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5

        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
        t_c = target

        # l_c is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        return l_c

    def reconstruction_loss(self, reconstruction, image):
        """This is the sum of squared differences between
        reconstructed image (output of logistic units) and
        original image (input image)

        Args:
            reconstruction: [batch_size,784] Decoder outputs of reconstructed image tensor
            image: [batch_size,1,28,28] MNISt samples

        Returns:
            recon_error: Scalar Variable of reconstruction loss
        """
        # flat image
        image = image.view(image.size(0), -1)

        error = reconstruction - image
        recon_error = torch.sum(error**2, dim=1)

        return recon_error
