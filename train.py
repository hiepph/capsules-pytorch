import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from model import CapsulesNet

from data import Data


def main(args):
    # Set logger
    writer = SummaryWriter(log_dir='logs')

    # DATA
    data = Data(args)
    # Embed and visualize
    data.embed(writer)
    # Load data loader
    train_loader, test_loader = data.load()

    # MODEL
    model = CapsulesNet(args.n_conv_in_channel, args.n_conv_out_channel,
                        args.n_primary_unit, args.primary_unit_size,
                        args.n_classes, args.output_unit_size,
                        args.n_routing, args.regularization_scale,
                        args.input_width, args.input_height,
                        args.use_cuda)

    if args.use_cuda:
        # Use multiple GPUs if possible
        if torch.cuda.device_count() > 1:
            print('[INFO] Using {} GPUs'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

        model.cuda()

    # Info
    print(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train & test


    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--n_conv_in_channel', type=int, default=1)
    parser.add_argument('--n_conv_out_channel', type=int, default=256)
    parser.add_argument('--n_primary_unit', type=int, default=8)
    parser.add_argument('--primary_unit_size', type=int, default=1152)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--output_unit_size', type=int, default=16)
    parser.add_argument('--n_routing', type=int, default=3)
    parser.add_argument('--regularization_scale', type=float, default=0.0005)
    parser.add_argument('--input_height', type=int, default=28)
    parser.add_argument('--input_width', type=int, default=28)

    parser.add_argument('--no_cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    main(args)
