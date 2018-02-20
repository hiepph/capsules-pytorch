import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import CapsulesNet
from data import Data


def one_hot_encode(target, length):
    batch_size = target.size(0)
    one_hot_vec = torch.zeros((batch_size, length))
    for i in range(batch_size):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def main(args):
    # Set logger
    writer = SummaryWriter(log_dir='logs')

    # DATA
    data = Data(args)
    # Embed and visualize
    # data.embed(writer)
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

    # TRAINING
    # Train helper
    def train(epoch):
        # Switch model to train mode
        model.train()

        for i, (data, target) in enumerate(tqdm(train_loader, unit='batch')):
            # One-hot encode for labels
            target_one_hot = one_hot_encode(target, args.n_classes)

            # Wrap inputs into Variable
            data, target = Variable(data), Variable(target_one_hot)
            if args.use_cuda:
                data = data.cuda()
                target = target.cuda()

            # Forward
            optimizer.zero_grad()
            output = model(data)

            # Calculate loss
            loss, margin_loss, recon_loss = model.loss(data, output, target)

            # Backward
            loss.backward()
            optimizer.step()

            # Log
            if (i+1) % args.log_interval == 0:
                template = """[Epoch {}/{}]
                Total loss: {:.6f}, Margin loss: {:.6f}, Reconstruction loss: {:.6f}
                """
                tqdm.write(template.format(epoch, args.epochs,
                                           loss.data[0], margin_loss.data[0], recon_loss.data[0]))

    # Test helper
    def test():
        pass


    # Save model (checkpoint) helper
    def checkpoint():
        pass


    # Start training
    for epoch in range(1, args.epochs+1):
        train(epoch)

        if epoch % args.test_interval == 0:
            test()

        if epoch % args.save_interval == 0:
            checkpoint()

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)

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
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    args.use_cuda = (not args.no_cuda) and torch.cuda.is_available()

    print(args)
    main(args)
