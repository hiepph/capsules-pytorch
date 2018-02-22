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
    n_train_batches = len(train_loader)
    n_test_batches = len(test_loader)

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

            # Tensorboard log
            global_step = (epoch-1) * n_train_batches + i
            writer.add_scalar('train/total_loss', loss.data[0], global_step)
            writer.add_scalar('train/margin_loss' , margin_loss.data[0], global_step)
            writer.add_scalar('train/reconstruction_loss', recon_loss.data[0], global_step)

            # STDOUT log
            if (i+1) % args.log_interval == 0:
                template = """[Epoch {}/{}]
                Total loss: {:.6f}, Margin loss: {:.6f}, Reconstruction loss: {:.6f}
                """
                tqdm.write(template.format(epoch, args.epochs,
                                           loss.data[0], margin_loss.data[0], recon_loss.data[0]))


    # Test helper
    def test(epoch):
        # Switch model to evaluate mode
        model.eval()

        loss, margin_loss, recon_loss = 0., 0., 0.
        correct = 0.

        for data, target in test_loader:
            target_indices = target

            # One-hot encode for labels
            target_one_hot = one_hot_encode(target, args.n_classes)

            # Wrap inputs into Variable
            data, target = Variable(data, volatile=True), Variable(target_one_hot)
            if args.use_cuda:
                data = data.cuda()
                target = target.cuda()

            # Forward
            output = model(data)

            # Calculate loss, and sum up
            t_loss, m_loss, r_loss = model.loss(data, output, target,
                                                 size_average=False)
            loss += t_loss.data[0]
            m_loss += m_loss.data[0]
            r_loss += r_loss.data[0]

            # Count number of correct prediction
            # v_magnitude shape: [batch_size, 10, 1, 1]
            v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
            # pred shape: [batch_size, 1, 1, 1]
            pred = v_magnitude.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target_indices.view_as(pred)).sum()

        # TODO: RECONSTRUCTION

        # Tensorboard log
        loss /= n_test_batches
        margin_loss /= n_test_batches
        recon_loss /= n_test_batches
        accuracy = correct / len(test_loader.dataset)

        global_step = epoch * n_train_batches
        writer.add_scalar('test/total_loss', loss, global_step)
        writer.add_scalar('test/margin_loss', margin_loss, global_step)
        writer.add_scalar('test/recon_loss', recon_loss, global_step)
        writer.add_scalar('test/accuracy', accuracy, global_step)

        # STDOUT log
        template = """[Test {}]
        Total loss: {:.6f}, Margin loss: {:.6f}, Reconstruction loss: {:.6f}
        Accuracy: {:.4f}%
        """
        tqdm.write(template.format(epoch,
                                   loss.data[0], margin_loss.data[0], recon_loss.data[0],
                                   accuracy * 100))


    # Save model (checkpoint) helper
    def checkpoint():
        pass


    # Start training
    for epoch in range(1, args.epochs+1):
        train(epoch)
        test(epoch)

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
    # parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()
    args.use_cuda = (not args.no_cuda) and torch.cuda.is_available()

    print(args)
    main(args)
