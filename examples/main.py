import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("..")
from utilities import *


def get_parser():
    parser = argparse.ArgumentParser('OCN examples')
    parser.add_argument('--model_true', type=str, choices=['LGF', 'NGF', 'Pendulum', 'Lorenz'], default='Pendulum')
    parser.add_argument('--model_ocn', type=str, choices=['OCN', 'OCN_GF'], default='OCN')
    parser.add_argument('--get_grad', type=str, choices=['bpp', 'splt'], default='splt')
    parser.add_argument('--x0', nargs='+', type=float, default=[-1, -1])
    parser.add_argument('--train_time', type=float, default=5)
    parser.add_argument('--step_size', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_time', type=int, default=0)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--hidden_neurons', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=10000)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--resume', action='store_true')
    return parser


parser = get_parser()
args = parser.parse_args()
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
ckpt_name = get_ckpt_name(model_true=args.model_true,
                          hidden_layers=args.hidden_layers,
                          hidden_neurons=args.hidden_neurons,
                          train_time=args.train_time, 
                          batch_size=args.batch_size)

# Load checkpoint
if args.resume:
    ckpt = load_checkpoint(ckpt_name)
    best_loss = ckpt['loss']
    start_epoch = ckpt['epoch']
else:
    ckpt = None
    best_loss = 1e8
    start_epoch = -1

if args.get_grad == 'bpp':
    from torchdiffeq import odeint
elif args.get_grad == 'splt':
    from torch_symplectic_adjoint import odeint_symplectic_adjoint as odeint


# Problem setup
model_true = eval(args.model_true)
model_ocn = eval(args.model_ocn)
y0 = initial_point(args, device)

# Generate training data and divide the dataset into mini-batches
true_y_train, train_t = build_dataset(args, odeint, model_true=model_true, data_type='train', y0=y0, device=device)
batch_y0, batch_t, batch_y = get_batch(args, true_y_train, train_t, device)
if args.model_ocn == "OCN_GF":
    batch_y0.requires_grad = True

# Build neural networks 
trf = model_true().to(device)
ocn = build_model(args, model_ocn, None, device, ckpt_name, ckpt=ckpt)
lossfunc = nn.MSELoss()
optimizer = optim.Adam(ocn.parameters(), args.lr, weight_decay=args.weight_decay)

# Training
batch_losses = []
train_losses = []
train_func_losses = []

for epoch in range(start_epoch + 1, args.nepochs):
    batch_loss = train(args, odeint, ocn, lossfunc, optimizer, batch_y0, batch_t, batch_y, epoch, device)
    train_loss, train_func_loss = test(args, odeint, ocn, trf, lossfunc, true_y_train, train_t, y0, epoch, device)

    # Save checkpoint
    if train_loss < best_loss:
        state = {
            'params': ocn.state_dict(),
            'loss': train_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join('checkpoint', ckpt_name))
        best_loss = train_loss

    batch_losses.append(batch_loss)
    train_losses.append(train_loss)
    train_func_losses.append(train_func_loss)

    if not os.path.isdir('curve'):
        os.mkdir('curve')
    torch.save({'batch_loss': batch_losses,
                'train_loss': train_losses,
                'train_func_loss': train_func_losses,
                },os.path.join('curve', ckpt_name))

