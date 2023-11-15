import os
import numpy as np
import torch
import torch.nn as nn

# for gradient flows, use the following
# from torch_symplectic_adjoint import odeint
# from torch_symplectic_adjoint import odeint_symplectic_adjoint as odeint

from .models_ocn import *
from .models_true import *



def initial_point(args, device):
    if args.model_true == 'LGF':
        y0 = torch.tensor([[-2., 0.], [0., -2.], [2., 0.], [0., 2.], 
                           [-2., -2.], [2., 2.], [-2., 2.], [2., -2.]]).to(device)
        y0.requires_grad = True
    elif args.model_true == 'NGF':
        y0 = torch.tensor([[-2., 3.8], [-2.2, 3.5], [-2.2, 2.8],[-2.2, 2.5],
                           [-1.2, 2.5],[-1.2, 2.8], [-1.2, 3.5],[-1.1, 4],
                           [1.2, 5.9], [1.9, 5.8], [4, 3.5],[4, 2.5],
                           [2, 0.5], [1.2, 0.3], [1.2, -0.3],[2.2, 0.5],
                           [2.2, -0.5],[2, -0.5],[1, -1],[1.2, 0.5],
                           [-2.2, -2.5],[-1.5, -3],[-4, 0.5],[-4, -0.5]]).to(device)
        y0.requires_grad = True
    elif args.model_true == 'Lorenz' and args.x0 == [0., 0., 0.]:
        x0 = np.load('output/T3_initial_points_train.npy')
        y0 = torch.tensor(x0).to(device)
    else: 
        y0 = torch.tensor([args.x0]).to(device)
    return y0





def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)

def get_ckpt_name(model_true='Pendulum', hidden_layers=1, hidden_neurons=100, 
                  train_time=5, batch_size=2):
    return '{}-G{}x{}-T{}-bs{}'.format(
        model_true, hidden_layers, hidden_neurons, train_time, batch_size)

def get_batch(args, true_y, t, device):
    # Divide training dataset into batches

    total_number_of_intervals = round(args.train_time / args.step_size)
    number_of_intervals_per_batch = args.batch_size - 1
    batch_nums = int(total_number_of_intervals / number_of_intervals_per_batch)

    print("The training time interval is [0, {}]".format(args.train_time))
    print("The step size is set as {}".format(args.step_size))
    print("There are {} intervals in total".format(total_number_of_intervals))
    print("with {} intervals in each batch".format(number_of_intervals_per_batch))
    print("We divide the dataset into {} batches".format(batch_nums))

    s = torch.linspace(0, total_number_of_intervals-number_of_intervals_per_batch, 
                       steps=batch_nums, dtype=torch.long)
    batch_y0 = true_y[s] # number of batches x number of y0 x dimension of y0
    batch_t = t[:args.batch_size] # number of points per batch
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_size)], dim=0)
    # number of points per batch x number of batches x number of y0 x dimension of y0
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)




def build_dataset(args, odeint, model_true, data_type, y0, device):
    y0 = y0.to(device)

    # initialization of the model
    model = model_true()
    
    if data_type == 'train':
        train_size = round(args.train_time / args.step_size)+1
        t = torch.linspace(0., args.train_time, train_size).to(device)
    elif data_type == 'test':
        test_size = round(args.test_time / args.step_size)+1
        t = torch.linspace(0., args.test_time, test_size).to(device)

    with torch.no_grad():
        true_y = odeint(model, y0, t, method="dopri5")
    return true_y, t

def build_model(args, model_nn, y0, device, ckpt_name, ckpt=None):
    ocn = model_nn(args).to(device)

    if ckpt:
        ocn.load_state_dict(torch.load(os.path.join('checkpoint', ckpt_name))['params'])
    return ocn





def train(args, odeint, ocn, lossfunc, optimizer, batch_y0, batch_t, batch_y, epoch, device):
    optimizer.zero_grad()
    pred_batch_y = odeint(ocn, batch_y0, batch_t, method="dopri5").to(device)
    loss = lossfunc(pred_batch_y, batch_y)
    loss.backward()
    optimizer.step()

    # if epoch % args.print_freq == 0:
    #     print('Epoch {:04d} | Batch Loss {:.6f}'.format(epoch, loss.item()))

    return loss.item()


def test(args, odeint, ocn, trf, lossfunc, true_y_train, train_t, y0, epoch, device):
    pred_y_train = odeint(ocn, y0, train_t).to(device)
    train_loss = lossfunc(pred_y_train, true_y_train)
    train_func_loss = lossfunc(ocn.F(true_y_train), trf.F(true_y_train))

    if epoch % args.print_freq == 0:
        print('Epoch {:04d} | Train Loss {:.6f} | Train func Loss {:.6f}'.format(
                epoch, train_loss.item(), train_func_loss.item()))

    return train_loss.item(), train_func_loss.item()