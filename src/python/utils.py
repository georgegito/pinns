import numpy as np
import torch
import csv
import os

def stack_xyzt_tensors(x, y, z, t):
  return torch.stack((x, y, z, t), axis=-1)

def stack_xyz_tensors(x, y, z):
  return torch.stack((x, y, z), axis=-1)

def tensor_from_array(arr, device, requires_grad):
  return torch.tensor(arr, device=device, requires_grad=requires_grad, dtype=torch.float32)

def save_checkpoint(pinn, epoch, optimizer, filepath):
  print("=> saving checkpoint '{}'".format(filepath))
  state = {'epoch': epoch, 'state_dict': pinn.state_dict(),
             'optimizer': optimizer.state_dict()}
  torch.save(state, filepath)

def load_checkpoint(model, optimizer, filepath):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filepath, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))

    return model, optimizer, start_epoch

def sample_points_in_domain(_min, _max, num_of_samples):
  return np.random.uniform(_min, _max, size=num_of_samples)

def zeros(num):
  return np.zeros(num)

def ones(num):
  return np.ones(num)