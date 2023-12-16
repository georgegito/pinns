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

def save_log(log_filepath, epoch, epoch_loss):
  new_row = {
    "epoch": epoch,
    "total_loss": epoch_loss[0],
    "pde_loss": epoch_loss[1],
    "ic_loss": epoch_loss[2],
    "bc_loss": epoch_loss[3],
    "no_slip_loss": epoch_loss[4]
  }

  with open(log_filepath, 'a', newline='') as file:

    writer = csv.DictWriter(file, fieldnames=new_row.keys())

    file.seek(0, 2)
    if file.tell() == 0:
      writer.writeheader()

    writer.writerow(new_row)

def print_log(epoch, epoch_loss):
    print(f'Epoch: {epoch},\tTotal loss: {epoch_loss[0]},\tPDE loss: {epoch_loss[1]}\tIC loss: {epoch_loss[2]},\tBC loss: {epoch_loss[3]},\tNo-slip loss: {epoch_loss[4]}')

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