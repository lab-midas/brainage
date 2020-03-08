from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchio.data.sampler.slice import SliceSelectionSampler


import models.models2d.convnet2d
from models.metrics import mean_absolute_error
from deprecated import datasets
import misc.utils


class Train25d:

    def __init__(self):

        # Parameters:
        self.model_path = '/mnt/share/raheppt1/pytorch_models/age/25d/'
        self.batch_size = 16
        self.max_epochs = 300
        self.learning_rate = 0.0001
        self.validation_split = 50
        self.print_interval = 2
        self.weight_decay = 0.0001
        self.gamma_decay = 0.001
        self.run_name = 'convnet'
        # Define minimal evluation loss variable.
        self.val_loss_best = None
        self.num_workers = 15
        self.selected_slices = [10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90]

        # Create 3d-datasets and convert to datasets containing only the selected slices.
        train_dataset, validate_dataset = datasets.create_IXI_datasets(validation_split=50)

        # SliceSelectionSampler load the whole dataset into memory! Be careful
        # with large datasets (better use a torchio/queue for these cases).
        train_dataset_sel = SliceSelectionSampler(train_dataset,
                                                  self.selected_slices,
                                                  num_workers=self.num_workers,
                                                  shuffle_subjects=False)

        self.slice_train_dl = torch.utils.data.DataLoader(train_dataset_sel,
                                                          batch_size=self.batch_size,
                                                          shuffle=True)

        validate_dataset_sel = SliceSelectionSampler(validate_dataset,
                                                     self.selected_slices,
                                                     num_workers=self.num_workers,
                                                     shuffle_subjects=False)

        self.slice_validate_dl = torch.utils.data.DataLoader(validate_dataset_sel,
                                                             batch_size=self.batch_size,
                                                             shuffle=True)

        # Get cuda device.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define network architecture.
        self. net = models.models2d.convnet2d.ConvNet2D(
            in_channels=len(self.selected_slices))
        self.net.to(self.device)

        # Loss function.
        self.criterion = nn.MSELoss()
        # Optimizer.
        #self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.learning_rate)
        # weight_decay=weight_decay
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate,
                                         momentum=0.9, nesterov=True,
                                         weight_decay=self.weight_decay)

        # learning rate decay
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma_decay)

        # Initialize tensorboard writer.
        self.writer = misc.utils.init_tensorboard('25d_' + self.run_name + '_')

    @staticmethod
    def _extract_tensors(sample):
        # Extract input and target tensors.
        inputs = sample['img']['data']
        # Swap channel and z-dimension, so that the selected
        # slices become input channels.
        inputs = torch.transpose(inputs, 1, 4)
        inputs = inputs[..., 0]
        # Target = age/10.0
        targets = sample['img']['info'][:, 0] / 10.0
        targets = targets.unsqueeze(-1)
        return inputs, targets

    def _save_checkpoint(self, epoch, val_loss):
        checkpoint_name = self.run_name + 'chkpt_model.pt'
        print('Saving new checkpoint ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss},
            str(Path(self.model_path).joinpath(checkpoint_name)))

    def validate(self, epoch):
        # Evaluate.
        mae = 0.0
        val_loss = 0.0

        # Evaluate network.
        self.net.eval()
        with torch.no_grad():
            for step, sample in enumerate(self.slice_validate_dl, 0):
                # Load data and send it to the GPU.
                inputs, targets = self._extract_tensors(sample)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Only forward propagation
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                mae += mean_absolute_error(outputs, targets)
                val_loss += loss.item()

            num_batches = len(self.slice_validate_dl)
            val_loss = val_loss / num_batches
            mae = mae / num_batches

            # Print statistics.
            print('validation loss:', val_loss)
            print('mae:', mae)
            self.writer.add_scalar('Validation/MSE',
                                   val_loss,
                                   global_step=epoch + 1)
            self.writer.add_scalar('Validation/MAE',
                                   mae,
                                   global_step=epoch + 1)

            # Save checkpoint if the current val_loss is the lowest.
            if not self.val_loss_best:
                self.val_loss_best = val_loss
            if val_loss < self.val_loss_best:
                self.val_loss_best = val_loss
                self._save_checkpoint(epoch, val_loss)
        return

    def train(self, epoch):
        # Running variables.
        running_loss = 0.0

        # Train network for one epoch.
        self.net.train()
        for step, sample in enumerate(self.slice_train_dl, 0):

            # Extract input and target tensors.
            inputs, targets = self._extract_tensors(sample)

            # Send data to GPU.
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward + backward + optimize.
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            # Decay Learning Rate
            self.scheduler.step()

            # Print statistics.
            if step % self.print_interval == (self.print_interval-1):  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / self.print_interval))
                global_step = (epoch+1)*len(self.slice_train_dl) + (step+1)
                self.writer.add_scalar('Train/MSE',
                                       running_loss / self.print_interval,
                                       global_step=global_step)
                running_loss = 0.0
        return

    def run(self):
        for epoch in range(self.max_epochs):
            print(f'epoch {epoch}')
            # Train
            self.train(epoch)
            # Evaluate
            self.validate(epoch)


        return


def main():
    trainer = Train25d()
    trainer.run()
    return


if __name__ == '__main__':
    main()


#
# def train25d():
#     # Parameters:
#     model_path = '/mnt/share/raheppt1/pytorch_models/age/25d/'
#     batch_size = 16
#     max_epochs = 100
#     print_interval = 100
#     learning_rate = 0.00005
#     num_workers = 15
#     selected_slices = [10, 20, 30, 40]
#
#     # Create 3d-datasets and convert to datasets containing only the selected slices.
#     train_dataset, eval_dataset = datasets.create_IXI_datasets(validation_split=10)
#
#     # SliceSelectionSampler load the whole dataset into memory! Be careful
#     # with large datasets (better use a torchio/queue for these cases).
#     train_dataset_sel = SliceSelectionSampler(train_dataset,
#                                               selected_slices,
#                                               num_workers=num_workers,
#                                               shuffle_subjects=False)
#
#     slice_train_dl = torch.utils.data.DataLoader(train_dataset_sel, batch_size=batch_size, shuffle=True)
#
#     eval_dataset_sel = SliceSelectionSampler(eval_dataset,
#                                              selected_slices,
#                                              num_workers=num_workers,
#                                              shuffle_subjects=False)
#
#     slice_eval_dl = torch.utils.data.DataLoader(eval_dataset_sel, batch_size=batch_size, shuffle=True)
#
#     # Initialize tensorboard writer.
#     writer = misc.utils.init_tensorboard('25d_')
#
#     # Get cuda device.
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # Define network architecture.
#     net = models.models2d.convnet2d.ConvNet2D(in_channels=len(selected_slices))
#     net.to(device)
#
#     # Loss function.
#     criterion = nn.MSELoss()
#     # Optimizer.
#     optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
#
#     # Define minimal evluation loss variable.
#     eval_loss_min = None
#
#     for epoch in range(max_epochs):
#         print(f'epoch {epoch}')
#
#         # Train
#         running_loss = 0.0
#         net.train()
#         for step, sample in enumerate(slice_train_dl, 0):
#             # Extract input and target tensors.
#             inputs, targets = _extract_tensors(sample)
#             # Send data to GPU.
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#             # Forward + backward + optimize.
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             # Print statistics.
#             running_loss += loss.item()
#             if step % print_interval == (print_interval - 1):  # print every 10 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, step + 1, running_loss / print_interval))
#                 global_step = (epoch + 1) * len(slice_train_dl) + (step + 1)
#                 writer.add_scalar('training loss',
#                                   running_loss / print_interval,
#                                   global_step=global_step)
#                 running_loss = 0.0
#
#         # Evaluate
#         mae = 0.0
#         running_loss = 0.0
#         step = 0
#         net.eval()
#         with torch.no_grad():
#             for step, sample in enumerate(slice_eval_dl, 0):
#                 inputs, targets = _extract_tensors(sample)
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = net(inputs)
#                 loss = criterion(outputs, targets)
#                 mae += mean_absolute_error(outputs, targets)
#                 running_loss += loss.item()
#                 step = step + 1
#
#         # Tensorboard logs.
#         writer.add_scalar('evaluation_loss',
#                           running_loss / step,
#                           global_step=epoch + 1)
#         writer.add_scalar('evaluation_mae',
#                           mae / step,
#                           global_step=epoch + 1)
#
#         # Save checkpoint if the current eval_loss is the lowest.
#         eval_loss = running_loss
#         if not eval_loss_min:
#             eval_loss_min = eval_loss
#
#         if eval_loss < eval_loss_min:
#             eval_loss_min = eval_loss
#             print('Saving new checkpoint ...')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': net.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': eval_loss_min},
#                  str(Path(model_path).joinpath('chkpt_model.pt')))
#
#         print('eval loss:', running_loss / step)
#         print('mae:', mae / step)
#
#     return
#
#
# def main():
#     train25d()
#
#
# if __name__ == '__main__':
#     main()