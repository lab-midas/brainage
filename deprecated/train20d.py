from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchio.data.sampler.slice import SliceSampler

import models.models2d.convnet2d
from models.metrics import mean_absolute_error
from deprecated import datasets
import misc.utils


# def _create_train_queue(train_dataset):
#     # slice_train_queue = Queue(
#     #     train_dataset,
#     #     max_length=40000,
#     #     samples_per_volume=100,
#     #     patch_size=(121, 145, 1),
#     #     sampler_class=RandomLabelSampler,
#     #     num_workers=15,
#     #     shuffle_subjects=False,
#     #     shuffle_patches=True)
#     #
#     # slice_train_dl = torch.utils.data.DataLoader(slice_train_queue, batch_size=batch_size)
#     return

# todo add resnet, googlenet etc.
# ggf. resize

class Train20d:

    def __init__(self):
        # Parameters
        num_workers = 15
        batch_size = 16
        # Parameters.
        self.max_epochs = 100
        self.print_interval = 100
        self.val_loss_best = None
        self.validation_split = 50
        self.learning_rate = 0.00001
        self.weight_decay = 0.001
        self.run_name = 'convnet'
        self.model_path = '/mnt/share/raheppt1/pytorch_models/age/20d/'

        # Create training and test dataset.
        train_dataset, validate_dataset = datasets.create_IXI_datasets(validation_split=self.validation_split)

        print('Loading training data ...')
        slices_train_ds = SliceSampler(train_dataset,
                                       num_workers=num_workers,
                                       shuffle_subjects=True)

        # SliceSampler loads the whole dataset into memory! Be careful
        # with large datasets (better use a torchio/queue for these cases).
        self.slice_train_dl = torch.utils.data.DataLoader(slices_train_ds, batch_size=batch_size, shuffle=True)

        print('Loading validation data ...')
        slices_validate_ds = SliceSampler(validate_dataset,
                                          num_workers=num_workers,
                                          shuffle_subjects=False)

        # SliceSampler loads the whole dataset into memory! Be careful
        # with large datasets (better use a torchio/queue for these cases).
        self.slices_validate_dl = torch.utils.data.DataLoader(slices_validate_ds, batch_size=batch_size, shuffle=False)

        # Get cuda device.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', self.device)

        # Define network architecture.
        self.net = models.models2d.convnet2d.ConvNet2D()
        # send it to device ...
        self.net.to(self.device)
        # Optimizer.
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
        # Loss function.
        self.criterion = nn.MSELoss()

        # Initialize tensorboard writer.
        self.writer = misc.utils.init_tensorboard('20d_' + self.run_name + '_')

    @staticmethod
    def _extract_tensors(sample):
        # Extract input and target tensors.
        inputs = sample['img']['data']
        inputs = inputs[..., 0]
        # Target = slice/10.0
        targets = sample['index_ini'][:, 2] / 10.0
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
        # Running variables.
        mae = 0.0
        val_loss = 0.0
        ages = {}
        target_ages = {}
        # Evaluate network.
        self.net.eval()
        with torch.no_grad():
            for step, sample in enumerate(self.slices_validate_dl, 0):
                # Load data and send it to the GPU.
                inputs, targets = self._extract_tensors(sample)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Only forward propagation
                outputs = self.net(inputs)
                # Calculate loss and metrics.
                loss = self.criterion(outputs, targets)
                mae += mean_absolute_error(outputs, targets)

                val_loss += loss.item()

                for k, stem in enumerate(sample['img']['stem']):
                    ages[stem] = outputs[k]
                    target_ages[stem] = targets[k]

            num_batches = len(self.slices_validate_dl)
            val_loss = val_loss/num_batches
            mae = mae/num_batches

        print(ages)
        print(target_ages)
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

        print('Training')
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

            # Print statistics.
            if step % self.print_interval == (self.print_interval - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / self.print_interval))
                global_step = (epoch + 1) * len(self.slice_train_dl) + (step + 1)
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
    trainer = Train20d()
    trainer.run()


if __name__ == '__main__':
    main()