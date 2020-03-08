from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from deprecated.squeezenet import SqueezeNet
from models.metrics import mean_absolute_error
import misc.utils
from deprecated import datasets


class Train30d:

    def __init__(self):

        # Parameters:
        self.model_path = '/mnt/share/raheppt1/pytorch_models/age/30d/'
        self.batch_size = 16
        self.max_epochs = 300
        self.learning_rate = 0.001
        self.validation_split = 50
        self.print_interval = 2
        self.weight_decay = 0.00005
        self.gamma_decay = 0.9
        self.run_name = 'convnet'
        # Define minimal evluation loss variable.
        self.val_loss_best = None

        # Create training and test dataset.
        train_dataset, validate_dataset = datasets.create_IXI_datasets(validation_split=self.validation_split)

        self.train_dl = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True, )
        self.validate_dl = torch.utils.data.DataLoader(validate_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)

        # Get cuda device.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define network architecture.
        self.net = SqueezeNet(sample_size=105, sample_duration=180, num_classes=1)
        #self.net = models.models3d.convnet.ConvNet3D()
        #net = resnet.resnet18(sample_size=240, sample_duration=180, num_classes=1)
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
        self.writer = misc.utils.init_tensorboard('30d_' + self.run_name + '_')

    @staticmethod
    def _extract_tensors(sample):
        # Extract input and target tensors.
        inputs = sample['img']['data']
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
            for step, sample in enumerate(self.validate_dl, 0):
                # Load data and send it to the GPU.
                inputs, targets = self._extract_tensors(sample)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Only forward propagation
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                mae += mean_absolute_error(outputs, targets)
                val_loss += loss.item()

            num_batches = len(self.validate_dl)
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
        for step, sample in enumerate(self.train_dl, 0):

            # Extract input and target tensors.
            inputs, targets = self._extract_tensors(sample)

            print(sample['img']['stem'][0])

            print(inputs.size())
            plt.imshow(inputs[0,0,:,:,50].numpy())
            plt.imshow(inputs[0, 0, :, :, 51].numpy())
            plt.show()
            return
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
            if step % self.print_interval == (self.print_interval-1):  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / self.print_interval))
                global_step = (epoch+1)*len(self.train_dl) + (step+1)
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
            #self.validate(epoch)

            # Decay Learning Rate
            self.scheduler.step()
        return


def main():
    trainer = Train30d()
    trainer.run()
    return


if __name__ == '__main__':
    main()