import numpy as np
from torch.utils.data import dataset

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchio.data.sampler.slice import SliceSampler
import torchio.utils

import models.models2d.siamese
import misc.utils
from deprecated import datasets


class SiameseDatasetB(Dataset):

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, indexa):
        x = np.random.rand()

        sample_a = self.ds[indexa]
        slice_a = sample_a['index_ini'][2]

        while True:
            # find similiar slices
            indexb = np.random.choice(range(self.__len__()))
            sample_b = self.ds[indexb]
            slice_b = sample_b['index_ini'][2]
            slice_distance = abs(slice_a - slice_b)
            if ((slice_distance < 3 and x > 0.5) or
                    (slice_distance >= 3 and x <= 0.5)):
                break

        return {'imga': sample_a,
                'imgb': sample_b}

    def __len__(self):
        return len(self.ds)



class SiameseDataset(SliceSampler):

    def __init__(self, subjects_dataset,
                 num_workers=0,
                 shuffle_subjects=False,
                 delete_dim=False):

        super().__init__(subjects_dataset,
                 num_workers=0,
                 shuffle_subjects=False,
                 delete_dim=False)
        # Sort subjects according to their age.
        self.subjects.sort(key=lambda slicelist: self._get_age(slicelist[0]))

    @staticmethod
    def _get_age(sample):
        # Get the age from the first image in sample.
        for key, value in sample.items():
            if torchio.utils.is_image_dict(value):
                return value['info'][0]

    def __getitem__(self, index_a):

        self.index_list[index_a]

        x = np.random.rand()
        subj_a, slice_a = self.index_list[index_a]

        subj_b = np.random.choice(range(subj_a, len(self.subjects)))
        slice_b = np.random.choice(range(len(self.subjects[subj_b])))
        sample_a = self.subjects[subj_a][slice_a]
        sample_b = self.subjects[subj_b][slice_a]

        return {'imga': sample_a,
                'imgb': sample_b}

    def __len__(self):
        return super().__len__()

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def _extract_tensors(sample):
    # Extract input and target tensors.
    inputs_a = sample['imga']['img']['data'][..., 0]
    inputs_b = sample['imgb']['img']['data'][..., 0]

    # Target = Age
    #targets = 0.0
    #targets = targets.unsqueeze(-1)
    #age_a = sample[0]['img']['info'][:, 0] / 10.0
    #age_b = sample[1]['img']['info'][:, 0] / 10.0
    #targets = torch.abs(age_a - age_b)

    slice_a = sample['imga']['index_ini'][:, 2]
    slice_b = sample['imgb']['index_ini'][:, 2]
    targets = torch.abs(slice_a - slice_b) > 2
    targets = targets.unsqueeze(-1).type(torch.float32)
    return inputs_a, inputs_b, targets


def _plot_images(sample):
    # Extract information from sample.
    inputs_a = sample['imga']['img']['data'][..., 0]
    inputs_b = sample['imgb']['img']['data'][..., 0]
    slice_a = sample['imga']['index_ini'][:, 2][0]
    slice_b = sample['imgb']['index_ini'][:, 2][0]
    id_a = sample['imga']['img']['id'][0]
    id_b = sample['imgb']['img']['id'][0]
    age_a = sample['imga']['img']['info'][0][0]
    age_b = sample['imgb']['img']['info'][0][0]

    fig, ax = plt.subplots()
    tmp = torch.cat([inputs_a[0, 0, ...].squeeze(),
                     inputs_b[0, 0, ...].squeeze()],
                    dim=1)

    ax.imshow(tmp, cmap='gray')
    ax.set_title(f'id {id_a}, {id_b} - slice {slice_a}, {slice_b} - age {age_a:.2f}, {age_b:.2f} - loss: {0.8}')
    ax.set_axis_off()
    return fig


def train2d_siamese():
    # Parameters.
    model_path = '/mnt/share/raheppt1/pytorch_models/age/siam20d/'
    batch_size = 16
    max_epochs = 100
    print_interval = 100
    learning_rate = 0.001
    num_workers = 15

    # Create datasets
    train_dataset, eval_dataset = datasets.create_IXI_datasets(500)

    # SliceSampler load the whole dataset into memory! Be careful
    # with large datasets (better use a torchio/queue for these cases).
    #slices_train_ds = SliceSampler(train_dataset,
    #                               num_workers=num_workers,
    #                               shuffle_subjects=True)

    siamese_ds_train = SiameseDataset(train_dataset,
                                      num_workers=num_workers,
                                      delete_dim=False)
    siamese_dl_train = torch.utils.data.DataLoader(siamese_ds_train, batch_size=batch_size, shuffle=False)

    # SliceSampler load the whole dataset into memory! Be careful
    # with large datasets (better use a torchio/queue for these cases).
    #slices_eval_ds = SliceSampler(eval_dataset,
    #                              num_workers=num_workers,
    #                              shuffle_subjects=True)

    #siamese_ds_eval = SiameseDataset(slices_eval_ds)
    #siamese_dl_eval = torch.utils.data.DataLoader(siamese_ds_eval, batch_size=batch_size, shuffle=True)

    # Initialize tensorboard writer.
    writer = misc.utils.init_tensorboard('siam20d_')

    # Get cuda device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define network architecture.
    net = models.models2d.siamese.Siamese()
    net.to(device)

    # Loss function.
    criterion = ContrastiveLoss()
    # Optimizer.
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

    # Define minimal evluation loss variable.
    eval_loss_min = None

    for epoch in range(max_epochs):
        print(f'epoch {epoch}')

        # Train
        running_loss = 0.0
        net.train()

        for step, sample in enumerate(siamese_dl_train, 0):

            # Extract input and target tensors.
            inputs_a, inputs_b, targets = _extract_tensors(sample)
            age_a = sample['imga']['img']['info'][:, 0][0]
            age_b = sample['imgb']['img']['info'][:, 0][0]
            # targets = torch.abs(age_a - age_b)

            slice_a = sample['imga']['index_ini'][:, 2][0]
            slice_b = sample['imgb']['index_ini'][:, 2][0]
            print(slice_a, slice_b, age_a, age_b)
        return
        for step, sample in enumerate(siamese_dl_train, 0):

            # Extract input and target tensors.
            inputs_a, inputs_b, targets = _extract_tensors(sample)
            print(inputs_b)
            # Send data to GPU.
            inputs_a, inputs_b = inputs_a.to(device), inputs_b.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients.
            optimizer.zero_grad()
            # Forward + backward + optimize.
            output_a, output_b = net(inputs_a, inputs_b)
            loss = criterion(output_a, output_b, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print statistics.
            if step % print_interval == (print_interval - 1):  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / print_interval))
                global_step = (epoch + 1) * len(siamese_dl_train) + (step + 1)
                writer.add_scalar('Train/ContrastiveLoss',
                                  running_loss / print_interval,
                                  global_step=global_step)
                running_loss = 0.0

        print('evaluation')
        # Evaluation
        eval_loss = 0.0
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(siamese_dl_eval, 0):
                # Extract input and target tensors.
                inputs_a, inputs_b, targets = _extract_tensors(sample)

                # Send data to GPU.
                inputs_a, inputs_b = inputs_a.to(device), inputs_b.to(device)
                targets = targets.to(device)
                # Forward propagation only.
                output_a, output_b = net(inputs_a, inputs_b)
                loss = criterion(output_a, output_b, targets)
                eval_loss += loss.item()

                if step % print_interval == (print_interval - 1):
                    out_fig = _plot_images(sample)
                    writer.add_figure(tag='sample', figure=out_fig, global_step=epoch)

        eval_loss = eval_loss/len(siamese_dl_eval)
        print('eval loss: ', eval_loss)



    return


def main():
    train2d_siamese()

if __name__ == '__main__':
    main()