def _get_slice(filename):
    """
    Gets slice number from filename .._sl<slice_number(4)>.nii
    Args:
        filename:

    Returns: Slice number

    """
    # Parse subject id from filename
    pattern = '.*_sl([0-9]{4}).*'
    m = re.match(pattern, filename)
    sl = int(m.group(1))
    return sl



class IXI2DDataset(dataset):

    def __init__(self,
                 filenames,
                 path_info,
                 columns,
                 transform=None):

        # Read csv file with additional patient information.
        self.df_info = pd.read_csv(path_info, index_col='IXI_ID')
        self.columns = columns
        self.transform = transform
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = self.filenames[idx]
        img = np.load(file)
        # expand for channel
        img = np.expand_dims(img, 0)

        noise = np.random.normal(0.0, 0.25, img.shape)
        img = img + noise
        img = img.astype(np.float32)

        if self.transform:
            img = self.transform(img)

        f = Path(file).stem
        id = _get_id(f)
        info = _get_info(id, self.df_info, self.columns)
        sl = np.array([_get_slice(f)/10.0]).astype(np.float32)

        age = np.array([info[0]/10.0]).astype(np.float32)

        return {'id': id,
                'info': age,
                'slice': sl,
                'img': img}




def test2d_training():

    # Path to nii files and csv with subject information.
    path_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'
    path_nii = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/npy2dz')
    path_subjects = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'
    # Columns to extract target information from.
    columns = ['AGE', 'SEX_ID']

    # Create list with selected subjects from csv file.
    df_subjects = pd.read_csv(path_subjects)
    subject_list = df_subjects['ID'].values

    # Convert subject indices to nii filenames.
    pattern = 'smwc1rIXI{:03d}*'
    dirnames = _id2niipath(subject_list, path_nii, pattern)

    filenames = []
    for dir in dirnames:
        filenames += [str(f) for f in Path(dir).glob('*npy*')]

    dataset = IXI2DDataset(filenames[100:], path_info, columns)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    eval_dataset = IXI2DDataset(filenames[:100], path_info, columns)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=True)

    import torch.optim as optim

    net = ConvNet()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    net.to(device)

    for epoch in range(1):
        running_loss = 0.0
        for i, sample in enumerate(dataloader, 0):

            inputs, slices, info = sample['img'].to(device), sample['slice'].to(device), sample['info'].to(device)

            # Get network input data.
            #plt.imshow(inputs[0,0,:,:])
            #plt.show()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, info)
            loss.backward()
            optimizer.step()
            #print(outputs[0], slices[0], loss)

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        for i, sample in enumerate(eval_dataloader, 0):
            inputs, slices, info = sample['img'].to(device), sample['slice'].to(device), sample['info'].to(device)
            outputs = net(inputs)
            print(slices[0], outputs[0], info[0])



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(34560, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out