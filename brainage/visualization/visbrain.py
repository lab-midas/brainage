
def vis_brain(image, slice_sel, axis_sel=1, normalize=False):
    if normalize:
        image = (image - np.mean(image)) / (np.std(image))

    if axis_sel == 1:
        image = image.transpose([1, 2, 0])
    elif axis_sel == 2:
        image = image.transpose([2, 0, 1])

    fig = plt.figure(figsize=(7., 7.))
    plt.imshow(image[slice_sel, :, :], cmap='gray')
    plt.axis('off')
    plt.show()
