import tensorflow as tf
from pathlib import Path
from deprecated.TF import input_fn

train_loss = tf.keras.metrics.MeanSquaredError()

def main():
    # Path to tfrecord files and xls with subject information.
    path_info = '/Users/tobiashepp/projects/age_prediction/data/IXI.xls'
    path_tfrec = Path('/Users/tobiashepp/projects/age_prediction/data/tfrec')

    # List with selected subjects.
    subject_list = [2, 12]

    ds_imgs = input_fn.create_ds_images(subject_list,
                                        path_tfrec)

    ds_info = input_fn.create_ds_info(subject_list,
                                      path_info)

    # debug
    for t in ds_info.take(2):
        print(f'targets {t}')

    # Zip (3d images and info) to dataset.
    ds = tf.data.Dataset.zip((ds_imgs, ds_info))

    # debug
    for image, info in ds.take(2):
        print(f'image shape {tf.shape(image)} : AGE, SEX_ID: {info}')

    # Create dataset with (slice numbers, 2d slices image and info) elements.
    ds_slices = ds.flat_map(input_fn.img_to_slices)

    # debug
    ds_slices = ds_slices.shuffle(1000).batch(32)
    sl, img, info = next(iter(ds_slices))
    #plt.imshow(img[0])
    #plt.show()

    # PP is already normalized.
    # todo Add normalization Preprocessing
    # todo Add noise preprocessing

    from models.models2d import convnet2d
    model = convnet2d.MyModel()

    for sl, img, info in ds_slices.take(10):
        print(tf.shape(img))
        output = model(img[:, :, :, tf.newaxis])

        loss = tf.keras.losses.MSE(sl, output)
        print(output)
        print(loss)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = tf.keras.losses.MSE(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    EPOCHS = 100

    print('Training started ...')
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        for sl, img, info in ds_slices:
            train_step(img[:, :, :, tf.newaxis], sl)
        print(train_loss.result())


if __name__ == '__main__':
    main()