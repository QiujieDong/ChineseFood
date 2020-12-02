from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def setup_generator(train_path, test_path, batch_size, dimentions):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        test_path, # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    return train_generator, validation_generator


def load_image(img_path, dimentions, rescale=1. / 255):
    img = load_img(img_path, target_size=dimentions)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x


def get_classes(file_path):
    with open(file_path) as f:
        classes = f.read().splitlines()

    return classes


def create_model(num_classes, dropout, shape):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(
            shape=shape))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model_final = Model(inputs=base_model.input, outputs=predictions)

    return model_final


def train_model(model_final, train_generator, validation_generator, callbacks, args):
    model_final.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model_final.fit_generator(train_generator, validation_data=validation_generator,
                              epochs=args.epochs, callbacks=callbacks,
                              steps_per_epoch=train_generator.samples//args.batch_size,
                              validation_steps=validation_generator.samples//args.batch_size)


#def load_model(model_final, weights_path, shape):
def load_model(weights_path, shape):
   model_final = create_model(203, 0, shape)
   model_final.load_weights(weights_path)

   return model_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Food 203 Program')
    parser.add_argument('-m', help='train or inference model', dest='mode',
                        type=str, default='test')
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-p', help='path of the saved model', dest='model_path',
                        type=str, default='./ChineseFood/savedModels/weights.epoch-95-val_loss-val_acc-0.31.hdf5')
    parser.add_argument('-i', help='path to test image', dest='image_path',
                        type=str, default='./ChineseFood/new_picture/1.jpg')
    parser.add_argument('-e', help='epochs to train the model', dest='epochs',
                        type=int, default=77)
    parser.add_argument('-d', help='decimal value for dropout', dest='dropout',
                        type=float, default=0.2)
    parser.add_argument('-c', help='the model path to continue training', dest='continue_train_path',
                        type=str, default='./ChineseFood/savedModels/weights.epoch-95-val_loss-val_acc-0.31.hdf5')


    args = parser.parse_args()

    shape = (224, 224, 3)

    if args.mode == 'train':
        X_train, X_test = setup_generator('./ChineseFood/train', './ChineseFood/test', args.batch_size, shape[:2])

        # debug purposes
        print(X_train)

        # call backs have to be array
        callbacks = []

        # add a callback
        #callbacks.append(TensorBoard(log_dir='./ChineseFood/log/'))
        callbacks.append(ModelCheckpoint(filepath='./ChineseFood/savedModels/weights.epoch-{epoch:02d}-val_loss-{val_loss:.2f}-val_acc-{val_acc:.2f}.hdf5',
                                       verbose=1, save_best_only=True))


        model_final = create_model(X_train.num_classes, args.dropout, shape)
        train_model(model_final, X_train, X_test, callbacks, args)
    elif args.mode == 'continue_train':
        X_train, X_test = setup_generator('./ChineseFood/train', './ChineseFood/test', args.batch_size, shape[:2])

        # debug purposes
        print(X_train)

        # call backs have to be array
        callbacks = []

        # add a callback
        #callbacks.append(TensorBoard(log_dir='./ChineseFood/log/'))
        callbacks.append(ModelCheckpoint(filepath='./ChineseFood/savedModels/weights.epoch-{epoch:02d}-val_loss-{val_loss:.2f}-val_acc-{val_acc:.2f}.hdf5',
                                       verbose=1, save_best_only=True))


        #model_final = create_model(X_train.num_classes, args.dropout, shape)
        model_final = load_model(args.continue_train_path, shape)
        train_model(model_final, X_train, X_test, callbacks, args)
    else:
        # trained_model = load_model(model_final, args.model_path, shape)
        trained_model = load_model(args.model_path, shape)
        image = load_image(args.image_path, shape[:2])
        preds = trained_model.predict(image)
        classes = get_classes('./ChineseFood/meta/labels.txt')
        # Top-1
        print("The image is: {} - {:.2f}%".format(classes[np.argmax(preds)], max(preds[0, :]) * 100))
        
        # Top-5
        preds_ = np.argsort(-preds) #descending sort
        print('The TOP-5 predictions!')
        for i in range(0,5):
            print("{}. {} - {:.2f}%".format((i+1), classes[preds_[0, i]], preds[0, preds_[0, i]] * 100))

