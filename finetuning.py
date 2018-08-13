# -*- encoding: UTF-8 -*-
import json
import os
import pickle
from datetime import datetime

from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.python.keras.layers import (Activation, Dense, Dropout,
                                            Flatten, Input)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import (ImageDataGenerator,
                                                         load_img)

TRAIN_DATA_DIR = "img/shrine_temple/train"
VALID_DATA_DIR = "img/shrine_temple/validation"


def build_finetuning_model(vgg16):
    """
    Args:
        vgg16: pretrained vgg16 model imported from keras
    return:
        model: layers added model
    """
    x = Flatten()(vgg16.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='relu')(x)
    model = Model(inputs=vgg16.input, outputs=predictions)

    for layer in model.layers[:15]:
        layer.trainable = False

    return model


if __name__ == "__main__":
    # exclude top layer
    vgg16 = VGG16(weights="imagenet", include_top=False,
                  input_shape=(224, 224, 3))
    model = build_finetuning_model(vgg16)
    model.compile(
        loss='binary_crossentropy',
        optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )

    model.summary()

    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    test_datagem = ImageDataGenerator(rescale=1/255.)

    train_itr = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary'
    )

    valid_itr = train_datagen.flow_from_directory(
        VALID_DATA_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary'
    )

    MODEL_DIR = os.path.join('model', datetime.now().strftime('%y%m%d_%H%M'))
    os.mkdir(MODEL_DIR, exist_ok=True)
    WEIGHTS_DIR = os.path.join(MODEL_DIR, 'weights')
    os.mkdir(WEIGHTS_DIR, exist_ok=True)

    model_json = os.path.join(MODEL_DIR, 'model.json')
    with open(model_json, 'w') as f:
        json.dump(model.to_json(), f)

    model_class = os.path.join(MODEL_DIR, 'class.pkl')
    with open(model_class, 'wb') as f:
        pickle.dump(train_itr.class_indices, f)

    import math

    batch_size = 16
    steps_per_epoch = math.ceil(
        train_itr.sample/batch_size
    )
    validation_steps = math.ceil(
        valid_itr.sample/batch_size
    )

    cp_filepath = os.path.join(WEIGHTS_DIR, 'ep_{epoch: 02d}_ls_{loss:.1f}.h5')

    cp = ModelCheckpoint(
        cp_filepath,
        monitor='loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
        mode='auto',
        period=5)

    csv_filepath = os.path.join(MODEL_DIR, 'loss.csv')
    csv = CSVLogger(csv_filepath, append=True)

    n_epoch = 1

    history = model.fit_generator(
        train_itr,
        septs_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        validation_data=valid_itr,
        validation_steps=validation_steps,
        callback=[cp, csv]
    )
