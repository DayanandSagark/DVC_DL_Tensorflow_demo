import tensorflow as tf
import os
import logging
from tensorflow.python.keras.backend import flatten
from src.utils.all_utils import get_timestamp
# We are getting the tranfer learning base model VGG16 and save this in the base model dir path
def get_VGG_16_model(input_shape, model_path):

    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    model.save(model_path)
    logging.info(f"VGG16 base model saved at: {model_path}")
    return model

 # Preparing the final model taht will take i/p base model, num of class is 2, freeze
 #every thing in base model,freeze_till as None, Learning rate
def prepare_model(model,CLASSES,freeze_all,freeze_till,learning_rate):
    if freeze_all:
        for layers in model.layers:
            layers.trainable = False
    elif(freeze_till is not None) and (freeze_till > 0):
        for layers in model.layers[:-freeze_till]:
            layers.trainable = False
    # Add our fully connected layers (from maxpooling layer in vgg 16 -  convert 7*7*12 to single array)
    # refer - https://neurohive.io/en/popular-networks/vgg16/
    # taken the base model out put and feeding it to flatten_in
    flatten_in = tf.keras.layers.Flatten()(model.output)
    # Functional approach where model.add is sequentiall approach
    prediction = tf.keras.layers.Dense(
        units = CLASSES,
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )
   
    full_model.compile(
        tf.keras.optimizers.SGD(learning_rate = learning_rate),
        tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )
    logging.info("custom model is compiled and ready to be trained")
    
    full_model.summary()
    return full_model


