# For python 2 support
from __future__ import absolute_import, print_function, division, unicode_literals

# import necessary packages
import tensorflow as tf

# Import helper packages
import numpy as np
import pandas as pd
import sys, traceback

# import in-house packages
import preprocessing_utils
import model_utils.densenet_model as densenet_model
import os_utils.os_utilities as os_utilities

def new_train_model(config: dict,):
    """
    Function to new train a model from scratch
    
    Arguments:
        config {dict} -- The configuration to train consisting of 
        {
            "no_of_epochs" : no of epochs to train
            "learning_rate" : The learning rate
            "batch_size" : The batch size to use
            "threads" : The no. of threads to use
            "gpus" : The total number of gpus to use 
            "train_data_path": The meta data for training
            "val_data_path": The meta data for validation
            "test_data_path": The meta data for testing
            "model_name" : The model name, NOTE: This will be used to save the model.
        }
    """
    # Get the configuration to train
    no_of_epochs = config["no_of_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    model_name = config["model_name"]
    no_gpus = config["gpus"]
    train_data_path = config["train_data_path"]
    val_data_path =  config["val_data_path"]
    test_data_path = config["test_data_path"]

    print("[INFO]: Using config: \n", config)
    # Set up environment
    folders = [
        "./models/checkpoints/", 
        "./output/csv_log/",
        "./models/best_model",
        "./models/saved_model",
        "./output/graphs/"
    ]
    os_utilities.make_directory(folders)

    # Get steps_per_epoch
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)
    train_steps = int(len(train_data) / batch_size)
    val_steps = int(len(val_data) / batch_size)
    test_steps = int(len(test_data) / batch_size)
    del train_data
    del val_data
    del test_data
    print("[INFO]:Train Steps per Epoch: ", train_steps)
    print("[INFO]:Train Steps per Epoch: ", val_steps)
    print("[INFO]:Train Steps per Epoch: ", test_steps)

    # Load the generators for the data
    train_gen = preprocessing_utils.load_tf_dataset_generator(dataset_meta_path = train_data_path, batch_size = batch_size, shuffle = True, shuffle_buffer = 1000)
    val_gen = preprocessing_utils.load_tf_dataset_generator(dataset_meta_path = val_data_path, batch_size = batch_size, shuffle = False, shuffle_buffer = 100)
    test_gen = preprocessing_utils.load_tf_dataset_generator(dataset_meta_path = test_data_path, batch_size = batch_size, shuffle = False, shuffle_buffer = 100)
    
    # instantiate the new model and start train
    if no_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()

        print("[INFO]: No. of GPU devices used: ", strategy.num_replicas_in_sync)
        with strategy.scope():
            model = densenet_model.densenet121_xray_model()
            losses = ["categorical_crossentropy" for i in range(14)]
            metrics = ["accuracy" for i in range(14)]
            model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), loss = losses, metrics = metrics)
    
    else:
        model = densenet_model.densenet121_xray_model()
        losses = ["categorical_crossentropy" for i in range(14)]
        metrics = ["accuracy" for i in range(14)]
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), loss = losses, metrics = metrics)
    
    # set callbacks the model
    csv_callback = tf.keras.callbacks.CSVLogger("./output/csv_log/{0}_log.csv".format(model_name))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./{0}_logs".format(model_name)) 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./models/checkpoints/{0}_checkpoint.h5".format(model_name), period = 1, save_weights_only=True)
    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./models/best_model/best_{0}_checkpoint.h5".format(model_name), save_best_only = True, save_weights_only=True)
    model_save_path = "./models/saved_model/"

    # Train model
    try:
        model.fit(train_gen, 
                epochs = no_of_epochs, 
                callbacks = [csv_callback, checkpoint_callback, best_model_checkpoint_callback], #, tensorboard_callback],
                validation_data = val_gen,
                validation_steps = val_steps, 
                steps_per_epoch = train_steps)
    
    except KeyboardInterrupt:
        print("\n[INFO] Train Interrupted")
        model.save_weights(model_save_path + "_interrupted.h5")
        sys.exit(2)

    except Exception as err:
        print("\n{CRITICAL}: Error, UnHandled Exception: ", err, "\n", traceback.print_exc())
        print("{CRITICAL}: Trying to save the model")
        model.save_weights(model_save_path + "_error.h5")
        del model
        sys.exit(2)
        
    # Model saving
    model.save_weights(filepath = model_save_path + "{0}_weights.h5".format(model_name))
    model.save(filepath = model_save_path + "{0}.h5".format(model_name))

    # Testing Results
    print("[+] Testing the model")
    loss, accuracy = model.evaluate(test_gen, steps = test_steps, verbose = 1)
    print("[+] Test Loss: ", loss)
    print("[+] Test Accuracy: ", accuracy)


def deterred_train_model(model_path:str, resume_epoch:int, loading_weights: bool, config: dict):
    """
    Function to resume a paused training model
    
    Arguments:
        model_weight_path {str} -- the weight path of the model to load and resume train
        resume_epoch {int} -- The epoch to resume
        loading_weights {bool} -- The flag to indicate if we are loading weights or model file itself
        config {dict} -- The configuration to train consisting of 
        {
            "no_of_epochs" : no of epochs to train
            "learning_rate" : The learning rate
            "batch_size" : The batch size to use
            "threads" : The no. of threads to use
            "train_data_path": The meta data for training
            "val_data_path": The meta data for validation
            "test_data_path": The meta data for testing
            "model_name" : The model name, NOTE: This will be used to save the model.
        }
    """
    # Get the configuration to train
    no_of_epochs = config["no_of_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    model_name = config["model_name"]
    no_gpus = config["gpus"]
    train_data_path = config["train_data_path"]
    val_data_path =  config["val_data_path"]
    test_data_path = config["test_data_path"]

    print("[INFO]: Using config: \n", config)
    # Set up environment
    folders = [
        "./models/checkpoints/", 
        "./output/csv_log/",
        "./models/best_model_{0}".format(resume_epoch),
        "./models/saved_model_{0}".format(resume_epoch),
        "./output/graphs/"
    ]
    os_utilities.make_directory(folders)

    # Get steps_per_epoch
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)
    test_data = pd.read_csv(test_data_path)
    train_steps = int(len(train_data) / batch_size)
    val_steps = int(len(val_data) / batch_size)
    test_steps = int(len(test_data) / batch_size)
    del train_data
    del val_data
    del test_data
    print("[INFO]:Train Steps per Epoch: ", train_steps)
    print("[INFO]:Train Steps per Epoch: ", val_steps)
    print("[INFO]:Train Steps per Epoch: ", test_steps)

    # Load the generators for the data
    train_gen = preprocessing_utils.load_tf_dataset_generator(dataset_meta_path = train_data_path, batch_size = batch_size, shuffle = True, shuffle_buffer = 1000)
    val_gen = preprocessing_utils.load_tf_dataset_generator(dataset_meta_path = val_data_path, batch_size = batch_size, shuffle = False, shuffle_buffer = 100)
    test_gen = preprocessing_utils.load_tf_dataset_generator(dataset_meta_path = test_data_path, batch_size = batch_size, shuffle = False, shuffle_buffer = 100)
    
    # instantiate the new model and start train
    if no_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()

        print("[INFO]: No. of GPU devices used: ", strategy.num_replicas_in_sync)
        if loading_weights:
            with strategy.scope():
                model = densenet_model.densenet121_xray_model()
                model.load_weights(model_path)
                losses = ["categorical_crossentropy" for i in range(14)]
                metrics = ["accuracy" for i in range(14)]
                model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), loss = losses, metrics = metrics)
    
    else:
        print("[INFO]: Not using Distribution strategy")
        if loading_weights:        
            model = densenet_model.densenet121_xray_model()
            model.load_weights(model_path)
            losses = ["categorical_crossentropy" for i in range(14)]
            metrics = ["accuracy" for i in range(14)]
            model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), loss = losses, metrics = metrics)
        
    
    # set callbacks the model
    csv_callback = tf.keras.callbacks.CSVLogger("./output/csv_log/{0}_log_{1}.csv".format(model_name, resume_epoch))
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./{0}_logs".format(model_name)) 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./models/checkpoints/{0}_checkpoint_{1}.h5".format(model_name, resume_epoch), period = 1, save_weights_only=True)
    best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./models/best_model/best_{0}_checkpoint_{1}.h5".format(model_name, resume_epoch), save_best_only = True, save_weights_only=True)
    model_save_path = "./models/saved_model_{0}/".format(resume_epoch)

    # Train model
    try:
        model.fit(train_gen, 
                epochs = no_of_epochs, 
                callbacks = [csv_callback, checkpoint_callback, best_model_checkpoint_callback], #, tensorboard_callback],
                validation_data = val_gen,
                validation_steps = val_steps, 
                steps_per_epoch = train_steps,
                initial_epoch = resume_epoch)
    
    except KeyboardInterrupt:
        print("\n[INFO] Train Interrupted")
        model.save_weights(model_save_path + "_interrupted.h5")
        sys.exit(2)

    except Exception as err:
        print("\n{CRITICAL}: Error, UnHandled Exception: ", err, "\n", traceback.print_exc())
        print("{CRITICAL}: Trying to save the model")
        model.save_weights(model_save_path + "_error.h5")
        del model
        sys.exit(2)
        
    # Model saving
    model.save_weights(filepath = model_save_path + "{0}_weights.h5".format(model_name))
    model.save(filepath = model_save_path + "{0}.h5".format(model_name))

    # Testing Results
    print("[+] Testing the model")
    loss, accuracy = model.evaluate(test_gen, steps = test_steps, verbose = 1)
    print("[+] Test Loss: ", loss)
    print("[+] Test Accuracy: ", accuracy)