"""pipeline.

This pipeline loads the data, extract features, trains the model and predicts
The pipeline has the following stages:

results are saved to ``results`` directory

"""
import argparse
import glob
import os
import sys

import numpy as np
from argparse import ArgumentParser
from argparse import Namespace

import audioread
import librosa

from typing import List
from typing import Tuple
from typing import Union

from flask import Flask, request


import tensorflow as tf
from keras import models
from sklearn.metrics import accuracy_score, roc_auc_score

import random
from model_generator.model_list import simple_nn
from utilities.feature_extract import extract_features
from utilities.preprocessing import split_data, preprocess_single_audio


# tf.compat.v1.disable_eager_execution()


class Pipeline:
    # path for audio files directory:
    DATA_DIR = 'data/cats_dogs/'
    MODEL_PATH = 'results/model_keras.h5'

    def reset_random_seeds():
        '''
        This function is to set seed so that we get reproducible results
        '''
        os.environ['PYTHONHASHSEED'] = str(1)
        tf.random.set_seed(1)
        np.random.seed(1)
        random.seed(1)

    # make some random data
    reset_random_seeds()
    def load_data(self, data_path: str, sr: Union[int, None] = None) -> Tuple[list, int, list, list, int, list]:
        '''
        This function loads data from the audio files and returns a list of audio files, sampling rate of each file,
        number of channels for dogs and cats
        :param data_path : Its the path to audio files
        :param sr: sampling rate
        :return : list of audio files, sampling rate, number of channels for dogs and cats

        '''

        cats_wave_list = []
        cats_sr = []
        channel_cats = []
        dogs_wave_list = []
        channel_dogs = []
        dogs_sr = []

        for cat_file in glob.glob(data_path + '/cat*.wav'):
            cat_i, sr_i = librosa.load(cat_file, sr=sr)
            cats_wave_list.append(cat_i)
            cats_sr.append(sr_i)
            with audioread.audio_open(cat_file) as input_file:
                channel_cats.append(input_file.channels)

        for dog_file in glob.glob(data_path + '/dog*.wav'):
            dog_j, sr_j = librosa.load(dog_file, sr=sr)
            dogs_wave_list.append(dog_j)
            dogs_sr.append(sr_j)
            with audioread.audio_open(dog_file) as input_file:
                channel_dogs.append(input_file.channels)

        if len(set(cats_sr)) !=1 or len(set(dogs_sr)) != 1:
            raise Exception("current pipeline works for all files having same sampling rate")
        if set(cats_sr) != set(dogs_sr):
            raise Exception("current pipeline requires cats sampling rate to be equal to dogs sampling rate")


        return cats_wave_list, list(set(cats_sr))[0], channel_cats, dogs_wave_list, list(set(dogs_sr))[0], channel_dogs

    def pipeline(self,
                 arguments: List[str]) -> None:

        parser: ArgumentParser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("-T", "--train", help="train models", action="store_true")
        parser.add_argument("-P", "--predict", help="generate predictions", action="store_true")
        parser.add_argument("-D", "--deploy", help="deploy as mlflow serving", action="store_true")

        args: Namespace = parser.parse_args(args=arguments)

        model = None
        try:
            if args.train:
                # load data
                print(" Loading data .....")
                cats_wave_list, cats_sr, _, dogs_wave_list, _, _ = self.load_data(self.DATA_DIR)
                X_train, X_test, Y_train, Y_test = split_data(cats_wave_list, dogs_wave_list, test_size_ratio=0.1)
                X_train_features = extract_features(X_train, cats_sr)
                X_test_features = extract_features(X_test, cats_sr)
                print(f"shape of training input: {len(X_train)}")
                print(f"shape of test data: {len(X_test)}")
                model = simple_nn(X_train_features, Y_train, self.MODEL_PATH)

                pred = [(model.predict(data.reshape(1, 41, ))[0][0] > 0.5).astype("int32") for data in X_test_features]
                print(" Test accuracy :", accuracy_score(Y_test, pred))
                print(" Test accuracy roc_auc :", roc_auc_score(Y_test, pred))

            if args.predict:
                if not model:
                    print(" loading model for prediction ")
                    model = models.load_model(self.MODEL_PATH)

                app = Flask(__name__)


                @app.route('/')
                def welcome():
                    return "Check if the audio file is a dog or a cat"

                @app.route('/predict', methods=["POST"])
                def predict():
                    test_data_path = request.files.get("audio file")
                    feature = preprocess_single_audio(test_data_path)
                    pred = model.predict(feature.reshape(1, 41, ))
                    return "Model predicted as Cat: {} Dog: {}".format(pred[0][0], pred[0][1])

                # remove host when running pipeline without docker
                # port = int(os.environ.get('PORT', 8888))
                # app.run(host='172.17.0.2', port=port)
                app.run()
        except BaseException as e:
            raise e


if __name__ == '__main__':
    pipeline: Pipeline = Pipeline()
    pipeline.pipeline(sys.argv[1:])
