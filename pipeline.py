"""pipeline.

This pipeline loads the data, extract features, trains the model and predicts
The pipeline has the following stages:


Intermediate results are saved to ``tmp`` and final results
are saved to ``target``

"""
import argparse
import glob
import os
import sys
from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime, timedelta
import traceback

# import dill

from pathlib import Path

import audioread
import librosa
import pandas as pd
from pandas import Series
from pandas import DataFrame

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional



import tensorflow as tf
from tensorflow.keras.models import load_model

from utilities.feature_extract import extract_features
from utilities.preprocessing import split_data

tf.compat.v1.disable_eager_execution()


class Pipeline:
    # path for audio files directory:
    DATA_DIR = 'data/cats_dogs/'

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
        parser.add_argument("-p", "-predict", help="generate predictions", action="store_true")
        args: Namespace = parser.parse_args(args=arguments)
        try:
            if args.train:
                # load data
                cats_wave_list, cats_sr, _, dogs_wave_list, _, _ = self.load_data(self.DATA_DIR)
                X_train, X_test, Y_train, Y_test = split_data(cats_wave_list, dogs_wave_list, test_size_ratio=0.3)
                X_train_features = extract_features(X_train, cats_sr)
                X_test_features = extract_features(X_test, cats_sr)
                


            if args.predict:
                pass

        except BaseException as e:
            raise e


if __name__ == '__main__':
    pipeline: Pipeline = Pipeline()
    pipeline.pipeline(sys.argv[1:])
