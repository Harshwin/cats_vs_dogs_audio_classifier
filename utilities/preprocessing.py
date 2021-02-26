import glob
import librosa
from sklearn.model_selection import train_test_split
from utilities.feature_extract import extract_features


def create_label(cats_list:list, dogs_list:list) -> [list,[list]]:
    '''
    This function generates labels for cats(0) and dogs(1)
    :param cats_list: list of cat audio data files
    :param dogs_list: list of dog audio data files
    :return: list of labels for dogs and cats
    '''
    cats_label = 0
    dogs_label = 1
    cat_y = [cats_label]*len(cats_list)
    dog_y = [dogs_label]*len(dogs_list)

    return cat_y, dog_y

def split_data(cats_list: list, dogs_list: list, test_size_ratio: float )-> [list, list, list, list]:
    '''

    :param cats_list: list of cat audio data
    :param dogs_list: list of dog audio data
    :param test_size_ratio: percentage of test size required
    :return: training and test input and labels
    '''
    cat_y, dog_y = create_label(cats_list, dogs_list)
    X = cats_list + dogs_list
    Y = cat_y + dog_y
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size_ratio, stratify=Y)

    return X_train, X_test, y_train, y_test

def preprocess(data_path:str)-> list:
    '''
    given a data_path of audio files, this function reads and converts into a feature feedable to model
    :param data_path: path to audio files
    :return: features
    '''
    wave_list = []
    sr = []
    for file in glob.glob(data_path + '/*.wav'):
        audio, sr_current = librosa.load(file, sr=None)
        wave_list.append(audio)
        sr.append(sr_current)
    data_features = extract_features(wave_list, sr[0])
    return data_features

def preprocess_single_audio(data_path:str)-> list:
    '''
    audio file path, this function reads and converts into a feature feedable to model
    :param data_path: path to audio files
    :return: features
    '''

    audio, sr = librosa.load(data_path, sr=None)
    data_features = extract_features(audio, sr)
    return data_features

