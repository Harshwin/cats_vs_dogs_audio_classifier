
import librosa
import numpy as np


# function to extract all the features needed for the classification
def extract_features(audio_samples, sample_rate):
    print(" Extracting features ..... ")
    extracted_features = np.empty((0, 41,))
    if not isinstance(audio_samples, list):
        audio_samples = [audio_samples]

    for sample in audio_samples:
        # calculate the zero-crossing feature
        zero_cross_feat = librosa.feature.zero_crossing_rate(sample).mean()

        # calculate the mfccs features
        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

        # add zero crossing feature to the feature list
        mfccsscaled = np.append(mfccsscaled, zero_cross_feat)
        mfccsscaled = mfccsscaled.reshape(1, 41, )

        extracted_features = np.vstack((extracted_features, mfccsscaled))

    # return the extracted features
    return extracted_features
