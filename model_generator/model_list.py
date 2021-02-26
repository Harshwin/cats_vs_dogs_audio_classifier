import os

from keras import layers
from keras import models
from keras import losses
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt

from model_generator.utils import delete_file_if_exists
from model_generator.wrapping import ModelWrapper
from utilities.preprocessing import preprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def simple_nn(X_train_features, y_train, model_path):
    # Convert the labels to match what our model will expect
    train_labels = to_categorical(y_train)
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(41,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()

    # Choose the parameters to train the neural network
    best_model_weights = './base.model'
    checkpoint = ModelCheckpoint(
        best_model_weights,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=False,
        period=1
    )

    callbacks = [checkpoint]

    model.compile(optimizer='adam',
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(
        X_train_features,
        train_labels,
        validation_split=0.2,
        epochs=200,
        verbose=1,
        callbacks=callbacks,
    )

    # Save the model
    print(f"saving models to {model_path}")
    model.save_weights(model_path)
    model.save('results/model_keras.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc)+1)

    print("plotting learning curve")
    plt.plot(epochs, acc, 'b', label = "training accuracy")
    plt.plot(epochs, val_acc, 'r', label = "validation accuracy")
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('results/learning_curve.png')
    plt.show()

    print("packagine the model to combine preprocessing and predict for mlflow serving")
    ## Package the model
    # Location in our gdrive where we want the model to be saved
    model_path = f"results/model_package"
    delete_file_if_exists(model_path)

    data_path = 'data/cats_dogs'
    # # Package the model!
    preprocess_model = ModelWrapper(model=model, preprocess=preprocess)
    # Todo: getting error in below line - TypeError: can't pickle tensorflow.python._tf_stack.StackSummary objects
    # mlflow.pyfunc.save_model(path=model_path,
    #                          python_model=preprocess_model)

    # Todo: TypeError: can't pickle _thread.RLock objects
    # with open('results/preprocess_model.pickle', 'wb') as f:
    #     pickle.dump(preprocess_model, f)
    return model



# def tuner_mlp_network() -> ModelSpec:
#     # Model Generating Function
#     def baseline_model(hp, optimizer='adam', lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=True, reg=0.1,
#                        tol=1e-7) -> Sequential:
#         """
#         This function builds a Sequential nueral network model with Tensorflow 2.0 and Keras Backend
#         :param input_nodes: number of features in training set
#         :param optimizer: the optimizer being used for in the sequential neural network
#         :param lr: the learning rate of the optimizer
#         :param beta_1: the learning rate decay
#         :param beta_2: the second learning rate decay
#         :param amsgrad: Whether or not to use the amsgrad Adam optimizer modification
#         :param reg: the magnitude of L1 regularization
#         :param tol: the tolerance for the stopping criteria
#         :return: sequential Tensorflow 2.0 model
#         """
#         input_nodes = 16
#         # Generate Nodes Per Layer
#         # num_hidden_layers = 3
#         num_hidden_layers = hp.Int(
#             'num_layers',
#             min_value=3,
#             max_value=6,
#             step=1)
#         # current options for method: gpr, ''
#         nodes = generate_node_levels(num_hidden_layers + 2, input_nodes, method='')
#
#         # Build Sequential Model
#         model = Sequential()
#         # To note that kernel_initializer=glorot_uniform by default,
#         model.add(Dense(nodes[1], input_shape=(input_nodes,), kernel_regularizer=l1(l=reg)))
#         model.add(Activation(hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'],default='relu')))
#         # model.add(BatchNormalization())
#         for x in range(1, num_hidden_layers + 1):
#             model.add(Dense(nodes[x], kernel_regularizer=l1(l=reg)))
#             model.add(Activation(hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'],default='relu')))
#             # model.add(BatchNormalization())
#         model.add(Dense(1))  # Output Layer
#         model.add(Activation('relu'))
#         # model.add(BatchNormalization())
#         if optimizer == 'adam':
#             optimizer = Adam(lr=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=lr),
#                              beta_1=hp.Choice('beta_1', values=[0.7, 0.8, 0.9], default=beta_1), beta_2=beta_2, epsilon=tol, amsgrad=amsgrad)
#         model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
#         return model