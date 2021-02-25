from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt


def simple_nn(X_train_features, X_test_features, y_train, y_test):
    # Convert the labels to match what our model will expect
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(41,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()

    # Choose the parameters to train the neural network
    best_model_weights = './base.model'
    checkpoint = ModelCheckpoint(
        best_model_weights,
        monitor='val_acc',
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
        validation_data=(X_test_features, test_labels),
        epochs=200,
        verbose=1,
        callbacks=callbacks,
    )

    # Save the model
    model.save_weights('results/model_wieghts.h5')
    model.save('results/model_keras.h5')

    # list all data in history
    print(history.history.keys())

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'b', label = "training accuracy")
    plt.plot(epochs, val_acc, 'r', label = "validation accuracy")
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.show()

    plt.savefig('results/learning_curve.png')



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