import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import numpy as np


NB_EPOCHS_MAX = 50

def redefine_model(input_dims):

    model = tf.keras.Sequential([
    tf.keras.Input(shape=(input_dims,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics='accuracy', loss=tf.keras.losses.BinaryCrossentropy())

    return model


def train_model(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    callbacks_val_loss = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=3,
        verbose=1,
    )
    ]

    callbacks_train_loss = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=3,
        verbose=1,
    )
    ]

    input_dims = np.shape(X_train)[1]

    model = redefine_model(input_dims)


    model.fit(x=X_train, y=Y_train, epochs=NB_EPOCHS_MAX, batch_size=16, validation_data=(X_test, Y_test), callbacks=callbacks_val_loss)

    y_preds = np.round(model.predict(X_train),0).flatten()

    y_preds_test = np.round(model.predict(X_test),0).flatten()

    train_accuracy = accuracy_score(Y_train, y_preds)

    test_accuracy = accuracy_score(Y_test, y_preds_test)



    model = redefine_model(input_dims)

    model.fit(x=X, y=Y, epochs=NB_EPOCHS_MAX, batch_size=16, callbacks=callbacks_train_loss)

    y_preds_tot = np.round(model.predict(X),0).flatten()

    total_accuracy = accuracy_score(Y, y_preds_tot)

    return model, train_accuracy, test_accuracy, total_accuracy