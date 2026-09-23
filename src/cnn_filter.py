# Importiere relevante Bibliotheken
import numpy as np
import keras
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------------
# Die Daten einlesen
# --------------------------------
# Das Merkmal einlesen
X = np.load("../Material/data.npy")

# Den gemittelten Pixelwert berechnen
mean = np.mean(X)

# Die Standardabweichung der Pixelwerte berechnen
std_dev = np.std(X)

# Die Pixelwerte standardisieren
X_std = (X - mean)/std_dev
# X_std = X

# Die Labels einlesen
y = np.load("../Material/labels.npy")

# Die NumPy-Arrays in Tensor umwandeln
t_X = tf.convert_to_tensor(tf.cast(X_std, tf.float32))
t_y = tf.convert_to_tensor(y)

# Konfigurationen
BUFFER_SIZE = len(X)
BATCH_SIZE = 15
NUM_EPOCHS = 20
NUM_FILTER_CONV_1 = 18
NUM_FILTER_CONV_2 = 10
NUM_HIDDEN  = 12

# --------------------------------
# X und y in einen Datensatz kombinieren
# --------------------------------
# Fuer Reproduzierbarkeit
tf.random.set_seed(1)

# Die beiden Tensoren zu einem Datensatz kombinieren
ds = tf.data.Dataset.from_tensor_slices((t_X, t_y))

# Die Datenmenge durchmischen
#! reshuffle_each_iteration=True, wenn alles abgestimmt hat.
#! Hier ist es zwecks Training auf False gesetzt, damit Debugging einfacher ist.
ds = ds.shuffle(buffer_size=BUFFER_SIZE,
                reshuffle_each_iteration=True)


# Verhaeltnis Trainings- zu Testdatenmenge in Prozent 80:20
n_train_valid = np.uint16(0.8 * len(X_std))

# Verhaeltnis Trainings- zu Validierungsdatenmenge in Prozent 80:20
n_train = np.uint16(0.8 * n_train_valid)

# Die ersten n_train-Datenpunkten in die Trainingsdatenmenge reinziehen
ds_train_valid = ds.take(n_train_valid)

# Die Trainingsdatenmenge weiter in Training und Valid aufteilen
ds_train = ds_train_valid.take(n_train).batch(BATCH_SIZE, drop_remainder=True)
ds_valid = ds_train_valid.skip(n_train).batch(BATCH_SIZE, drop_remainder=True)

# Die letzten 20-Prozent der gesamten Datenmenge in Testdatenmenge reinziehen
ds_test = ds.skip(n_train_valid)

# --------------------------------
# Ein CNN-Modell definieren
# --------------------------------
# Ein Modell durch die Klasse Sequential() instanzieren
model = tf.keras.Sequential(name='cnn_filter')

# Eine Eingabeschicht
model.add(tf.keras.Input(shape=(40, 40, 1), batch_size=BATCH_SIZE))

# Erste Faltungsschicht
model.add(tf.keras.layers.Conv2D(
    filters=NUM_FILTER_CONV_1,
    kernel_size=(3, 3),
    data_format='channels_last',
    activation='relu',
    name = 'conv_1'))

# Erste Max-Poolingsschicht
model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2,2),
    name='pool_1'))

# Zweite Faltungsschicht
model.add(tf.keras.layers.Conv2D(
    filters=NUM_FILTER_CONV_2,
    kernel_size=(3, 3),
    data_format='channels_last',
    activation='relu',
    name = 'conv_2'))

# Zweite Max-Poolingsschicht
model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2),
    name='pool_2'))

# Flatten Schicht um den Tensor aus Rang 3 in 2 umzuwandeln
model.add(tf.keras.layers.Flatten(name='flat'))

# Die verdeckte Schicht
model.add(tf.keras.layers.Dense(
    NUM_HIDDEN,
    activation='relu',
    name='hidden_1'
))

# Die Ausgabeschicht mit 3 Neuronen, jeweils eine Klassenbezeichnung
model.add(tf.keras.layers.Dense(
    3,
    activation='sigmoid',
    name='out'
))

# Den Adam Optimizer definieren
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Das Modell kompilieren
# BinaryCrossEntropy, da ein Bild mehreren Schadenkategorien zugeordnet werden kann.
# CategoricalCrossentropy, wenn ein Bild exklusiv einer Shadenskategorie gehoert.
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', 'precision'])

# Das Modellsummary anzeigen lassen
model.summary()

# Das Modell trainieren
history = model.fit(ds_train, epochs=NUM_EPOCHS,
          validation_data=ds_valid,
            #class_weight={0:1,1:1,2:1},
         shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)]
          )

# Das Modell mit Testdaten evaluieren
batch_test = next(iter(ds_test.batch(100)))
pred = model(batch_test[0])

# --------------------------------
# Confusion Matrix
# --------------------------------
cm = metrics.multilabel_confusion_matrix(batch_test[1], np.round(model.predict(batch_test[0])))
for i, label in enumerate(['Point defects', 'Hole point defects', 'Split defects']):
    print(f"Confusion matrix for {label}:")
    print(cm[i])