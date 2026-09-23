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

# Transformiere y zu 1 oder 0, entsprechend ob einer der Einträge vorher 1 war, also das Bild einen Fehler enthalten hat.
# y = np.any(y >= 1, axis=-1).astype(int)

# Kontrolle
# print(X_std.shape)
# print(y.shape)
# print(np.min(X_std))
# print(np.max(X_std))

# Die NumPy-Arrays in Tensor umwandeln
t_X = tf.convert_to_tensor(tf.cast(X_std, tf.float32))
t_y = tf.convert_to_tensor(y)

# t_y_reshaped = np.expand_dims(t_y, axis=1)

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
# ds = tf.data.Dataset.from_tensor_slices((t_X, t_y_reshaped))

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

# Kontrolle
# print(ds)
batch = next((iter(ds)))
# print(batch[0].numpy(), tf.reshape(batch[1], shape=(-1,3)).shape)
# print(tf.reshape(batch[1], shape=(-1,3)).shape)

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

for i, batch in enumerate(ds_train):
    print(f"Batch {i}: {tf.reduce_sum(batch[1], axis=0).numpy()}")

# --------------------------------
# Ein CNN-Modell definieren
# --------------------------------
# Ein Modell durch die Klasse Sequential() instanzieren
model = tf.keras.Sequential(name='cnn_filter')

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

merkmalskarten_pooling_shape = model.compute_output_shape(input_shape=(BATCH_SIZE, 40, 40, 1))
# print(merkmalskarten_pooling_shape)

# Die verdeckte Schicht
model.add(tf.keras.layers.Dense(
    NUM_HIDDEN,
    activation='relu',
    name='hidden_1'
))

# Eine Dropout Schicht mit 50-Prozent Aktivierung der Neuronen
# model.add(tf.keras.layers.Dropout(0.2, name='dropout'))

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

#model.build(input_shape=(None, 40, 40, 1))


# Ein Modellsummary anzeigen lassen
model.summary()

# # Das Modell trainieren
history = model.fit(ds_train, epochs=NUM_EPOCHS,
          validation_data=ds_valid,
            #class_weight={0:1,1:1,2:1},
         shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)]
          )


# Das Modell mit Testdaten evaluieren
batch_test = next(iter(ds_test.batch(100)))
# print(batch_test[1])
pred = model(batch_test[0])

# Klassenverteilung anzeigen
print(tf.reduce_sum(batch_test[1], axis=0))

# --------------------------------
# Die Metriken auf einem Diagramm anzeigen
# --------------------------------
# hist = history.history

# fig = plt.figure(figsize=(12,15))
# ax = fig.add_subplot(1,3,1)
# ax.plot(hist['loss'], lw=3)
# ax.set_title('Training loss', size=15)
# ax.set_xlabel('Epoch', size=15)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax = fig.add_subplot(1,3,2)
# ax.plot(hist['accuracy'], lw=3)
# ax.set_title('Training accuracy', size=15)
# ax.set_xlabel('Epoch', size=15)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax = fig.add_subplot(1,3,3)
# ax.plot(hist['precision'], lw=3)
# ax.set_title('Training precision', size=15)
# ax.set_xlabel('Epoch', size=15)
# ax.tick_params(axis='both', which='major', labelsize=15)
# plt.show()

# --------------------------------
# Confusion Matrix
# --------------------------------
cm = metrics.multilabel_confusion_matrix(batch_test[1], np.round(model.predict(batch_test[0])))
for i, label in enumerate(['Point defects', 'Hole point defects', 'Split defects']):
    print(f"Confusion matrix for {label}:")
    print(cm[i])