# Importiere relevante Bibliotheken
import numpy as np
import keras
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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

# Die Labels einlesen
y = np.load("../Material/labels.npy")

# Belibiges Ergebniss nach Fehlerkategorien anzeigen
def print_results_categories(results, sep=''):
    labels = ['Point defects', 'Hole point defects', 'Split defects']
    for i, result in enumerate(results):
        print(f"{labels[i]}: {sep}{result}")

# Klassenverteilung anzeigen
print("***** Class distribution overall *****")
print_results_categories(np.sum(y, axis=0))

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
ds = ds.shuffle(buffer_size=BUFFER_SIZE,
                reshuffle_each_iteration=True)


# Verhaeltnis Trainings- zu Testdatenmenge in Prozent 80:20
n_train_valid = np.uint16(0.8 * len(X_std))

# Verhaeltnis Trainings- zu Validierungsdatenmenge in Prozent 80:20
n_train = np.uint16(0.8 * n_train_valid)

# Die ersten n_train-Datenpunkten in die Trainingsdatenmenge reinziehen
ds_train_valid = ds.take(n_train_valid)

# Die Trainingsdatenmenge weiter in Training und Valid aufteilen
ds_train_orig = ds_train_valid.take(n_train)
ds_valid_orig = ds_train_valid.skip(n_train)

# MiniBatches aus den beiden Datenmengen erstellen‚
ds_train = ds_train_orig.batch(BATCH_SIZE, drop_remainder=True)
ds_valid = ds_valid_orig.batch(BATCH_SIZE, drop_remainder=True)

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
              metrics=['accuracy', 'precision', 'f1_score'])

# Das Modellsummary anzeigen lassen
model.summary()

# Das Modell trainieren
history = model.fit(ds_train, epochs=NUM_EPOCHS,
          validation_data=ds_valid,
            #class_weight={0:1,1:1,2:1},
         shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)]
          )

# Trainingsergebnis anzeigen
hist = history.history

print("\n***** Trainingsergebnis *****")
print("loss_training: ", hist['loss'][-1])
print("accuracy_training: ", hist['accuracy'][-1])
print("precision_training: ", hist['precision'][-1])
print("f1_score_training for")
print_results_categories(hist['f1_score'][-1])

# # --------------------------------
# # Das Modell mit Testdaten evaluieren
# # --------------------------------
# # Das Modell mit Testdaten evaluieren
# batch_test = next(iter(ds_test.batch(100)))
# result = model.evaluate(ds_test.batch(BATCH_SIZE),
#                         verbose=1,
#                         return_dict=True)
# for key in result:
#     print(f"{key}: {result[key]}")


# pred = model(batch_test[0])

# # --------------------------------
# # Confusion Matrix
# # --------------------------------
# Jede Komponente der Trainingsdaten separat extrahieren
X_train, y_train = next(iter(ds_train_orig.batch(batch_size=len(ds_train_orig))))

# Confusion Matrix berechnen
cm = metrics.multilabel_confusion_matrix(y_train, np.round(model.predict(X_train)))

# Jede Confusion Matrix anzeigen
print("\n***** Confusion Matrix Training *****")
print_results_categories(cm, sep='\n')

# # 3 Subplots, jeweils fuer eine Confusion Matrix
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Jede Matrix separat nebeneinander plotten
# for i, ax in enumerate(axes):
#     sns.heatmap(cm[i], 
#                 annot=True, 
#                 fmt='d', 
#                 cmap='YlGn', 
#                 ax=ax, 
#                 cbar=(i == 2), # Show colorbar only on the last plot
#                 xticklabels=["Pred Negative", "Pred Positive"],
#                 yticklabels=["True Negative", "True Positive"])
#                 #? TODO: F1-Score mit aufzeichnen
#     ax.set_title(labels[i])

# plt.suptitle("Confusion Matrix of each defect", fontsize=14, y=1.03)
# plt.tight_layout()
# plt.show()

# --------------------------------
# Auswertung (ohne Kategorie)
# --------------------------------
def label_with_error(y):
    return np.any(y >= 1, axis=-1).astype(np.uint8)

# Ein Batch mit der Laenge von ds_test erstellen
batch_test = next(iter(ds_test.batch(batch_size=len(ds_test))))

# Greife auf jeweilige Spalte in der ds_test zu
X_test, y_test = batch_test

# Arrays zum Zwischenspeichern der kategorienlosen y-Werte
y_true_test = np.zeros(len(ds_test), dtype=np.uint8)
y_pred_test = np.zeros(len(ds_test), dtype=np.uint8)

# y_true in kategorienlos umwandeln
for i, y in enumerate(y_test):
    y_true_test[i] = label_with_error(y)

# Vohersage mit den Testdaten treffen
y_pred = model(X_test)

# y_pred in kategorienlos umwandeln
for i, y in enumerate(y_pred):
    y_pred_test[i] = label_with_error(np.round(y))

# Die Ergebnisse auf der Terminal ausgeben
print("\n***** Auswertung ohne Kategorie *****")
print(y_true_test, "n_labels_true: ", len(y_true_test), "n_defects_true: ", np.count_nonzero(y_true_test))
print(y_pred_test, "n_labels_pred: ", len(y_pred_test), "n_defects_pred: ", np.count_nonzero(y_pred_test))
print(f"  Accuracy:  {metrics.accuracy_score(y_true_test, y_pred_test):.3f}")
print(f"  Precision:  {metrics.precision_score(y_true_test, y_pred_test):.3f}")
print(f"  F1-Score:  {metrics.f1_score(y_true_test, y_pred_test):.3f}")

# --------------------------------
# Confusion Matrix
# --------------------------------
cm = metrics.multilabel_confusion_matrix(batch_test[1], np.round(model.predict(batch_test[0])))
for i, label in enumerate(['Point defects', 'Hole point defects', 'Split defects']):
    print(f"Confusion matrix for {label}:")
    print(cm[i])