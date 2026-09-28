# Importiere relevante Bibliotheken
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

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

# Beliebiges Ergebniss nach Fehlerkategorien anzeigen
def print_results_categories(results, sep=''):
    labels = ['Point defects', 'Hole point defects', 'Split defects']
    for i, result in enumerate(results):
        print(f"{labels[i]}: {sep}{result}")

# Klassenverteilung anzeigen
print("***** Class distribution overall *****")
print_results_categories(np.sum(y, axis=0))

# Die NumPy-Arrays in Tensor umwandeln
t_X = tf.convert_to_tensor(tf.cast(X_std, tf.float32), name="pixel")
t_y = tf.convert_to_tensor(y, name="label")

# Konfigurationen
BUFFER_SIZE = len(X)
BATCH_SIZE = 15
NUM_EPOCHS = 15
NUM_FILTER_CONV_1 = 16
NUM_FILTER_CONV_2 = 10
NUM_HIDDEN  = 10

# --------------------------------
# X und y in einen Datensatz kombinieren
# --------------------------------
# Fuer Reproduzierbarkeit
tf.random.set_seed(1)

# Die beiden Tensoren zu einem Datensatz kombinieren
ds = tf.data.Dataset.from_tensor_slices((t_X, t_y))

# Die Datenmenge durchmischen
ds = ds.shuffle(buffer_size=BUFFER_SIZE,
                reshuffle_each_iteration=False)

# Verhaeltnis Trainings- zu Testdatenmenge in Prozent 80:20
n_train_valid = np.uint16(0.8 * len(X_std))

# Verhaeltnis Trainings- zu Validierungsdatenmenge in Prozent 80:20
n_train = np.uint16(0.8 * n_train_valid)

# Die ersten n_train-Datenpunkten in die Trainingsdatenmenge reinziehen
ds_train_valid = ds.take(n_train_valid)

# Die Trainingsdatenmenge weiter in Training und Valid aufteilen
ds_train_orig = ds_train_valid.take(n_train)
ds_valid_orig = ds_train_valid.skip(n_train)

# Ein Batch mit der Laenge von n_train erstellen, um auf jeweilige Spalte in der ds_train_orig zuzugreifen
ds_train_orig_batch = next(iter(ds_train_orig.batch(batch_size=n_train)))

# Klassenverteilung in der Trainingsdatenmenge
print("\n***** Class distribution ds_training *****")
print_results_categories(tf.reduce_sum(ds_train_orig_batch[1], axis=0))

# MiniBatches aus den beiden Datenmengen erstellen‚
ds_train = ds_train_orig.batch(BATCH_SIZE, drop_remainder=True)
ds_valid = ds_valid_orig.batch(BATCH_SIZE, drop_remainder=True)

# Die letzten 20-Prozent der gesamten Datenmenge in Testdatenmenge reinziehen
ds_test = ds.skip(n_train_valid)

# --------------------------------
# Ein CNN-Modell definieren
# --------------------------------
def cnn_filter(hidden_units, filters):
    # Ein Modell durch die Klasse Sequential() instanzieren
    model = tf.keras.Sequential(name='cnn_filter')

    # Eine Eingabeschicht
    model.add(tf.keras.Input(shape=(40, 40, 1), batch_size=BATCH_SIZE))

    # Erste Faltungsschicht
    model.add(tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        name = 'conv_1'))

    # Erste Max-Poolingsschicht
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2,2),
        name='pool_1'))

    # Zweite Faltungsschicht
    model.add(tf.keras.layers.Conv2D(
        filters=filters,
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
        units=hidden_units,
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

    return model

params={
    'filters':[16, 10, 12, 20],
    'hidden_units':[16, 10, 32, 24]
}

model = KerasClassifier(model=build_filter, filters=12, hidden_units=10)

gs = GridSearchCV(estimator=model, param_grid=params, cv=10)
print(gs.estimator.get_params().keys())
# gs = gs.fit(X, y)


# print("Parameter: ", gs.best_params_)
# print("Accuracy: ", gs.best_score_)
# # Das Modellsummary anzeigen lassen
# # model.summary()

# # ModellCheckpoint einrichten
# # Dateipfad zum Abspeichern des Modells
# cp_filepath = './checkpoint.model.keras'

# # Die callback Funktion ModelCheckpoint
# # Die Metrik precision ist als das Entscheidungskriterium ausgewaehlt, da positive
# # und negative Daten ungleich viel sind. 
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=cp_filepath,
#     monitor='precision',
#     mode='max',
#     save_best_only=True
# )

# # # Das Modell trainieren
# # history = model.fit(ds_train, epochs=NUM_EPOCHS,
# #           validation_data=ds_valid,
# #          #class_weight={0:1.2, 1:1, 2:1.6},
# #         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), 
# #                    model_checkpoint_callback],
# #         verbose=0
# #         )

# # # Trainingsergebnis anzeigen
# # hist = history.history

# # print("\n***** Trainingsergebnis *****")
# # print("loss_training: ", hist['loss'][-1])
# # print("accuracy_training: ", hist['accuracy'][-1])
# # print("precision_training: ", hist['precision'][-1])
# # print("f1_score_training for")
# # print_results_categories(hist['f1_score'][-1])

# # Das abgespeicherte Modell reinladen
# model = tf.keras.models.load_model(cp_filepath)

# # Das absgespeicherte Modell mit Trainingsdaten evaluieren
# results = model.evaluate(ds_train,
#                          return_dict=True)
# print(model.summary())
# # Das Ergebnis der Evaluierung anzeigen
# print("\n***** Auswertung nach Kategorien mit Trainingsdaten *****")
# print("loss_training: ", results['loss'])
# print("accuracy_training: ", results['accuracy'])
# print("precision_training: ", results['precision'])
# print_results_categories(results['f1_score'])

# # # --------------------------------
# # # Confusion Matrix
# # # --------------------------------
# # Jede Komponente der Trainingsdaten separat extrahieren
# X_train, y_train = ds_train_batch

# # Vorhersage mit den Trainingsdaten treffen
# y_pred = model(X_train)

# # Confusion Matrix berechnen
# cm = metrics.multilabel_confusion_matrix(y_train, np.round(y_pred))

# # Jede Confusion Matrix anzeigen
# print("\n***** Confusion Matrix Training *****")
# print_results_categories(cm, sep='\n')
