# Importiere relevante Bibliotheken
import numpy as np
import keras
import tensorflow as tf
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from config import FilterConfig as config
# --------------------------------
# Die Daten einlesen
# --------------------------------
# Das Merkmal einlesen
X = np.load("../Material/data.npy")

# Die Bildaufloesung bzw. resolution anpassen
if config.PIC_RES != X.shape[1:3]:
    X = X[:, :config.PIC_RES[0], :config.PIC_RES[1], :]

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
NUM_HIDDEN  = 10
INPUT_SHAPE = t_X.shape[1:]

# --------------------------------
# X und y in einen Datensatz kombinieren
# --------------------------------
# Fuer Reproduzierbarkeit
tf.random.set_seed(config.SEED)
keras.utils.set_random_seed(config.SEED)
tf.config.experimental.enable_op_determinism()

# Die beiden Tensoren zu einem Datensatz kombinieren
ds = tf.data.Dataset.from_tensor_slices((t_X, t_y))

# Die Datenmenge durchmischen
ds = ds.shuffle(buffer_size=config.NUM_PIC, seed=1,
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
ds_train_batch = next(iter(ds_train_orig.batch(batch_size=n_train)))

# Klassenverteilung in der Trainingsdatenmenge
print("\n***** Class distribution ds_training *****")
print_results_categories(tf.reduce_sum(ds_train_batch[1], axis=0))

# MiniBatches aus den beiden Datenmengen erstellen‚
ds_train = ds_train_orig.batch(config.BATCH_SIZE, drop_remainder=True)
ds_valid = ds_valid_orig.batch(config.BATCH_SIZE, drop_remainder=True)

# Die letzten 20-Prozent der gesamten Datenmenge in Testdatenmenge reinziehen
ds_test = ds.skip(n_train_valid)

# --------------------------------
# Ein CNN-Modell definieren
# --------------------------------
# Ein Modell durch die Klasse Sequential() instanzieren
model = tf.keras.Sequential(name='cnn_filter')

# Eine Eingabeschicht
model.add(tf.keras.Input(shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE))

# Erste Faltungsschicht
model.add(tf.keras.layers.Conv2D(
    filters=config.NUM_FILTER_CONV_1,
    kernel_size=config.KERNEL_CONV_1,
    padding='same',
    data_format='channels_last',
    activation='relu',
    name = 'conv_1'))

# Erste Max-Poolingsschicht
model.add(tf.keras.layers.MaxPool2D(
    pool_size=config.KERNEL_MAX_POOL_1,
    name='pool_1'))

# Zweite Faltungsschicht
model.add(tf.keras.layers.Conv2D(
    filters=config.NUM_FILTER_CONV_2,
    kernel_size=config.KERNEL_CONV_2,
    data_format='channels_last',
    activation='relu',
    name = 'conv_2'))

# Zweite Max-Poolingsschicht
model.add(tf.keras.layers.MaxPool2D(
    pool_size=config.KERNEL_MAX_POOL_2,
    name='pool_2'))

# Flatten Schicht um den Tensor aus Rang 3 in 2 umzuwandeln
model.add(tf.keras.layers.Flatten(name='flat'))

# Die verdeckte Schicht
model.add(tf.keras.layers.Dense(
    config.NUM_HIDDEN,
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
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARN_RATE)

# Das Modell kompilieren
# BinaryCrossEntropy, da ein Bild mehreren Schadenkategorien zugeordnet werden kann.
# CategoricalCrossentropy, wenn ein Bild exklusiv einer Shadenskategorie gehoert.
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', 'precision', 'f1_score'])

# Das Modellsummary anzeigen lassen

# ModellCheckpoint einrichten
# Dateipfad zum Abspeichern des Modells
cp_filepath = './checkpoint.model.keras'

# Die callback Funktion ModelCheckpoint
# Die Metrik precision ist als das Entscheidungskriterium ausgewaehlt, da positive
# und negative Daten ungleich viel sind. 
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=cp_filepath,
    monitor='precision',
    mode='max',
    save_best_only=True
)

# Das Modell trainieren
history = model.fit(ds_train, epochs=config.NUM_EPOCHS,
          validation_data=ds_valid,
         #class_weight={0:1.2, 1:1, 2:1.6},
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3), 
                   model_checkpoint_callback],
        verbose=0
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
X_train, y_train = ds_train_batch

# # Vorhersage mit den Trainingsdaten treffen
y_pred = model(X_train)

# # Confusion Matrix berechnen
cm = metrics.multilabel_confusion_matrix(y_train, np.round(y_pred))

# Jede Confusion Matrix anzeigen
print("\n***** Confusion Matrix Training *****")
print_results_categories(cm, sep='\n')

# Debugging
#y_pred = model(X_train)
cm = metrics.multilabel_confusion_matrix(y_train, np.round(y_pred), samplewise=True)

def print_sample_info(i, sample_cm):
    print(f"Sample {i}:\n{sample_cm}")
    print("y_true: \n", y_train[i].numpy())
    print("y_pred: \n", y_pred[i].numpy())

cnt_true_pos = 0
cnt_false_pos = 0
cnt_true_neg = 0
cnt_false_neg = 0

for i, sample_cm in enumerate(cm):
    # CM von Samples anzeigen, die als falsches Negativ vorhergesagt wurden
    if ((sample_cm[1,0] > 0)):
        #print_sample_info(i, sample_cm)
        cnt_false_neg+=1

    # CM von Samples anzeigen, die als falsches Positiv vorhergesagt wurden
        if ((sample_cm[0,1] > 0)):
            print_sample_info(i, sample_cm)
            cnt_false_pos+=1

    # CM von Samples anzeigen, die richtig als Positiv vorhergesagt wurden
            if ((sample_cm[1,1] > 0)):
                #print_sample_info(i, sample_cm)
                cnt_true_pos+=1

print("False negativ:", cnt_false_neg)
print("False positiv:", cnt_false_pos)
print("True positiv:", cnt_true_pos)