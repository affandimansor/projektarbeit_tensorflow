
""" Datenvorbereitung """

# Importiere relevante Bibliotheken
import numpy as np
import keras
import tensorflow as tf
from config import FilterConfig as config
from utils import print_results_categories

# --------------------------------
# Die Daten einlesen und in Tensor umwandeln
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

# # Die Labels einlesen
y = np.load("../Material/labels.npy")

# Klassenverteilung anzeigen
print("***** Class distribution overall *****")
print_results_categories(np.sum(y, axis=0))

# Die NumPy-Arrays in Tensor umwandeln
t_X = tf.convert_to_tensor(tf.cast(X_std, tf.float32))
t_y = tf.convert_to_tensor(y)

# --------------------------------
# X und y in einen Datensatz kombinieren und ihn aufteilen
# --------------------------------
def create_datasets():
    # Fuer Reproduzierbarkeit
    tf.random.set_seed(config.SEED)

    # Die beiden Tensoren zu einem Datensatz kombinieren
    ds = tf.data.Dataset.from_tensor_slices((t_X, t_y))

    # Die Datenmenge durchmischen
    ds = ds.shuffle(buffer_size=config.NUM_PIC, seed=1,
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

    # Ein Batch mit der Laenge von n_train erstellen, um auf jeweilige Spalte in der ds_train_orig zuzugreifen
    ds_train_orig_batch = next(iter(ds_train_orig.batch(batch_size=n_train)))

    # Klassenverteilung in der Trainingsdatenmenge
    print("\n***** Class distribution ds_train_orig *****")
    print_results_categories(tf.reduce_sum(ds_train_orig_batch[1], axis=0))

    # MiniBatches aus den beiden Datenmengen erstellen‚
    ds_train_batch = ds_train_orig.batch(config.BATCH_SIZE, drop_remainder=True)
    ds_valid_batch = ds_valid_orig.batch(config.BATCH_SIZE, drop_remainder=True)

    # Die letzten 20-Prozent der gesamten Datenmenge in Testdatenmenge reinziehen
    ds_test = ds.skip(n_train_valid)

    # Ein Batch mit der Laenge von ds_test erstellen, um auf jeweilige Spalte in der ds_test zuzugreifen
    ds_test_batch = next(iter(ds_test.batch(batch_size=len(ds_test))))

    return ds_train_orig_batch, ds_train_batch, ds_valid_batch, ds_test_batch