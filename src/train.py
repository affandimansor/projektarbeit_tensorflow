""" Einen CNN-Filter trainieren """

# Importiere relevante Bibliotheken
import tensorflow as tf
import numpy as np
from config import FilterConfig as config
from config import TRAINING, VERBOSE
from utils import print_results_categories
from dataset import create_datasets
from model import cnn_filter
from analysis import confusion_matrix, analysis_uncategorized_defects, visualize_image, barchart_metrics

# ------------------------------
# Die Datensaetze reinladen
# ------------------------------
ds_train_orig, ds_train, ds_valid, ds_test = create_datasets()

# ------------------------------
# Das Modell initialisieren und trainieren
# ------------------------------
if TRAINING:
    # Ein Modell initialisieren
    model = cnn_filter()

    # Die Modellzusammenfassung anzeigen
    model.summary()

    # Ein ModellCheckpoint einrichten
    # Die Metrik val_loss ist als das Entscheidungskriterium ausgewaehlt, da die Minimierung des Verlusts
    # das Ziel vom Training ist. 
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.CHECKPOINT_FILEPATH,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        save_best_only=True)

    # Das Modell trainieren
    # EarlyStopping, um das Overfitting an Trainingsdaten zu verhindern
    history = model.fit(ds_train, epochs=config.NUM_EPOCHS,
                        validation_data=ds_valid,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), 
                        model_checkpoint_callback],
                        verbose=VERBOSE)

    # Trainingsergebnis anzeigen
    hist = history.history

    print("\n***** Trainingsergebnis *****")
    print("Picture resolution: ", config.PIC_RES)
    print("loss_training: ", hist['loss'][-1])
    print("accuracy_training: ", hist['accuracy'][-1])
    print("precision_training: ", hist['precision'][-1])
    print("f1_score_training for")
    print_results_categories(hist['f1_score'][-1])

# ------------------------------
# Das abgespeicherte Modell gegenchecken
# ------------------------------
print("\n***** Auswertung des abgespeicherten Modells mit Trainingsdaten (nach Kategorien) *****")

# Das abgespeicherte Modell reinladen
model = tf.keras.models.load_model(config.CHECKPOINT_FILEPATH)

# Das absgespeicherte Modell mit Trainingsdaten evaluieren
results = model.evaluate(ds_train,
                         return_dict=True)
print(model.summary())

# Das Ergebnis der Evaluierung anzeigen
print("\n***** Auswertung nach Kategorien mit Trainingsdaten *****")
print("loss_training: ", results['loss'])
print("accuracy_training: ", results['accuracy'])
print("precision_training: ", results['precision'])
print("f1_score_training for")
print_results_categories(results['f1_score'])

# --------------------------------
# Auswertung (ohne Kategorie)
# --------------------------------
# Jede Komponente der Trainingsdaten separat extrahieren
X_train, y_train = ds_train_orig

# Vorhersage mit den Trainingsdaten treffen
y_pred = model(X_train)

# Ergebnisse aus dem Vorhersagen mit Trainingsdaten auswerten und anzeigen
y_true_train, y_pred_train, y_train_acc, y_train_prec, y_train_f1  = analysis_uncategorized_defects(y_train, y_pred, ds="ds_train ")

# Greife auf jeweilige Spalte in der ds_test zu
X_test, y_test = ds_test

# Vohersage mit den Testdaten treffen
y_pred = model(X_test)

# Ergebnisse auswerten und anzeigen
y_true_test, y_pred_test, y_test_acc, y_test_prec, y_test_f1 = analysis_uncategorized_defects(y_test, y_pred, ds="ds_test ")

# Confusion Matrix anzeigen
confusion_matrix(y_true_test, y_pred_test, multilabel=False)

# Die Metriken zw. ds_test und ds_train miteinander vegleichen
d_acc = y_test_acc - y_train_acc
d_prec = y_test_prec - y_train_prec
d_f1 = y_test_f1 - y_train_f1

metrics = [[y_train_acc, y_train_prec, y_train_f1],
           [y_test_acc, y_test_prec, y_test_f1]]

# Balkendiagramm anzeigen
barchart_metrics(values=metrics, deltas=d_acc)

# --------------------------------
# Darstellung eines Beispielsdiagramms
# --------------------------------
# Zeige das erste Bild im ds_test mit Fehler an
visualize_image(X_test[np.argmax(y_true_test > 0)], "Defect example")