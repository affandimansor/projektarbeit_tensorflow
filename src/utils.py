""" Utilities bzw. gemeinsam benutzte Funktionen """

# Importiere relevante Bibliotheken
import numpy as np
import sklearn.metrics as metrics
from config import LABELS

# Belibiges Ergebniss nach Fehlerkategorien anzeigen
def print_results_categories(results, sep=''):
    labels = ['Point defects', 'Hole point defects', 'Split defects']
    for i, result in enumerate(results):
        print(f"{LABELS[i]}: {sep}{result}")

# # --------------------------------
# # Confusion Matrix
# # --------------------------------
# # Jede Komponente der Trainingsdaten separat extrahieren
# X_train, y_train = ds_train_batch

# #  Vorhersage mit den Trainingsdaten treffen
# y_pred = model(X_train)

# #  Confusion Matrix berechnen
# cm = metrics.multilabel_confusion_matrix(y_train, np.round(y_pred))

# # Jede Confusion Matrix anzeigen
# print("\n***** Confusion Matrix Training *****")
# print_results_categories(cm, sep='\n')

# # Debugging
# #y_pred = model(X_train)
# cm = metrics.multilabel_confusion_matrix(y_train, np.round(y_pred), samplewise=True)

# def print_sample_info(i, sample_cm):
#     print(f"Sample {i}:\n{sample_cm}")
#     print("y_true: \n", y_train[i].numpy())
#     print("y_pred: \n", y_pred[i].numpy())

# cnt_true_pos = 0
# cnt_false_pos = 0
# cnt_true_neg = 0
# cnt_false_neg = 0

# for i, sample_cm in enumerate(cm):
#     # CM von Samples anzeigen, die als falsches Negativ vorhergesagt wurden
#     if ((sample_cm[1,0] > 0)):
#         #print_sample_info(i, sample_cm)
#         cnt_false_neg+=1

#     # CM von Samples anzeigen, die als falsches Positiv vorhergesagt wurden
#         if ((sample_cm[0,1] > 0)):
#             print_sample_info(i, sample_cm)
#             cnt_false_pos+=1

#     # CM von Samples anzeigen, die richtig als Positiv vorhergesagt wurden
#             if ((sample_cm[1,1] > 0)):
#                 #print_sample_info(i, sample_cm)
#                 cnt_true_pos+=1

# print("False negativ:", cnt_false_neg)
# print("False positiv:", cnt_false_pos)
# print("True positiv:", cnt_true_pos)