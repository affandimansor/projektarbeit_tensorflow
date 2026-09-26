# Importiere relevante Bibliotheken
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from utils import print_results_categories
from config import LABELS, EXAMPLE_FILEPATH, CM_FILEPATH

# --------------------------------
# Confusion Matrix
# --------------------------------
def confusion_matrix(y_train, y_pred, multilabel=True):
    xticklabels=["Pred Negative", "Pred Positive"]
    yticklabels=["True Negative", "True Positive"]

    # Auswertung nach Fehlerkategorien
    if multilabel:
        # Confusion Matrix berechnen
        cm = metrics.multilabel_confusion_matrix(y_train, np.round(y_pred))
        
        # Jede Confusion Matrix anzeigen
        print("\n***** Confusion Matrix Training *****")
        print_results_categories(cm, sep='\n')

        # 3 Subplots, jeweils fuer eine Confusion Matrix
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Jede Matrix separat nebeneinander plotten
        for i, ax in enumerate(axes):
            sns.heatmap(cm[i], 
                        annot=True, 
                        fmt='d', 
                        cmap='YlGn', 
                        ax=ax, 
                        # Zeigen colorbar nur bei letzter Fehlerkategorie
                        cbar=(i == 2),
                        xticklabels=xticklabels,
                        yticklabels=yticklabels)
                        #? TODO: F1-Score mit aufzeichnen
            ax.set_title(LABELS[i])
        plt.suptitle("Confusion Matrix nach Fehlerkategorien", fontsize=14)

    # Auswertung ohne Fehlerkategorien
    else:
        # Confusion Matrix berechnen
        cm = metrics.confusion_matrix(y_train, y_pred, normalize='true')
        sns.heatmap(cm, 
                    annot=True, 
                    cmap='YlGn', 
                    cbar=True,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels)
        plt.title("Confusion Matrix ohne Fehlerkategorien")
    plt.tight_layout()
    plt.savefig(CM_FILEPATH)
    plt.show()


# Ein Beispiel als defekt markieren, falls eine der Schadenarten positiv bzw. 1 gesetzt ist
def _label_with_error(y):
    return np.any(y >= 1, axis=-1).astype(np.uint8)

# Anlaysieren ohne Kategorisierung
def analysis_uncategorized_defects(y_true, y_pred, ds=''):

    # Arrays zum Zwischenspeichern der kategorienlosen y-Werte
    y_true_uncat = np.zeros(len(y_true), dtype=np.uint8)
    y_pred_uncat = np.zeros(len(y_pred), dtype=np.uint8)

    # y_true in kategorienlos umwandeln
    for i, y in enumerate(y_true):
        y_true_uncat[i] = _label_with_error(y)

    # y_pred in kategorienlos umwandeln
    for i, y in enumerate(y_pred):
        y_pred_uncat[i] = _label_with_error(np.round(y))

    # Die Metriken berechnen
    accuracy = metrics.accuracy_score(y_true_uncat, y_pred_uncat)
    precision = metrics.precision_score(y_true_uncat, y_pred_uncat)
    f1_score = metrics.f1_score(y_true_uncat, y_pred_uncat)

    # Die Ergebnisse auf der Terminal ausgeben
    print(f"\n***** Auswertung ohne Kategorie {ds}*****")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision:  {precision:.3f}")
    print(f"  F1-Score:  {f1_score:.3f}")

    return y_true_uncat, y_pred_uncat, accuracy, precision, f1_score

# --------------------------------
# Ein Bild aus dem Datensatz darstellen
# --------------------------------
def visualize_image(img, title):
    # Das Bild wird auf Diagramm mit der Achsenskala [0,1] geplottet 
    plt.imshow(img, extent=(0,1,0,1), interpolation='nearest')
    plt.title(title)
    plt.savefig(EXAMPLE_FILEPATH)
    plt.show()

# --------------------------------
# Balkendiagramm zum Vergleich der Metriken
# --------------------------------
def barchart_metrics(values, deltas, datasets=['train', 'test'], metrics=['acccuracy', 'precision', 'f1_score']):
    barWidth = 0.25
    fig = plt.subplots(figsize=(15, 8))

    br1 = np.arange(len(metrics))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, values[0][:], color = 'b', width=barWidth,
            label=datasets[0])
    plt.bar(br2, values[1][:], color = 'g', width=barWidth,
            label=datasets[1])

    #? TODO: Add annotation to each bar

    plt.xlabel("Metrics", fontweight="bold", fontsize=15)
    plt.ylabel("Values", fontweight="bold", fontsize=15)
    plt.xticks([r + barWidth/2 for r in br1], metrics) #range(len(metrics))], metrics)

    plt.legend()
    plt.show()

    