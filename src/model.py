# Importiere relevante Bibliotheken
import tensorflow as tf
import keras
from config import FilterConfig as config

# --------------------------------
# Ein CNN-Modell definieren
# --------------------------------
def cnn_filter():
    # Fuer Reproduzierbarkeit
    keras.utils.set_random_seed(config.SEED)
    tf.config.experimental.enable_op_determinism()

    # Ein Modell durch die Klasse Sequential() instanzieren
    model = tf.keras.Sequential(name='cnn_filter')

    # Eine Eingabeschicht
    model.add(tf.keras.Input(shape=config.INPUT_SHAPE, batch_size=config.BATCH_SIZE))

    # Erste Faltungsschicht
    # Same-Padding, die Information an Randen zu beruecksichtigen
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
    return model