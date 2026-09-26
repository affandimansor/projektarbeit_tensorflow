""" Alle Hyperparameter und Pfade sind hier zentral aufgelistet """

# Modi fuer Training
TRAINING:   bool    = False
VERBOSE:    int     = 1

# Fehlerkategorien
LABELS:     str = ['Point defects', 'Hole point defects', 'Split defects']

# Analysiskonfigurationen
IMG_DIRPATH = "./images"
EXAMPLE_FILEPATH = IMG_DIRPATH + "/example_image.png"
CM_FILEPATH = IMG_DIRPATH + "/confusion_matrix.png"

# Konfiguration des neuronalen Netzwerks
class FilterConfig:
    SEED:                   int     = 15
    NUM_PIC:                int     = 10000
    PIC_RES:                int     = [36, 36]
    BATCH_SIZE:             int     = 20
    NUM_EPOCHS:             int     = 20
    NUM_FILTER_CONV_1:      int     = 24
    KERNEL_CONV_1:          int     = [3, 3]
    KERNEL_MAX_POOL_1:      int     = [2, 2]
    NUM_FILTER_CONV_2:      int     = 10
    KERNEL_CONV_2:          int     = [3, 3]
    KERNEL_MAX_POOL_2:      int     = [2, 2]
    NUM_HIDDEN:             int     = 10
    INPUT_SHAPE:            int     = [PIC_RES[0], PIC_RES[1], 1]
    LEARN_RATE:             float   = 1e-03
    CHECKPOINT_FILEPATH:    str     = './checkpoint.model.keras'
    CHECKPOINT_MONITOR:     str     = 'val_loss'
    CHECKPOINT_MODE:        str     = 'min'
