""" Alle Hyperparameter und Pfade sind hier zentral aufgelistet """

# Gibt an, ob ein Modell erneut trainiert werden muss. Sonst wird nur das bestehende
# Modell bewertet.
TRAINING:   bool = False

# Fehlerkategorien
LABELS:     str = ['Point defects', 'Hole point defects', 'Split defects']

# Konfiguration des neuronalen Netzwerks
class FilterConfig:
    SEED:                   int     = 10
    NUM_PIC:                int     = 10000
    PIC_RES:                int     = [38, 38]
    BATCH_SIZE:             int     = 15
    NUM_EPOCHS:             int     = 15
    NUM_FILTER_CONV_1:      int     = 16
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
