""" Alle Hyperparameter und Pfade sind hier zentral aufgelistet """

class FilterConfig:
    SEED:                   str     = 1
    NUM_PIC:                int     = 10000
    PIC_RES:                int     = [37, 37]
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
