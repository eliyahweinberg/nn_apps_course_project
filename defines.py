X = 0
Y = 1

WEIGHT_PASS = 'PretrainedWeights/'
TRAIN_INPUT_DIR = 'Data/finished/train/dataraw/hires'
TEST_INPUT_DIR = 'Data/finished/valid/dataraw/hires'
SAVED_SET_PATH = 'Data/finished/set/'
UNET_WEIGHTS = 'unet_weights.h5'
ZSSR_WEIGHTS = 'zssr_weights.h5'

TEST_IMAGES_NUM = 3
TEST_IMAGE_NAME = 'test_{}.png'
TEST_IMAGE_PATH = 'TestImages'

# scaling factor
SR_FACTOR = 2
# Activation layer
ACTIVATION = 'relu'
# Data generator random ordered
SHUFFLE = False
# scaling factors array order random or sorted
SORT = True
# Ascending or Descending: 'A' or 'D'
SORT_ORDER = 'A'
# number of time steps (pairs) per epoch
NB_STEPS = 1
# Batch size
BATCH_SIZE = 1
# Number of channels in signal
NB_CHANNELS = 3
# No. of NN filters per layer
FILTERS = 64  # 64 on the paper
# Number of internal convolutional layers
LAYERS_NUM = 6
# No. of scaling steps. 6 is best value from paper.
NB_SCALING_STEPS = 1
# No. of LR_HR pairs
EPOCHS = NB_PAIRS = 1500
# Default crop size (in the paper: 128*128*3)
CROP_SIZE = [96]#[32,64,96,128]
# Momentum # default is 0.9 # 0.86 seems to give lowest loss *tested from 0.85-0.95
BETA1 = 0.90  # 0.86
# Adaptive learning rate
INITIAL_LRATE = 0.001
DROP = 0.5
# Adaptive lrate, Number of learning rate steps (as in paper)
FIVE = 5
# Decide if learning rate should drop in cyclic periods.
LEARNING_RATE_CYCLES = False
#
# EPOCHS_DROP = np.ceil((NB_STEPS * EPOCHS ) / NB_SCALING_STEPS)
# Plot super resolution image when using zssr.predict
PLOT_FLAG = False
# Crop image for training
CROP_FLAG = True
# Flip flag
FLIP_FLAG = True
# initial scaling bias (org to fathers)
SCALING_BIAS = 1
# Scaling factors - blurring parameters
BLUR_LOW = 0.4
BLUR_HIGH = 0.95
# Add noise or not to transformations
NOISE_FLAG = False
# Mean pixel noise added to lr sons
NOISY_PIXELS_STD = 30
# Save augmentations
SAVE_AUG = True
# If there's a ground truth image. Add to parse.
GROUND_TRUTH = True
# If there's a baseline image. Add to parse.
BASELINE = True
# png compression ratio: best quality
CV_IMWRITE_PNG_COMPRESSION = 9