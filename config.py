SEED = 1234

PARTIAL_TRAIN = 0.5
TEST_SIZE = 100000
NUM_LABELS = 15

MAX_SEQUENCE_LENGTH = 200

NUM_EPOCH = 1
LEARNING_RATE = 2e-5 
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2

INPUT_DIR = "../input/"
WORK_DIR = "../working/"

BERT_MODEL_NAME = 'bert-base-uncased'
BERT_MODEL_PATH = INPUT_DIR + 'uncased_L-12_H-768_A-12/'

TOXICITY_COLUMN = 'target'

DATA_DIR = INPUT_DIR + "jigsaw-unintended-bias-in-toxicity-classification"

FINE_TUNED_MODEL_PATH = INPUT_DIR + "fine-tuned-model/bert_pytorch.bin"