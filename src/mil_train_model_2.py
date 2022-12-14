'''
Run script to train MIL_RC model.
Edit SCRIPT PARAMETERS section for hyper-parameter tuning, data loading, etc.
'''

import sys
sys.path.insert(0, './data')
import torch
import pickle
import time
from claim_data import process_mil_claim_data
from claim_data import load_mil_pickled_data
from claim_models import MIL_RC
from claim_utils import mil_train_model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# SCRIPT PARAMETERS:

# Data
PROCESS_DATA_FROM_CSV = False
PATH_X = "data/x_train.csv"
PATH_Y = "data/y_train.csv"
K = 1
LOAD_DATA_FROM_PICKLE = True
DATA_PICKLE_PATH='data/datasets_train_val.pkl'

# Fixed
VOCAB_SIZE = 13326

# Tune
EMBED_DIM = 150
LEARNING_RATE = 0.000001
WEIGHT_DECAY = 0.05
POOLING_MODE = "max"
EPOCHS = 5

# Checkpoint
CHECKPOINT_PATH = None # Indicate path as: 'results/models/20210522-173208_mil_rc_checkpoint.pth'


# RUN SCRIPT --------------------------------------------------------------------------------------
stamp = time.strftime("%Y%m%d-%H%M%S")
print('TIMESTAMP:', stamp)
print('DEVICE:', DEVICE)

# LOAD DATA

if PROCESS_DATA_FROM_CSV:
  process_mil_claim_data(PATH_X, PATH_Y, K, LOAD_DATA_FROM_PICKLE, DATA_PICKLE_PATH)

data_path = "data/datasets_train_val.pkl"

x, y, x_valid, y_valid = load_mil_pickled_data(data_path)

# INITIALIZE MODEL
print('Initializing model...')

input_size = x[0].shape[1]-19+18*EMBED_DIM

mil = MIL_RC(input_size=input_size, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, pooling_mode=POOLING_MODE, bias=True).to(DEVICE)
optimizer = torch.optim.Adam(mil.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

if CHECKPOINT_PATH is None:
    print('Model initialized!')
    start_epoch = 0
else:
    print('Loading checkpoint...')
    checkpoint = torch.load(CHECKPOINT_PATH)
    mil.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

# TRAIN MODEL
mil, metrics = mil_train_model(mil, optimizer, x, y, x_valid, y_valid, EPOCHS=EPOCHS, START_EPOCH=start_epoch, checkpoint_path="results/models/{}_mil_rc_checkpoint.pth".format(stamp))

# SAVE RESULTS
print('Saving results...')

# Save Parameters
text_log = open("results/metrics/{}_TRAIN_params_mil_rc.txt".format(stamp), "w")
text_log.write("EPOCHS: {} \nLR: {} \nWEIGHT_DECAY: {} \nPOOLING: {} \nEMBED_DIM: {}".format(EPOCHS, LEARNING_RATE, WEIGHT_DECAY, POOLING_MODE, EMBED_DIM))
text_log.close()

# Save Model
torch.save(mil, "results/models/{}_mil_rc.pkl".format(stamp))

# Save Metrics
pickle.dump(metrics, open("results/metrics/{}_TRAIN_mil_rc.pkl".format(stamp), "wb"))

print('END OF SCRIPT')
