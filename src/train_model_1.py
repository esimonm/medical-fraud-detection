import sys
sys.path.insert(0, './data')
import torch
import pickle
import time
from claim_data import load_claim_data
from claim_models import MIL
from claim_utils import load_data
from claim_utils import train_model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Data
LOAD_DATA = False
PATH_X = "data/x_train.csv"
PATH_Y = "data/y_train.csv"
K = 1
LOAD_DATA_FROM_PICKLE = True
DATA_PICKLE_PATH='data/datasets_train_val.pkl'

# Fixed
VOCAB_SIZE = 13326

# Tune
EMBED_DIM = 100
LEARNING_RATE = 1e-05
WEIGHT_DECAY = 0.05
POOLING_MODE = "max"
EPOCHS = 30

# RUN SCRIPT --------------------------------------------------------------------------------------
stamp = time.strftime("%Y%m%d-%H%M%S")
print('TIMESTAMP:', stamp)
print('DEVICE:', DEVICE)

# LOAD DATA

if LOAD_DATA:
  load_claim_data(PATH_X, PATH_Y, K, LOAD_DATA_FROM_PICKLE, DATA_PICKLE_PATH)

data_path = "data/datasets_train_val.pkl"

X, y, X_valid, y_valid = load_data(data_path)

# LOAD & TRAIN MODEL

print('Initializing model...')

input_size = X[0].shape[1]-19+18*EMBED_DIM

mil = MIL(input_size=input_size, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, dropout_pct=0.5, pooling_mode=POOLING_MODE, bias=True).to(DEVICE)
optimizer = torch.optim.Adam(mil.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print('Model initialized!')

mil, metrics = train_model(mil, optimizer, X, y, X_valid, y_valid, EPOCHS=EPOCHS)

# SAVE RESULTS
print('Saving results...')

# Save Parameters
text_log = open("results/metrics/{}_TRAIN_params.txt".format(stamp), "w")
text_log.write("EPOCHS: {} \nLR: {} \nWEIGHT_DECAY: {} \nPOOLING: {} \nEMBED_DIM: {}".format(EPOCHS, LEARNING_RATE, WEIGHT_DECAY, POOLING_MODE, EMBED_DIM))
text_log.close()

# Save Model
torch.save(mil, "results/models/{}_mil.pkl".format(stamp))

# Save Metrics
pickle.dump(metrics, open("results/metrics/{}_TRAIN_mil.pkl".format(stamp), "wb"))

print('END OF SCRIPT')
