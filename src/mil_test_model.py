import pickle
import time
import torch
from claim_utils import valid_loop

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL = "sample_mil.pkl"
MODEL_PATH = 'results/models/{}'.format(MODEL)

# RUN ---------------------------------------------------------------------------------------------

# Import Model
model = torch.load(MODEL_PATH)

# Import Data
test_data = pickle.load(open("data/dataset_test.pkl", "rb"))

x = []
y = []

for i, tup in enumerate(test_data['test']):
    x.append(tup[0])
    y.append(tup[1])

# Prediction
mean_loss, accuracy, TPR, FPR, con_mat = valid_loop(x, y, model, DEVICE)

metrics = {
        'test': {
            'losses': mean_loss,
            'accuracies': accuracy,
            'TPRs': TPR,
            'FPRs': FPR,
            'con_mats': con_mat
        }
    }

# SAVE RESULTS

print('Saving results...')
stamp = time.strftime("%Y%m%d-%H%M%S")

name = MODEL.split('_', 1)[0]
rest = MODEL.split('_', 1)[1]

# Save Metrics
pickle.dump(metrics, open("results/metrics/{}_TEST_{}".format(name, rest), "wb"))

print('Testing done!')
