import torch
from torch import nn
from torch.nn import functional as F
from claim_utils import pool


class MIL(nn.Module):
    
    def __init__(self, input_size, vocab_size, embed_dim, dropout_pct=0.5, pooling_mode="max", bias=True):
        super(MIL, self).__init__()

        self.pooling_mode = pooling_mode
        self.input_size = input_size
        self.embed_dim = embed_dim
        
        # layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_pct)
        
        # pooling
        self.kernel = nn.Parameter(torch.randn((64,1)), requires_grad=True)
        self.kernel_bias = nn.Parameter(torch.tensor(0.), requires_grad=True) if bias else False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = len(x)
        embedding = self.embed(x[:,-19:-1])
        embedding = embedding.view(batch_size, 18*self.embed_dim)
        features = x[:,:-19]
        x = torch.cat((features, embedding), dim=1)
        
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = x.matmul(self.kernel)
        if self.kernel_bias:
            x = x + self.kernel_bias
        x = self.sigmoid(x)

        if self.pooling_mode=="LSE":
            out = torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))[0]
            return out
        elif self.pooling_mode=="mean":
            out = torch.mean(x, dim=0, keepdim=True)[0]
            return out
        else:
            out = torch.max(x, dim=0, keepdim=True)[0]
            return out


class MIL_RC(nn.Module):

    def __init__(self, input_size, vocab_size, embed_dim, dropout_pct=0.5, pooling_mode="max", bias=True):
        super(MIL_RC, self).__init__()

        self.input_size = input_size
        self.pooling_mode = pooling_mode
        self.bias = bias
        self.embed_dim = embed_dim

        # fully connected layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        # dropout layers
        self.dropout1 = nn.Dropout(dropout_pct)
        self.dropout2 = nn.Dropout(dropout_pct)
        self.dropout3 = nn.Dropout(dropout_pct)

        # output layer
        self.out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # embedding the diag and proc codes
        batch_size = len(x)
        embedding = self.embed(x[:,-19:-1])
        embedding = embedding.view(batch_size, 18*self.embed_dim)
        features = x[:,:-19]
        x = torch.cat((features, embedding), dim=1)

        # model
        x = self.dropout1(F.relu(self.fc1(x.float())))
        rc1 = pool(x)
        x = self.dropout2(F.relu(self.fc2(x)))
        rc2 = pool(x)
        x = self.dropout3(F.relu(self.fc3(x)))
        rc3 = pool(x)

        # residuals summation
        rc_sum = rc1 + rc2 + rc3

        # ouput layer
        out = self.sigmoid(self.out(rc_sum))

        return out
