from models import MLPModel, RNNModel, LSTMModel, BERTModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import keyboard
import random

import matplotlib.pyplot as plt
import bertviz


# Setting a manual seed for reproducability (useful for ablations)
torch.manual_seed(67)

def seed_worker(worker_id): # Manual seeding init function for dataloaders
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)


# Making sure datasets exist of variable lengths (4,6,8,10 for the purpose of ablation, instead of just the original 10 long datasets)
if len(os.listdir('.\\Datasets\\'))==1:
    raw_data = pd.read_csv('.\\Datasets\\10_ranking_dataset.csv').to_numpy()
    seqs, _= raw_data[:,:10], raw_data[:,10:]
    for seq_len in range(4,10,2):
        new_seqs = seqs[:,:seq_len]
        new_ranks = new_seqs.argsort().argsort()
        new_data = np.concat((new_seqs, new_ranks), axis=-1)
        np.savetxt(f".\\Datasets\\{seq_len}_ranking_dataset.csv", new_data, delimiter=',', fmt='%d')



# One hot encoding
def expand_label(lab):
    n = len(lab)
    l=[]
    for i in range(n):
        temp = np.zeros((n,1))
        temp[int(lab[i])] = 1
        l.extend(temp)
    exp_labs = np.array(l, dtype=float).flatten()
    return exp_labs

# Splitting inputs from rank labels
def X_y_split(arr, seq_len):
    return arr[:,:seq_len], arr[:,seq_len:]


# Creating the Dataset class
class RankingDataset(Dataset):
    def __init__(self, X_train, y_train, device, enableEmbeds):
        super().__init__()
        self.nums = torch.tensor(X_train).to(device=device, dtype=torch.long if enableEmbeds else torch.float) # Numerical values
        self.labs = torch.tensor(y_train).to(device=device, dtype=torch.long) # Rank values

    def __len__(self):
        return len(self.nums)
    
    def __getitem__(self, idx):
        return (self.nums[idx], self.labs[idx])


# Training function
def train(model, tr_dataloader, val_dataloader, epochs, patience=10, optim=None): 
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=l_rate) # Stochastic Gradient Descent
    if not optim:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Adaptive Moment Estimation
    else:
        optimizer=optim
    losses = []
    val_losses = []
    best_loss = float('inf') # A large dummy amount, just for initialisation
    pat_count = 0
    epoch=0
    while epoch<epochs: # Gradient descent over epochs
        cancel = False

        total_loss = 0
        val_loss = 0

        model.train()
        i=0
        for nums, labels in tr_dataloader: # Retrieving features for training
            if keyboard.is_pressed('alt+c') and epochs>1:   # A training "cancel" function if epochs seem unfavourable
                cancel=True
                break

            pred = model.forward(nums) 

            # When seq_len=10 for example, in all the models, we have 100 output nodes - 10 at a time representing rank probabilities for
            # a particular position. We use view to convert this (batch,100) array to a (batch,10,10) array
            # so that the loss is computed, accounting for the dependence of nodes representing same position
            pred = pred.view(-1, model.seq_len, model.seq_len) # <--- LLM used for this line
            labels = labels.long()  # Long used since we are representing sparse categories
            loss = loss_fn(pred, labels)
            
            total_loss += loss.item()

            loss.backward() # Beginning backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        total_loss/=i+1e-9

        model.eval()
        i=0
        for nums, labels in val_dataloader:
            if (keyboard.is_pressed('alt+c') and epochs>1):   # A training "cancel" function if epochs seem unfavourable
                cancel=True
                break
            elif cancel==True:
                break
            
            pred = model.forward(nums)
            pred = pred[0].view(-1, seq_len, seq_len) if isinstance(pred, tuple) else pred.view(-1, seq_len, seq_len) 
            labels = labels.long()   
            loss = loss_fn(pred, labels)

            val_loss += loss.item()
            optimizer.zero_grad()
            i+=1
        val_loss/=i+1e-9

        if cancel:
            break

        if val_losses!=[] and val_loss-val_losses[-1]>1e-4: # Patience mechanism
            if pat_count==patience:
                break
            else:
                pat_count+=1
        else:
            pat_count=0

        losses.append(total_loss)
        val_losses.append(val_loss)

        if val_loss<best_loss:
            best_loss = val_loss
            dest = f'.\\New Models\\Ortho_{seq_len}_{model.tag}_model.pth' if not enableNormalisation else f'.\\New Models\\norm{enableNormalisation}_{seq_len}_{model.tag}_model.pth'
            # New Models destination to avoid over-writing any of the existing trained models...
            torch.save(model.state_dict(), dest)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return epoch, losses, val_losses


# Certain hyper-parameters. Ensure that these are set the same as in the inference notebook, for testing the correct model.
# Changing seq_len, selected_model will make train.py and the inference notebook to choose a particular filename.
# Advised to confirm the directories before training/inference.

seq_len=6 # Permissible seq_lens : 4, 6, 8, 10
vocab_size=1000
emb_dim = 5 #5 # set to None if you dont want embeddings to be used
               # Note. BERT model implementation has not been implemented with non-embedded inputs, please assign dmodel normally and keep emb_dim!=None!
batch_size=512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enableNormalisation = 0 # 0 for none, 1 for z-norm, 2 for global min-max norm, 3 for sequence-wise min-max norm

if enableNormalisation:
    emb_dim = None

# Main code

#Loading Data
dat = pd.read_csv(f".\\Datasets\\{seq_len}_ranking_dataset.csv", dtype=int).to_numpy() # [10000 rows x 20 columns]

# Train-Validation-Test split (80-10-10)
dat_dummy, dat_test = train_test_split(dat, test_size=0.1, random_state=42, shuffle=True)
dat_train, dat_val = train_test_split(dat_dummy, test_size=0.1/0.9, random_state=42, shuffle=True)

# Number - Rank splits
X_train, y_train = X_y_split(dat_train, seq_len)
X_val, y_val = X_y_split(dat_val, seq_len)

# Normalisations
if enableNormalisation==1: # (Z-normalisation)
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) 
    X_train = (X_train-X_mean)/(X_std+1e-8)   
    X_val = (X_val-X_mean)/(X_std +1e-8)
elif enableNormalisation==2: # (Global Min-Max normalisation)
    X_max, X_min = X_train.max(axis=0), X_train.min(axis=0) 
    X_train = (X_train-X_min)/(X_max-X_max+1e-8)
    X_val = (X_val-X_min)/(X_max-X_max+1e-8)     
elif enableNormalisation==3: # (Sequence-wise Min-Max normalisation)
    X_train = (X_train-X_train.min(axis=-1, keepdims=True))/(X_train.max(axis=-1, keepdims=True)-X_train.min(axis=-1, keepdims=True)+1e-8)
    X_val = (X_val-X_val.min(axis=-1, keepdims=True))/(X_val.max(axis=-1, keepdims=True)-X_val.min(axis=-1, keepdims=True)+1e-8)



training_data_loader = DataLoader(dataset=RankingDataset(X_train, y_train, device, enableEmbeds=emb_dim!=None),
                                  batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
validation_data_loader = DataLoader(dataset=RankingDataset(X_val, y_val, device, enableEmbeds=emb_dim!=None),
                                    batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker)
print("Starting! (Loaders initiated)\nInstancing and training model...")


model_map = {
    "mlp":  MLPModel(vocab_size, seq_len, emb_dim, device=device),
    "rnn":  RNNModel(vocab_size, seq_len, emb_dim, num_layers=1, device=device),
    "lstm": LSTMModel(vocab_size, seq_len, emb_dim, num_layers=1, bidirectional=True, device=device),
    "bert": BERTModel(N=3, h=2, dmodel=10, dk=5, dv=5, vocab_size=vocab_size, seq_len=seq_len, device=device), 
}                                                                                           


selected_model = "bert"  # Change this to the model you desire to train

model = model_map[selected_model]

# Training the model
print("Initiating model training, press 'ALT+C' to interrupt...")
epochs, loss_history, val_loss_history = train(model, training_data_loader, validation_data_loader, epochs=350, patience=10,
                                       optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3,betas=(0.9,0.99)))
print("\nTraining complete!")

# # Recording training loss / validation loss graph
plt.plot(np.arange(1, epochs+1),loss_history, color='b', label='training loss')
plt.plot(np.arange(1, epochs+1),val_loss_history, color='r', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss History')
plt.legend()
plt.savefig(f'.\\Ablation data\\Ablations B\\Ortho_{seq_len}_{model.tag}_loss.png')
    