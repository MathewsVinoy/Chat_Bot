from turtle import mode
import torch
from torch import nn
import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
from pathlib import Path
import json
from torch.utils.data import DataLoader
from lib.datasetsChat import DataSetsChat
from sklearn.model_selection import train_test_split
import pickle

stemmer = LancasterStemmer()

with open("data\intents.json") as file:
    data = json.load(file)


PATHS_RES = Path("resources")
RES_NAME = "data.pickle"
RES_SAVE_PATH = PATHS_RES / RES_NAME
MODEL_PATH = Path("models")
MODEL_NAME = "model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
BATCH=8
EPOCHS =25

try:
    with open(RES_SAVE_PATH, "rb") as f:
        words, labels, training, output = pickle.load(f)
        print(f"Loaded data from pickle file. Vocabulary size: {len(words)}")

except Exception:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w not in ["!","?",",","."]]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    
    PATHS_RES.mkdir(parents=True, exist_ok=True)
    
    with open(RES_SAVE_PATH, "wb") as f:
        pickle.dump((words, labels, training, output), f)
        print(f"Saved data to pickle file. Vocabulary size: {len(words)}")

data = torch.tensor(training, dtype=torch.float32)
labels = torch.tensor(output, dtype=torch.float32)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.1)

train_DataSets = DataSetsChat(x=x_train, y=y_train)
test_DataSets = DataSetsChat(x=x_test, y=y_test)

train_dataloader = DataLoader(train_DataSets,batch_size=BATCH,shuffle=True,num_workers=0)
test_dataloader = DataLoader(test_DataSets,batch_size=BATCH,shuffle=False,num_workers=0)


from model import ChatBot

model = ChatBot(input_size=len(words), hidden_size=8, output_size=38)



from tqdm.auto import tqdm

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_count =[]
loss_value =[]
test_loss_values = []


for epochs in tqdm(range(EPOCHS)):
    model.train()
    for (x, y)in train_dataloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    #testing 
    model.eval()
    with torch.inference_mode():#inference mode means predict there is no_grad()
        for (x, y)in train_dataloader:
            test_pred = model(x)
            test_loss = loss_fn(test_pred, y)
    
    print(f"Epoch: {epochs} | test: {loss} | test loss: {test_loss}")


MODEL_PATH.mkdir(parents=True, exist_ok=True)
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)