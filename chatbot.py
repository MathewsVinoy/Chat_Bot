import torch
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
from pathlib import Path
import numpy as np
import random
import json
from model import ChatBot


stemmer = LancasterStemmer()

with open("data\intents.json") as file:
    data = json.load(file)

RES_SAVE_PATH = Path("resources\data.pickle")
MODEL_PATH = Path("models\model.pth")

try:
    with open(RES_SAVE_PATH, "rb") as f:
        words, labels, training, output = pickle.load(f)
        print(f"Loaded data from pickle file. Vocabulary size: {len(words)}")
except Exception as e:
    print(f"This the error geting when loading the data: -->{e}")

def bag_of_words(s, words):
    bag = torch.zeros(len(words))

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    indices = [i for i, w in enumerate(words) if w in s_words]
    bag[indices] = 1

    out = torch.Tensor(bag)

    return torch.unsqueeze(out, 0)
out =bag_of_words(s="hai", words=words)
print(out.shape)

model = ChatBot(input_size=len(words), hidden_size=8, output_size=38)
model.load_state_dict(torch.load(f=MODEL_PATH))

def chat():
    print("Start chatting with the bot \n")
    while True:
        inp = input("You: ")
        if inp.lower() == "q":
            break
        user_input=bag_of_words(inp, words)
        model.eval()
        with torch.inference_mode():
            result = model(user_input)
        result_index = torch.argmax(result)
        test=result[0,result_index.item()]
        print(test)
        tag = labels[result_index]

        for intent in data["intents"]:
            if intent['tag'] == tag:
                responses = intent['responses']

        if test.item() > 9.0:
            print(random.choice(responses))
            print(test.item())
        else:
            print("Sorry")

chat()