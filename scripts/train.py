import json
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data():
    with open('./data/preprocessed_data/data.json', 'r') as f:
        encode = json.load(f)
    return torch.tensor(encode['Data'], dtype=torch.long)

def train_validation_split(data):
    n = int(0.8*len(data))
    training_data = data[:n]
    validation_data = data[n:]

    return training_data, validation_data

def get_batch(data, block_size, batch_size):
    randint = torch.randint(len(data) - block_size, (batch_size,))
    prediction = torch.stack([ data[i:i+block_size] for i in randint ])
    target = torch.stack([ data[i+1:i+block_size+1] for i in randint ])

    prediction, target = prediction.to(device), target.to(device)
    
    return prediction, target

def train_model(model, learning_rate, epochs, prediction, target):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        logits, loss = model.forward(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(loss.item())
