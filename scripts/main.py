import torch
import BigramLanguageModel

from data_preprocessing import *
from train import *

batch_size = 4
block_size = 8
learning_rate = 3e-4
epochs = 10000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    preprocess_data()
    vocab_size = vocabulary_size()

    data = load_data()
    training_data, validation_data = train_validation_split(data=data)
    prediction, target = get_batch(training_data, block_size=block_size, batch_size=batch_size)

    model = BigramLanguageModel.BigramLanguageModel(vocab_size=vocab_size)
    m = model.to(device)
    train_model(model, learning_rate, epochs, prediction=prediction, target=target)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_chars = character_decoding(m.generate(context, max_new_token=500)[0].tolist())
    print(generated_chars)

if __name__ == '__main__':
    main()
