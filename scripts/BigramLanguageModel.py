import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, prediction, targets=None):
        logits = self.embedding_table(prediction)

        if targets == None:
            loss = None
        else:
            batch_dim, seq_dim, channel_dim = logits.shape
            logits = logits.view(batch_dim*seq_dim, channel_dim)
            targets = targets.view(batch_dim*seq_dim)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, prediction, max_new_token):
        for _ in range(max_new_token):
            logits, loss = self.forward(prediction)
            # print(type(logits), logits)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_prediction = torch.multinomial(probs, num_samples=1)

            prediction = torch.cat((prediction, next_prediction), dim=1)
        
        return prediction
