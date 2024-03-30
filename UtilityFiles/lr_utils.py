import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out
    

def train(model, optim, criterion, x, y, epochs=10):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(e, loss.data))
    return model

def evaluate_model(model, x, y):
    model.eval()  
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in zip(x, y):
            output = model(data)
            predicted = output >= 0.5 
            y_true.extend(target.view(-1).tolist())
            y_pred.extend(predicted.view(-1).tolist())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, confusion
