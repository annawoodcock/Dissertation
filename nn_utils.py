import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class NeuralNet1(torch.nn.Module):
    def __init__(self,input_size=21, hidden=10, output=1):
        super(NeuralNet1, self).__init__()        
        self.fc1 = torch.nn.Linear(input_size, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x
    
class NeuralNet2(torch.nn.Module):
    def __init__(self,input_size=21, hidden=10, output=1):
        super(NeuralNet2, self).__init__()        
        self.fc1 = torch.nn.Linear(input_size, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        x = x * x
        x = self.fc3(x)
        return x

def train(model, train_loader, criterion, optimizer, n_epochs=10):
    # model in training mode
    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    
    # model in evaluation mode
    model.eval()
    return model

def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    t_start = time.perf_counter()

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            predicted = torch.sigmoid(output) >= 0.5  # Apply sigmoid and threshold to get binary predictions
            y_true.extend(target.view(-1).tolist())
            y_pred.extend(predicted.view(-1).tolist())

    t_end = time.perf_counter()

    print(f"Evaluated test_set of {len(y_pred)} entries in {int(t_end - t_start)} seconds")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, confusion