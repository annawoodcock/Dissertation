import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_diabetes_data_5050():
    data = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    data = data.dropna()
    y = torch.tensor(data["Diabetes_binary"].values).float().unsqueeze(1)
    data = data.drop(columns=['Diabetes_binary'])
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    return train_test_split(x, y, test_size=0.2, random_state=73)

def print_metrics(accuracy, precision, recall, f1, confusion):
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n {confusion}')
