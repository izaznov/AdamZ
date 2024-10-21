import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import time
import seaborn as sns
import pandas as pd
import os
import random
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from collections import deque
from torch.optim import AdamW
from AdamZ import AdamZ

"""
This script demonstrates the implementation and training of a simple neural network model for binary classification using the
 synthetic dataset generated by the `make_circles` function from scikit-learn. The primary focus is on comparing the performance 
 of various optimization algorithms, including the newly proposed optimizer, AdamZ.

Key Components:
- **Data Generation and Visualization**: The script uses `make_circles` to generate a dataset of circular patterns, which is then split into 
training and testing sets. Matplotlib is used to visualize the data distribution.
- **Dataset and DataLoader**: Custom PyTorch Dataset and DataLoader classes are implemented to handle batch processing of the data.
- **Neural Network Architecture**: A simple feedforward neural network with one hidden layer is defined using PyTorch's `nn.Module`.
 The network uses ReLU activation for the hidden layer and a sigmoid activation for the output layer.
- **Optimizers**: The script compares multiple optimizers, including Adam, AdamW, SGD, RMSprop, Adagrad, Adamax, ASGD, NAdam, 
and the newly proposed optimizer, AdamZ, to evaluate their performance in training the model.
- **Training and Evaluation**: The `train_and_evaluate` function handles the training process, loss computation using binary cross-entropy, 
and evaluation of model accuracy on the test set. It also measures training duration and tracks loss history.
- **Simulation and Results Visualization**: The script runs multiple simulations to gather statistics on accuracy and training duration for each optimizer.
 Results are visualized using seaborn for boxplots and matplotlib for loss evolution plots.
- **Statistical Analysis**: A function is provided to calculate and display median and percentile statistics for accuracy and training duration across 
different optimizers.

This script is designed to provide the benchmarks for the newly proposed optimizer, AdamZ, for a shallow neural network on a 
simple binary classification task.
"""

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


seed = 42

random.seed(seed)

# Set the seed for Python's built-in random module
random.seed(seed)

# Set the seed for NumPy
np.random.seed(seed)

# Set the seed for PyTorch
torch.manual_seed(seed)

# Define constants
batch_size = 64
input_dim = 2
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
num_epochs = 10
num_simulations = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If you are using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    
# Ensure deterministic behavior for CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data definition & preparation
X, y = make_circles(n_samples=10000, noise=0.05, random_state=26)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

# Visualize the data.
fig, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral)
train_ax.set_title("Training Data")
train_ax.set_xlabel("Feature #0")
train_ax.set_ylabel("Feature #1")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Feature #0")
test_ax.set_title("Testing data")
plt.show()


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
        return x

def train_and_evaluate(optimizer_name):
    model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)
    loss_fn = nn.BCELoss()
    
    # Initialize the chosen optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_name == "ASGD":
        optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "NAdam":
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamZ":
        optimizer = AdamZ(model.parameters(), lr=learning_rate, overshoot_factor=0.5, stagnation_factor=1.2, stagnation_threshold=0.2, patience=100, stagnation_period=10)
        
    start_time = time.time()
    
    # List to store loss values
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
    
            def closure():
                optimizer.zero_grad()
                pred = model(X)
                
                loss = loss_fn(pred, y.unsqueeze(-1))
                loss.backward()
                return loss
    
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            
            
            if optimizer_name == "AdamZ":
                optimizer.step(closure)  # L-BFGS requires closure                  
                
            else:
                loss.backward()
                optimizer.step()
                
                
            epoch_loss += loss.item() 
                
                # Average loss for the epoch
        loss_history.append(epoch_loss / len(train_dataloader))

    training_duration = time.time() - start_time
    
    torch.save(model.state_dict(), f'{optimizer_name}_circle_model.pth')  

    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X.to(device))
            predicted = (outputs >= 0.5).float().squeeze()
            total += y.size(0)
            correct += (predicted == y.to(device)).sum().item()

    accuracy = 100 * correct / total
    return accuracy, training_duration, loss_history

# Simulation for all optimizers
optimizers = ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adamax", "ASGD", "NAdam", "AdamZ"]

results = {opt: {'accuracy': [], 'training_duration': [], 'loss_history': []} for opt in optimizers}

for _ in range(num_simulations):
    for opt in optimizers:
        accuracy, training_duration, loss_history = train_and_evaluate(opt)
        results[opt]['accuracy'].append(accuracy)
        results[opt]['training_duration'].append(training_duration)
        results[opt]['loss_history'].append(loss_history)

# Plotting loss histories
plt.figure(figsize=(12, 8))
for opt in optimizers:
    # Average loss history across simulations
    avg_loss_history = np.mean(results[opt]['loss_history'], axis=0)
    plt.plot(avg_loss_history, label=f'{opt} Loss')

# Set X-axis to integer values
plt.xticks(ticks=np.arange(num_epochs), labels=np.arange(1, num_epochs + 1))

plt.title('Loss evolution')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_decline_optimizers.png')
plt.show()

# Plotting results
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy boxplot
sns.boxplot(data=[results[opt]['accuracy'] for opt in optimizers], ax=axs[0], showfliers=False)
axs[0].set_title('Accuracy Comparison')
axs[0].set_xticklabels(optimizers)
axs[0].set_ylabel('Accuracy (%)')

# Training duration boxplot
sns.boxplot(data=[results[opt]['training_duration'] for opt in optimizers], ax=axs[1], showfliers=False)
axs[1].set_title('Training Duration Comparison')
axs[1].set_xticklabels(optimizers)
axs[1].set_ylabel('Training Duration (s)')

plt.tight_layout()
plt.savefig('comparison_charts_circle_zigzag.png')
plt.show()

# Calculate statistics
def calculate_statistics(data):
    median = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    return median, percentile_25, percentile_75

def print_statistics_table(results):
    stats = {}
    for opt in results.keys():
        accuracy_stats = calculate_statistics(results[opt]['accuracy'])
        duration_stats = calculate_statistics(results[opt]['training_duration'])
        stats[opt] = {
            'Accuracy Median': accuracy_stats[0],
            'Accuracy 25th': accuracy_stats[1],
            'Accuracy 75th': accuracy_stats[2],
            'Duration Median': duration_stats[0],
            'Duration 25th': duration_stats[1],
            'Duration 75th': duration_stats[2],
        }
    
    df = pd.DataFrame(stats).T
    print("Statistics Table:")
    print(df.to_string())
    
    # Save the DataFrame to a CSV file
    df.to_csv('statistics_table_circle_zigzag.txt', index=True, sep='\t')

print_statistics_table(results)


def visualize_classification_results(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predictions = []
    data_points = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs >= 0.5).float().squeeze()

            true_labels.extend(y.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            data_points.extend(X.cpu().numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    data_points = np.array(data_points)

    # Separate data points based on classification results
    inner_correct = data_points[(true_labels == 0) & (predictions == 0)]
    inner_wrong = data_points[(true_labels == 0) & (predictions == 1)]
    outer_correct = data_points[(true_labels == 1) & (predictions == 1)]
    outer_wrong = data_points[(true_labels == 1) & (predictions == 0)]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(inner_correct[:, 0], inner_correct[:, 1], c='green', label='Outer Correct', alpha=0.6)
    plt.scatter(inner_wrong[:, 0], inner_wrong[:, 1], c='red', label='Outer Wrong', alpha=0.6)
    plt.scatter(outer_correct[:, 0], outer_correct[:, 1], c='blue', label='Inner Correct', alpha=0.6)
    plt.scatter(outer_wrong[:, 0], outer_wrong[:, 1], c='orange', label='Inner Wrong', alpha=0.6)
    
    plt.title('Classification Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()  # Add the legend
    plt.savefig('classification_results_circle_zigzag.png')
    plt.show()  

def load_model(optimizer_name):
    model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(f'{optimizer_name}_circle_model.pth'))
    model.eval()  # Set the model to evaluation mode
    return model

# Example of how to use the function
# Assuming `model` is your trained model and `test_dataloader` is your DataLoader for the test set

optimizer_name = "AdamZ"  # Example optimizer
model = load_model(optimizer_name)
test_accuracy = visualize_classification_results(model, test_dataloader, device)
