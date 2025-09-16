# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:49:45 2025

@author: izazn
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
from collections import deque
from AdamZ import AdamZ
from scipy.stats import t
from brokenaxes import brokenaxes
import pickle


"""
This script implements and trains a deep neural network model leveraging a Convolutional Neural Network (CNN) architecture for image classification using the CIFAR-10 dataset. 
It focuses on comparing the performance of various optimization algorithms, including a newly proposed optimizer, AdamZ.

Key Components:
- **Data Loading and Transformation**: The CIFAR-10 dataset is loaded using torchvision, with transformations applied to normalize the RGB images. 
DataLoader is used to facilitate batch processing for both training and testing datasets.
- **Neural Network Architecture**: A CNN model is defined using PyTorch's `nn.Module`, featuring convolutional layers, max-pooling, fully connected layers, 
and dropout regularization. The architecture uses ReLU activations and log-softmax for output probabilities.
- **Optimizers**: The script evaluates the performance of several optimizers, including Adam, AdamW, SGD, RMSprop, Adagrad, Adamax, ASGD, NAdam, 
and the newly proposed AdamZ, to determine their effectiveness in training the model.
- **Training and Evaluation**: The `train_and_evaluate` function manages the training loop, computes loss using cross-entropy, and evaluates 
model accuracy on the test set. It tracks training duration and loss history for analysis.
- **Simulation and Results Visualization**: Multiple simulations are conducted to compile statistics on accuracy and training duration for each optimizer.
 Results are visualized using seaborn for boxplots and matplotlib for loss evolution plots.
- **Statistical Analysis**: Functions are provided to calculate and display median and percentile statistics for accuracy and training duration 
across different optimizers.

This script is designed to benchmark the performance of the AdamZ optimizer against other established optimizers in training a CNN for the CIFAR-10 image classification task.
"""



# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
#learning_rate = 0.001
learning_rate = 0.01
num_epochs = 10
num_simulations = 100

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])



train_subset_size = 1000  # Use only 10,000 samples for training
test_subset_size = 200    # Use only 2,000 samples for testing


train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

# Create subsets using slicing
train_subset = Subset(train_dataset, list(range(train_subset_size)))
test_subset = Subset(test_dataset, list(range(test_subset_size)))

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)
    
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: RGB channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2x
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust dimensions for CIFAR-10
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
# Training and evaluation function
def train_and_evaluate(optimizer_name):
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()

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
        #optimizer = AdamZ(model.parameters(), lr=learning_rate, overshoot_factor=0.5855823910277775, stagnation_factor=1.3907164204177742, stagnation_threshold=0.17037573127640235, patience=97, stagnation_period=10)
        optimizer = AdamZ(model.parameters(), lr=learning_rate, overshoot_factor=0.4847935262885537, stagnation_factor=1.2927002482335022, stagnation_threshold=0.2697554885246995, patience=129, stagnation_period=15)
        
        
    start_time = time.time()
    
    # List to store loss values
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            def closure():
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                return loss

            # Use closure for optimizers that require it
            if optimizer_name == "AdamZ":
                loss = optimizer.step(closure) 
            else:
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() 
                
                # Average loss for the epoch
        loss_history.append(epoch_loss / len(train_loader))


    training_duration = time.time() - start_time
      
    
    # Save the model after training
    torch.save(model.state_dict(), f'{optimizer_name}_model_cifar10.pth')

    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X.to(device))
            _, predicted = torch.max(outputs, 1)
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




with open('results_cifar10.pkl', 'wb') as f:
    pickle.dump(results, f)


# Plotting loss histories
plt.figure(figsize=(12, 8))
bax = brokenaxes(ylims=((0, 5), (5, 10)), hspace=0.05)

    
for opt in optimizers:
    # Average loss history across simulations
    avg_loss_history = np.mean(results[opt]['loss_history'], axis=0)
    
    # Add dots specifically for AdamZ
    if opt == "AdamZ":
        bax.plot(np.arange(num_epochs), avg_loss_history, label=f'{opt} Loss', marker='o')  # Add dots for AdamZ
    else:
        bax.plot(np.arange(num_epochs), avg_loss_history, label=f'{opt} Loss')  # Regular lines for others    
    
    

# Set X-axis to integer values
bax.set_xticks(np.arange(num_epochs))
bax.set_xticklabels(np.arange(1, num_epochs + 1))

bax.set_title('Loss evolution')
bax.set_xlabel('Epoch')
bax.set_ylabel('Loss')
bax.legend()
plt.savefig('loss_decline_cifar10.png')
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
plt.savefig('comparison_charts_cifar10_optimizers.png')
plt.show()


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
    df.to_csv('statistics_table_cifar_adamz.txt', index=True, sep='\t')

print_statistics_table(results)



def calculate_confidence_interval(data):
    """
    Calculate the 95% confidence interval for the given data.
    """
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error
    confidence = t.ppf(0.975, df=len(data) - 1) * std_err  # 95% confidence interval
    return mean - confidence, mean + confidence

def print_confidence_interval_table(results):
    """
    Print a table showing the confidence intervals for accuracy across optimizers.
    """
    ci_stats = {}
    for opt in results.keys():
        accuracy_data = results[opt]['accuracy']
        lower_ci, upper_ci = calculate_confidence_interval(accuracy_data)
        ci_stats[opt] = {
            'Median Accuracy': np.median(accuracy_data),
            '95% CI Lower': lower_ci,
            '95% CI Upper': upper_ci,
        }
    
    ci_df = pd.DataFrame(ci_stats).T
    print("Confidence Interval Table:")
    print(ci_df.to_string())
    
    # Save the DataFrame to a CSV file
    ci_df.to_csv('confidence_interval_table_cifar_adamz.txt', index=True, sep='\t')

print_confidence_interval_table(results)


def test_and_display(model, data_loader, device, num_images=100):
    model.eval()  # Set the model to evaluation mode
    images_shown = 0
    correct_predictions = 0
    total_predictions = 0

    # Create a figure to display images
    fig, axs = plt.subplots(10, 10, figsize=(15, 15))
    axs = axs.flatten()

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            for i in range(X.size(0)):
                if images_shown < num_images:
                    # Display the image
                    #axs[images_shown].imshow(X[i].cpu().numpy().squeeze(), cmap='gray')
                    axs[images_shown].imshow(np.transpose(X[i].cpu().numpy(), (1, 2, 0)))  # Convert to HWC format for RGB
                    # Set the title of the subplot
                    axs[images_shown].set_title(f"True: {y[i].item()}\nPred: {predicted[i].item()}")
                    axs[images_shown].axis('off')

                    # Count correct predictions
                    if predicted[i] == y[i]:
                        correct_predictions += 1

                    images_shown += 1
                    total_predictions += 1

                if images_shown >= num_images:
                    break
            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig('Cifar_test_opt_trn_adamz.png')
    plt.show()

    accuracy = 100 * correct_predictions / total_predictions
    print(f"Accuracy on {total_predictions} test images: {accuracy:.2f}%")
    return accuracy


def load_model(optimizer_name):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(f'{optimizer_name}_model_cifar10.pth'))
    model.eval()  # Set the model to evaluation mode
    return model


# Example usage
optimizer_name = "AdamZ"  # Choose the optimizer for which the model was saved
model = load_model(optimizer_name)
test_accuracy = test_and_display(model, test_loader, device)