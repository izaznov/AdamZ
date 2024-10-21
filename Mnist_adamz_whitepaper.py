import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
from collections import deque
from AdamZ import AdamZ


"""
This script implements and trains a deep neural network model leveraging multi-head attention mechanism for digit classification using the MNIST dataset.
It focuses on comparing the performance of various optimization algorithms, including a newly proposed optimizer, AdamZ.

Key Components:
- **Data Loading and Transformation**: The MNIST dataset is loaded using torchvision, with transformations applied to normalize the images. 
DataLoader is used to facilitate batch processing for both training and testing datasets.
- **Neural Network Architecture**: A neural network model is defined using PyTorch's `nn.Module`, featuring an embedding layer, multi-head attention mechanism, 
and fully connected layers. The architecture includes dropout for regularization and uses ReLU and log-softmax activations.
- **Optimizers**: The script evaluates the performance of several optimizers, including Adam, AdamW, SGD, RMSprop, Adagrad, Adamax, ASGD, NAdam, 
and the newly proposed AdamZ, to determine their effectiveness in training the model.
- **Training and Evaluation**: The `train_and_evaluate` function manages the training loop, computes loss using cross-entropy, and evaluates 
model accuracy on the test set. It tracks training duration and loss history for analysis.
- **Simulation and Results Visualization**: Multiple simulations are conducted to compile statistics on accuracy and training duration for each optimizer.
 Results are visualized using seaborn for boxplots and matplotlib for loss evolution plots.
- **Statistical Analysis**: Functions are provided to calculate and display median and percentile statistics for accuracy and training duration 
across different optimizers.

This script is designed to benchmark the performance of the AdamZ optimizer against other established optimizers in training of a deep neural network for the 
MNIST digit classification task.

"""

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 5
num_simulations = 100

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
class NeuralNetwork(nn.Module):
    def __init__(self, embed_dim=28, num_heads=4, num_classes=10):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(28, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(embed_dim * 28, 128)  # Assuming the flattened attention output
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, feature_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, 28, 28)  # Treat each row as a sequence

        # Embed the input
        x = self.embedding(x)

        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)

        # Flatten the attention output
        x = attn_output.reshape(batch_size, -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
 
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
        optimizer = AdamZ(model.parameters(), lr=learning_rate, overshoot_factor=0.864, stagnation_factor=1.088, stagnation_threshold=0.076, patience=164, stagnation_period=28)

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
    torch.save(model.state_dict(), f'{optimizer_name}_model.pth')

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
plt.savefig('loss_decline_mnist_zigzag.png')
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
plt.savefig('comparison_charts_mnist_optimizers.png')
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
    df.to_csv('statistics_table_mnist_trn_zigzag.txt', index=True, sep='\t')

print_statistics_table(results)


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
                    axs[images_shown].imshow(X[i].cpu().numpy().squeeze(), cmap='gray')
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
    plt.savefig('Mnist_test_opt_trn_zigzag.png')
    plt.show()

    accuracy = 100 * correct_predictions / total_predictions
    print(f"Accuracy on {total_predictions} test images: {accuracy:.2f}%")
    return accuracy


def load_model(optimizer_name):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(f'{optimizer_name}_model.pth'))
    model.eval()  # Set the model to evaluation mode
    return model


# Example usage
optimizer_name = "AdamZ"  # Choose the optimizer for which the model was saved
model = load_model(optimizer_name)
test_accuracy = test_and_display(model, test_loader, device)