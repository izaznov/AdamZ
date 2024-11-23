# AdamZ is an enhanced variant of the widely-used Adam optimizer, designed to improve upon its predecessor by offering more efficient convergence and potentially better generalization capabilities across various neural network training tasks.

This repository provides the implementation of the newly proposed optimizer, AdamZ. It includes comprehensive benchmarking scripts to evaluate AdamZ's performance against other popular optimizers across two datasets and two neural network architectures, ranging from shallow to deep models.

# Repository Overview
adamz_optimizer/
├── Whitepaper/
│   ├── AdamZ.py         # Simplified implementation of the AdamZ optimizer
│   ├── Circle_adamz_whitepaper.py  # Implementation for synthetic dataset
│   ├── Mnist_adamz_whitepaper.py   # Implementation for MNIST dataset
├── README.md                # Project documentation
├── adamz/
│   ├── __init__.py          # Package initialization for the AdamZ optimizer 
│   ├── adamz.py         # Full-scale Torch implementation of AdamZ
│   ├── optimizer.py         # Torch optimizer methods from Pytorch library
├── tests/
│   ├── test_adamz.py    # Unit tests for the AdamZ optimizer
├── README.md                # Project documentation
├── setup.py                 # Setup script for packaging and installation
├── LICENSE                  # License information
```

### Description of Folders
- **`adamz/`**: Contains the core implementation of the AdamZ optimizer.
- **`tests/`**: Includes unit tests for validating the optimizer's functionality.
- **`whitepaper/`**: Holds scripts that implement the original research paper experiments, including synthetic and MNIST datasets.
- **Root files**:
  - `README.md`: Project documentation.
  - `setup.py` and `pyproject.toml`: Installation and build configuration.
  - `LICENSE`: License information for the project.





## Datasets used

1. **Synthetic Dataset**
   - **Description**: Utilizes the `make_circles` function to generate a synthetic dataset for binary classification tasks.
   - **Purpose**: Helps in assessing the optimizer's performance on a simple, controlled dataset.

2. **MNIST Dataset**
   - **Description**: A widely-used dataset for handwritten digit recognition containing 60,000 training images and 10,000 test images.
   - **Purpose**: Provides a more complex and real-world scenario to evaluate the optimizer's effectiveness in image classification tasks.

### Neural Networks

1. **Shallow Neural Network**
   - **Architecture**: Consists of a few fully connected layers, designed for quick experimentation and testing on smaller datasets.
   - **Use Case**: Ideal for the synthetic dataset to quickly benchmark optimizer performance.

2. **Deep Neural Network**
   - **Architecture**: A deeper architecture with multiple layers, including a multi-head attention mechanism for feature extraction.
   - **Use Case**: Applied to the MNIST dataset to evaluate the optimizer's capability in handling complex data and deeper architectures.
### Scripts

1. **AdamZ optimizer implementation**
   - **Script**: `AdamZ.py`
   - **Description**: This implementation of the AdamZ optimizer is an extension of the traditional Adam optimizer, 
designed to adjust the learning rate dynamically based on the characteristics of the loss function during training. 
It introduces mechanisms to handle overshooting and stagnation in the optimization process.

2. **Benchmarking of AdamZ against other popular optimizers for simple binary classification task**
   - **Script**: `Circle_adamz_whitepaper.py`
   - **Description**: This script demonstrates the implementation and training of a simple neural network model for binary classification using the synthetic dataset generated by the `make_circles` function from scikit-learn. The primary focus is on comparing the performance of various optimization algorithms, including the newly proposed optimizer, AdamZ.
  
3. **Benchmarking of AdamZ against other popular optimizers for the MNIST digit classification task**
   - **Script**: `Mnist_adamz_whitepaper.py`
   - **Description**: This script implements and trains a deep neural network model leveraging a multi-head attention mechanism for digit classification using the MNIST dataset. It focuses on comparing the performance of various optimization algorithms, including a newly proposed optimizer, AdamZ.

#### Prerequisites

- Python 3.8 or later
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas

#### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/izaznov/AdamZ.git
```

### Usage


Run the scripts to train models and compare optimizer performance:

```bash
python Circle_adamz_whitepaper.py
python Mnist_adamz_whitepaper.py
```
Ensure that the `AdamZ.py` file is located in the same directory as your script.

Add the following import statement at the beginning of your script to make the AdamZ optimizer available:

```python
from AdamZ import AdamZ
```

Instantiate the AdamZ optimizer similarly to other standard optimizers, ensuring you configure the hyperparameters to suit your specific task. Note that the performance of AdamZ is highly sensitive to these parameters, and default settings may not be optimal for all applications.

```python
optimizer = AdamZ(
    model.parameters(),
    lr=learning_rate,
    overshoot_factor=0.5,
    stagnation_factor=1.2,
    stagnation_threshold=0.2,
    patience=100,
    stagnation_period=10
)
```

### Results

The scripts generate visualizations and save models for further analysis. Results are saved as images and can be found in the respective directories.

## Contributions

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions and improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please contact i.zaznov@pgr.reading.ac.uk or open an issue on GitHub.

