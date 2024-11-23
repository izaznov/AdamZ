# AdamZ is an enhanced variant of the widely-used Adam optimizer, designed to improve upon its predecessor by offering more efficient convergence and potentially better generalization capabilities across various neural network training tasks.

This repository provides the implementation of the newly proposed optimizer, AdamZ. It includes comprehensive benchmarking scripts to evaluate AdamZ's performance against other popular optimizers across two datasets and two neural network architectures, ranging from shallow to deep models. It also contains simple unit tests to ensure the reliability of AdamZ.

# Repository Overview
```
AdamZ/
├── Whitepaper/
│   ├── AdamZ.py         # Simplified implementation of the AdamZ optimizer
│   ├── Circle_adamz_whitepaper.py  # Benchmarking experiments for the synthetic dataset
│   ├── Mnist_adamz_whitepaper.py   # Benchmarking experiments for the MNIST dataset
│   ├── README.md   # Project documentation
├── adamz/
│   ├── __init__.py          # Package initialization for the AdamZ optimizer 
│   ├── adamz.py         # Full-scale Torch implementation of AdamZ
│   ├── optimizer.py         # Torch optimizer methods from Pytorch library
├── tests/
│   ├── test_adamz.py    # Unit tests for the AdamZ optimizer
├── README.md                # Project documentation
├── setup.py                 # Setup script for packaging and installation
├── pyproject.toml           # Build system configuration
├── LICENSE                  # License information
```

### Description of Folders
- **`adamz/`**: Contains the core implementation of the AdamZ optimizer.
- **`tests/`**: Includes unit tests for validating the optimizer's functionality.
- **`whitepaper/`**: Holds scripts that implement the original research paper experiments, including synthetic and MNIST datasets.
- **Root files**:
  - `README.md`: Project documentation.
  - - **Root files** like `README.md`, `setup.py`, and `pyproject.toml` provide documentation, installation, and build configuration.
  - `LICENSE`: License information for the project.

#### Prerequisites

- Python 3.8 or later
- PyTorch 2.5.1

#### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/izaznov/AdamZ.git
```
or 
```bash
pip install git+https://github.com/izaznov/AdamZ.git
```

### Usage
Instantiate the AdamZ optimizer similarly to other standard optimizers, ensuring you configure the hyperparameters to suit your specific task. Note that the performance of AdamZ is highly sensitive to these parameters, and default settings may not be optimal for all applications.

```python
from adamz import AdamZ
import torch

model = torch.nn.Linear(10, 1)
optimizer = AdamZ(
    model.parameters(),
    lr=learning_rate,
    overshoot_factor=0.5,
    stagnation_factor=1.2,
    stagnation_threshold=0.2,
    patience=100,
    stagnation_period=10
)
# Training loop
for input, target in dataset:
    optimizer.zero_grad()
    loss = loss_fn(model(input), target)
    loss.backward()
    optimizer.step()
```

## Contributions

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions and improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please contact i.zaznov@pgr.reading.ac.uk or open an issue on GitHub.

