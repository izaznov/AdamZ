from setuptools import setup, find_packages

setup(
    name="adamz",  # Package name
    version="0.1.0",  # Initial version
    description="An enhanced Adam optimizer with overshoot and stagnation handling.",
    long_description=open("README.md").read(),
    long_description_content_type="AdamZ is an advanced variant of the Adam optimiser, developed to enhance convergence efficiency in neural network training. This optimiser dynamically adjusts the learning rate by incorporating mechanisms to address overshooting and stagnation, that are common challenges in optimisation. Specifically, AdamZ reduces the learning rate when overshooting is detected and increases it during periods of stagnation, utilising hyperparameters such as overshoot and stagnation factors, thresholds, and patience levels to guide these adjustments. While AdamZ may lead to slightly longer training times compared to some other optimisers, it consistently excels in minimising the loss function, making it particularly advantageous for applications where precision is critical. Benchmarking results demonstrate the effectiveness of AdamZ in maintaining optimal learning rates, leading to improved model performance across diverse tasks.",
    author="Ilia Zaznov",
    author_email="izaznov@gmail.com",
    url="https://github.com/zaznov/AdamZ",  # Replace with your repo
    packages=find_packages(),  # Automatically find the `adamz` package
    install_requires=[
        "torch>=1.9.0",  # Specify PyTorch as a dependency
        "numpy>=1.21.0",  # Specify NumPy as a dependency
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
)
