
# Physics-Constrained Neural Networks (PCNN) for Composite Failure Envelopes

This repository contains the official implementation of the paper **"Physics-Constrained Neural Networks for High-Fidelity Composite Failure Envelopes"**.

It introduces a framework that integrates classical failure criteria (Tsai-Wu, Cuntze, LaRC) into Deep Neural Networks via a physics-informed loss function. This approach reconciles the data-driven accuracy of neural networks with the physical consistency of established mechanics theories.

## 📂 Repository Architecture

The codebase is structured effectively to support multiple physical fidelity levels. It is organized into three independent modules, each corresponding to a specific failure criterion.

### Directory Structure
```text
.
├── Larc-PCNN/              # Implementation with LaRC04 constraints (High Fidelity)
│   ├── main.py             # Entry point for training and testing
│   ├── config.py           # Material properties and hyperparameters
│   ├── physics_loss.py     # Custom loss function implementing LaRC equations
│   └── ...
├── Cuntze-PCNN/            # Implementation with Cuntze constraints
│   ├── main.py
│   ├── cuntze_physics_loss.py
│   └── ...
├── Tsai-wu-PCNN/           # Implementation with Tsai-Wu constraints (Macroscopic)
│   ├── main.py
│   └── ...
├── datasetnew.csv          # [REQUIRED] The master dataset containing experimental data
└── README.md

```

### Core Logic & Workflow

The framework operates on a **Decoupled Prediction Strategy** and an **Adaptive Weighting Mechanism**. The logical flow for all three modules is as follows:

1. **Data Ingestion (`data_processor.py`)**:
* Loads experimental stress data (`sx`, `sy`, `txy`, etc.) from `datasetnew.csv`.
* Normalizes inputs and splits data into training/testing sets based on the Case ID.


2. **Neural Network (`model.py`)**:
* Architecture: A Multi-Layer Perceptron (MLP) with SwiGLU activation functions.
* **Input**: Normalized stress tensor components.
* **Output**: A scalar Failure Index () or Strength Ratio ().


3. **Physics-Constrained Optimization (`trainer.py` & `physics_loss.py`)**:
* The model minimizes a hybrid loss function: .
* ****: Ensures the model fits the experimental failure points.
* ****: Penalizes predictions that violate the specific failure criterion (e.g., LaRC04 fiber kinking or matrix cracking conditions).


4. **Adaptive Weighting (`adaptive_lambda.py`)**:
* The framework dynamically adjusts the weight  during training to balance the gradients between data fidelity and physical constraints, ensuring neither dominates the learning process prematurely.



## 🛠️ Environment Setup

The code is built using **Python 3.8+** and relies on **PyTorch** for automatic differentiation and model training.

### Dependencies

To set up the environment, you can create a Conda environment and install the required packages:

```bash
# 1. Create a virtual environment
conda create -n pcnn python=3.8
conda activate pcnn

# 2. Install PyTorch (Adjust cuda version based on your hardware)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install other scientific computing dependencies
pip install numpy pandas matplotlib scikit-learn

```

* **PyTorch**: Core deep learning framework.
* **Pandas**: For loading and manipulating `datasetnew.csv`.
* **NumPy**: For numerical operations and tensor transformations.
* **Matplotlib**: For plotting failure envelopes and loss curves.

## 🚀 Usage Guide

### 1. Data Preparation

Ensure that the `datasetnew.csv` file is present in the root directory. If you are running scripts from within a subfolder (e.g., `Larc-PCNN/`), ensure the data loader path is correctly set or copy the CSV into the subfolder.

### 2. Training the Model

Each physics module works independently. To train the model with the **LaRC04** criterion (recommended for highest fidelity):

```bash
cd Larc-PCNN

# Run with optimal configurations defined in the paper
python main.py --optimal

```

To train using **Tsai-Wu** or **Cuntze** criteria, simply navigate to the respective folder:

```bash
cd ../Tsai-wu-PCNN
python main.py --optimal

```

### 3. Configuration

You can modify hyperparameters (learning rate, batch size, network depth) and material properties in the `config.py` file located in each subdirectory.

* **`OPTIMAL_CONFIG`**: Contains the hyperparameters used to generate the results in the paper.
* **`MATERIAL_PROPS`**: Defines , etc., for the specific material cases.



```

```
