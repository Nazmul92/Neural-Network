# SparseNet: Development and Optimization of a Sparse Neural Network
## Project Overview
This project focuses on building and training SparseNet using PyTorch, leveraging pruning techniques such as magnitude, random, and structured pruning to minimize computational complexity without compromising accuracy. By visualizing sparsity patterns and performance metrics, the project demonstrates how SparseNet achieves a balance between model simplicity and effectiveness, making it a valuable solution for resource-constrained machine learning applications.

## 1. Dataset (HAM10000 image dataset)
- **Training Data:** Located in `data/train/`
- **Testing Data:** Located in `data/test/`
- Input image size: 64x64x3

## 2. Model Architecture
### 2.1. LeNet
The LeNet model is a classic convolutional neural network with the following layers:
- 2 convolutional layers with ReLU activations and max-pooling
- 3 fully connected layers

### 2.2. SparseNet
SparseNet is an extension of LeNet where convolutional and fully connected layers are replaced by sparse versions. It uses sparse masks to enforce sparsity during training.

## 3. Pruning Techniques
- **Magnitude Pruning:** Removes weights below a threshold.
- **Random Pruning:** Randomly prunes weights.
- **Structured Pruning:** Prunes entire filters or channels.

## 4. Training and Evaluation
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Metrics:** Accuracy, sparsity, and test loss curves
- **Visualization:** Sparsity heatmaps, filter weight visualizations, and layer-wise sparsity.


## 5. Pruning Analysis
Pruning analysis involves interpreting the sparsity of each layer and visualizing the number of zero weights.


```python
def interpret_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, SparseConv2d) or isinstance(module, SparseLinear):
            weights = module.weight.data.cpu().numpy()
            sparsity = 1 - (weights != 0).sum() / weights.size
            print(f'{name} (sparsity = {sparsity:.2f}): {int((weights == 0).sum())} weights are zero out of {weights.size}')
```


## Results
- Visualized the trade-off between sparsity and accuracy.
- Demonstrated the impact of different pruning techniques on model performance.

## Key Takeaways
- SparseNet reduces computational load while preserving accuracy.
- Pruning enhances model efficiency without significant performance degradation.

## Requirements
* Python 3.8+
* PyTorch 1.8+
* numpy
* matplotlib
* seaborn
## Install Dependencies
```python
pip install -r requirements.txt
```
## Running the project
* Clone the repository
```python
git clone https://github.com/Nazmul92/Neural-Network.git
cd SparseNet
```
