# SparseNet: Implementing and Pruning a Sparse Neural Network
## Project Overview
SparseNet is a neural network model that integrates sparsity into its architecture to reduce the number of active weights, aiming to optimize performance and efficiency. This project involves defining and training the SparseNet model using PyTorch and incorporating various pruning techniques to enhance model performance and reduce computational complexity.


## Dataset
- **Training Data:** Located in `data/train/`
- **Testing Data:** Located in `data/test/`
- Dataset consists of images with dimensions (64, 64, 3).

## Model Architecture
### LeNet
The LeNet model is a classic convolutional neural network with the following layers:
- 2 convolutional layers with ReLU activations and max-pooling
- 3 fully connected layers

### SparseNet
SparseNet is an extension of LeNet where convolutional and fully connected layers are replaced by sparse versions. It uses sparse masks to enforce sparsity during training.

## Pruning Techniques
### Magnitude Pruning
Weights below a certain magnitude threshold are set to zero.

### Random Pruning
Randomly selects weights to prune based on a pruning factor.

### Structured Pruning
Prunes entire structures such as filters or channels based on a pruning factor.

## Training
Training involves using the Adam optimizer and CrossEntropyLoss. Pruning is performed after each training epoch.

## Evaluation
Evaluation includes calculating the accuracy on the test dataset and visualizing the sparsity of the model.

## Visualization
Visualizations include loss curves, sparsity heatmaps, and bar charts of zero weights per layer.

## Pruning Analysis
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
* Train and test loss curves for different rounds of training
* Accuracy after pruning
* Visualization of filter weights and sparsity

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
git clone https://github.com/username/SparseNet.git
cd SparseNet
```
