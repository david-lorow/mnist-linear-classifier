# Linear Classifier: MNIST Database

A PyTorch implementation of a linear classifier trained to recognize handwritten digits from the MNIST (Modified National Institute of Standards and Technology) database, a benchmark dataset popularized by Yann LeCun et al., with hyperparameter grid search and evaluation tools.

 

## Overview

Each MNIST image is a 28×28 grayscale image, flattened into a vector of 784 pixel values. The model maps this input to one of 10 classes (digits 0–9) using a single linear (fully connected) layer. Training is done via Stochastic Gradient Descent (SGD) with cross-entropy loss and L2 regularization (weight decay). Hyperparameters are tuned via a greedy 1D grid search over learning rate, weight decay, and batch size.

 

## The Math

### The Classifier

The math for how the model works, comes out to a linear equation that is interpreted as a probability. The digits 0 to 9 yield 10 distinct **classes**. Each input image has pixels (or dimensions) 28 × 28, which results in 784 total **input features** after flattening. So essentially a flattened vector representing pixel values will be fed in and the predicted digit based on the input image will come out.

The logits for this linear classifier can be expressed as:

$$
z_k = b_k + \sum_{i=1}^{784} W_{k,i}\, x_i
$$

where:

- $x_i$ represents the $i$-th pixel value of the flattened input image
- $W_{k,i}$ is the weight associated with pixel $i$ for class $k$
- $b_k$ is the bias term for class $k$
- $z_k$ is the resulting logit (pre-softmax), after which argmax will provide de facto classification for class $k$

Or equivalently, this expression can be written in vector form as

$$
z_k = b_k + w_k^\top x
$$

where $x \in \mathbb{R}^{784}$ is the flattened input vector and $w_k \in \mathbb{R}^{784}$ is the weight vector corresponding to class $k$. This is the single equation that captures the essence of the model. In that vector, $z_k$, will be 10 values that are transformed into 10 possible choices, from which the most probable output is chosen, the classification.

### The Training

Cross-entropy loss is strict and that's thanks to the negative log applied to softmax. It is calculated like this for a single input with logits $z$ and true class label $y$, the loss is:

$$
\mathcal{L}_{CE} = - \log \left( \frac{e^{z_y}}{\sum_{k=1}^{K} e^{z_k}} \right)
$$

For a batch of $N$ samples:

$$
\mathcal{L}_{CE} = -\frac{1}{N} \sum_{n=1}^{N} \log \left( \frac{e^{z_{y_n}}}{\sum_{k=1}^{K} e^{z_k}} \right)
$$

L2 regularization, like mentioned before, is added to discourage excessively large weight magnitudes and improve generalization:

$$
\mathcal{L}_{L2} = \lambda \| W \|_F^2
$$

where $W \in \mathbb{R}^{10 \times 784}$ contains all model weights and

$$
\| W \|_F^2 = \sum_{i,j} W_{i,j}^2
$$

is the squared Frobenius norm of the weight matrix.

The full objective function becomes:

$$
\mathcal{L}(W,b) = \mathcal{L}_{CE}(W,b) + \lambda \| W \|_F^2
$$

This objective is minimized using Stochastic Gradient Descent (SGD) with learning rate $\eta$, where the weight update rule is:

$$
W \leftarrow W - \eta \nabla_W \mathcal{L}(W,b)
$$

With the information from **backpropagation** that identifies the source of loss, SGD can make the necessary adjustments to shift weights in the hopes of minimizing loss.

Expanding the gradient gives:

$$
W \leftarrow W - \eta \left( \nabla_W \mathcal{L}_{CE} + 2\lambda W \right)
$$

In practice, PyTorch implements weight decay directly in the optimizer step rather than explicitly adding the penalty term to the loss.

 

## Hyperparameters

| Parameter | Base Value | Search Grid |
|---|---|---|
| Learning rate $\eta$ | `1e-3` | `[3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]` |
| Weight decay $\lambda$ | `1e-4` | `[0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]` |
| Batch size | `128` | `[16, 32, 64, 128, 256, 512]` |
| Epochs | `10` | `—` |

 

## Data Split

The MNIST training set (60,000 samples) is split 90/10 into training and validation sets using a fixed seed (`2138`). The test set (10,000 samples) is held out for final evaluation.

| Split | Samples |
|---|---|
| Train | 54,000 |
| Validation | 6,000 |
| Test | 10,000 |

 

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- seaborn

 

## Usage

Open `LinearClassifier_by_Lorow.ipynb` in Jupyter and run cells sequentially. The notebook will:

1. Download MNIST if not already present
2. Define and train the linear classifier
3. Run grid search over hyperparameters
4. Plot training/validation loss and accuracy curves
5. Display a confusion matrix on the test set
