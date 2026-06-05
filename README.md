# Image-Classification-model

# Intel Image Scene Classification using EfficientNet V2 S

A high-performance deep learning pipeline designed to classify natural and urban landscapes into six distinct categories: Buildings, Forest, Glacier, Mountain, Sea, and Street. The system optimizes deep convolutional features to achieve a validation accuracy of approximately 95 percent.

## Technical Specifications

### Architecture

* **Base Model:** EfficientNet V2 Small (`efficientnet_v2_s`) initialized with pre-trained ImageNet weights.
* **Classification Head:** The default top layer was replaced with a custom linear layer mapping 1,280 features to 6 output channels representing the target scene classes.
* **Activation Function:** Internal layers leverage the native SiLU (Sigmoid Linear Unit / Swish) activation to maintain smooth non-monotonic gradient flow.

### Data Engineering

* **Resolution Scaling:** Images are preprocessed and scaled to 384x384 pixels to ensure high-fidelity spatial feature extraction, heavily reducing class ambiguity between visually similar categories such as mountains and glaciers.
* **Normalization:** Input tensors are normalized using standardized ImageNet channel statistics (mean and standard deviation) to stabilize loss convergence.
* **Data Integrity:** Strict mathematical isolation is enforced across the training, validation, and test subsets to guarantee completely leak-free evaluation.

### Training and Optimization Strategy

* **Optimizer:** NAdam (Nesterov-accelerated Adam), combining adaptive learning rates with Nesterov momentum to look ahead along the gradient trajectory.
* **Learning Rate:** Base learning rate set explicitly to 1e-4, optimized for stable transfer learning fine-tuning.
* **Regularization:** L2 regularization (weight decay of 1e-4) applied directly to the optimizer parameters to restrict network complexity and eliminate overfitting.
* **LR Scheduling:** Integrated an adaptive `ReduceLROnPlateau` scheduler set to scale down the learning rate by a factor of 0.1 if the validation accuracy plateaus for more than one epoch.
* **Runtime Duration:** Terminated at 5 epochs upon reaching a stable optimization plateau, balancing performance with computational economy.

### Hardware Acceleration and Resource Management

* **Parallelization:** Implemented distributed compute across dual NVIDIA T4 Graphics Processing Units via PyTorch's `nn.DataParallel` abstraction layer.
* **Memory Management:** Configured constrained batch sizing paired with automated asynchronous CUDA cache flushing (`torch.cuda.empty_cache()`) at the end of each epoch iteration to eliminate memory fragmentation and prevent out-of-memory errors.

## Performance Metrics

* **Peak Validation Accuracy:** ~95.0%
* **Final Cross-Entropy Loss:** 0.1650
* **Validation F1-Score:** 0.9410

## Dataset Setup

The pipeline evaluates the Intel Image Classification dataset. To replicate the execution environment:

1. Download the raw data directly from the official Intel Image Classification repository on Kaggle.
2. Extract the file contents into your local workspace directory.
3. Verify that the file layout conforms to the standard directory convention before initiating the execution notebook:

```text
├── seg_train/
│   └── seg_train/
│       ├── buildings/
│       ├── forest/
│       ├── glacier/
│       ├── mountain/
│       ├── sea/
│       └── street/
├── seg_test/
│   └── seg_test/
└── seg_pred/
    └── seg_pred/

```




Project and Data link: 
