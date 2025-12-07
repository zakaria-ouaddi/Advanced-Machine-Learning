# EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs

This document explains the **EEGNet** architecture, a compact CNN designed for EEG-based Brain-Computer Interfaces, capable of generalizing across various paradigms (ERP, oscillatory tasks) with very few parameters.

## 1. Summary

*   **Goal:** Generalize across multiple EEG BCI paradigms using a compact model.
*   **Key Concepts:**
    *   **Temporal Convolution:** Learns bandpass-like filters.
    *   **Depthwise Convolution:** Learns frequency-specific spatial filters.
    *   **Separable Convolution:** Mixes feature maps efficiently.
*   **Training:** Adam optimizer, categorical cross-entropy, ~500 epochs, dropout for regularization.

## 2. Architecture Breakdown

**Notation:**
*   `C`: Channels
*   `T`: Time samples
*   `F1`: Number of temporal filters
*   `D`: Depth multiplier (spatial filters per temporal filter)
*   `F2`: Number of pointwise filters
*   `N`: Number of classes

### Block 1: Temporal & Spatial Filtering

1.  **Input Reshape:** `(1, C, T)` (or `(C, T, 1)` in Keras channels_last).
2.  **Temporal Conv2D:**
    *   Filters: `F1`
    *   Kernel Size: `(1, kernel_length)` (approx. half sampling rate, e.g., 64 for 128Hz).
    *   **Purpose:** Learns narrow-band temporal filters (like a filter bank).
3.  **Depthwise Conv2D:**
    *   Kernel Size: `(C, 1)`
    *   Depth Multiplier: `D`
    *   Constraint: Max-norm = 1.
    *   **Purpose:** Learns frequency-specific spatial filters for each temporal band.
4.  **Activation & Pooling:**
    *   BatchNorm -> ELU
    *   AveragePool `(1, 4)` (reduces temporal resolution).
    *   Dropout.

### Block 2: Separable Convolution

1.  **SeparableConv2D:**
    *   **Depthwise Part:** Temporal kernel `(1, 16)` (summarizes time).
    *   **Pointwise Part:** `1x1` conv, `F2` filters (mixes features).
    *   **Purpose:** Efficiently summarizes and mixes temporal/spatial features.
2.  **Activation & Pooling:**
    *   BatchNorm -> ELU
    *   AveragePool `(1, 8)`.
    *   Dropout.

### Classification Block

*   **Flatten**
*   **Dense:** `N` units, Softmax activation.
*   Constraint: Max-norm = 0.25.

## 3. Keras Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model

def EEGNet(C, T, N, F1=8, D=2, F2=None, dropoutRate=0.5, kernel_length=64):
    """
    Build EEGNet model.
    
    Args:
        C: Number of EEG channels
        T: Number of time samples
        N: Number of classes
        F1: Number of temporal filters
        D: Depth multiplier
        F2: Number of pointwise filters (default: D * F1)
        dropoutRate: Dropout rate
        kernel_length: Length of temporal kernel (default: 64 for 128Hz)
    """
    if F2 is None:
        F2 = D * F1

    input_shape = (C, T, 1) 
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(filters=F1, kernel_size=(1, kernel_length), padding='same', use_bias=False, name='temporal_conv')(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(C, 1), depth_multiplier=D, use_bias=False, 
                        depthwise_constraint=max_norm(1.0), padding='valid', name='depthwise_spatial_conv')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D(pool_size=(1, 4))(x)
    x = Dropout(dropoutRate)(x)

    # Block 2
    x = SeparableConv2D(filters=F2, kernel_size=(1, 16), padding='same', use_bias=False, name='separable_conv')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D(pool_size=(1, 8))(x)
    x = Dropout(dropoutRate)(x)

    # Classification
    x = Flatten()(x)
    outputs = Dense(N, activation='softmax', kernel_constraint=max_norm(0.25), name='classify')(x)

    return Model(inputs=inputs, outputs=outputs, name='EEGNet')
```

## 4. Training Recipe

*   **Optimizer:** Adam (default params).
*   **Loss:** Categorical Cross-Entropy.
*   **Epochs:** Up to 500 (use Early Stopping).
*   **Dropout:** 0.5 (within-subject), 0.25 (cross-subject).
*   **Class Imbalance:** Use `class_weight`.

## 5. Hyperparameters & Preprocessing

*   **Preprocessing:** Bandpass filter (task dependent), Downsample to 128Hz.
*   **Hyperparameters:**
    *   `F1`: 4 or 8
    *   `D`: 2
    *   `kernel_length`: 64 (at 128Hz)
