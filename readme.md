# KKLayer: A Theoretical Neural Network Layer

**Status**: 🚫 Not suitable for practical use 🚫

## ⚠️ Important Note

This layer was developed as a theoretical exploration and shows no additional benefits over traditional convolutional layers. Due to its computational complexity and lack of performance improvements, **it is not recommended for practical use.**

## Overview

KKLayer is a custom neural network layer inspired by the Kramers-Kronig relations and Fourier transformations. It was designed as an experimental alternative to classical convolutional layers (Conv2d) in deep learning models, aiming to provide a new way to extract features from input data.

However, after testing, it has been observed that the KKLayer does not offer any significant advantages and is, in fact, much more computationally intensive. When used in place of standard convolutional layers in common neural network architectures, the KKLayer results in slower training times and lower model accuracy.

## Motivation

The idea behind KKLayer was to explore the potential of using Fourier transforms combined with learnable parameters (α and β) to create a feature extraction mechanism. This layer attempts to mimic the Kramers-Kronig relations, which are used in physics to relate the real and imaginary parts of complex functions, by applying a parameterized transformation to the frequency domain representation of the input.

## Theoretical Foundations

**Kramers-Kronig Relations**: The layer was inspired by the integral transforms in the Kramers-Kronig relations, aiming to relate components in the frequency domain.

**Fourier Transform**: The KKLayer applies a 2D Fourier transform to the input, followed by a parameterized linear transformation involving learnable alpha and beta parameters.

## Performance

Despite its innovative design, the KKLayer has proven to be impractical for real-world applications:

**Training Time**: The KKLayer significantly increases training time. For instance, on a GPU T4:
- Replacing the first convolutional layer with KKLayer (output channels: 16) increased the time per epoch on FashionMNIST from 19 seconds to 50 seconds, with a noticeable drop in accuracy.
- Replacing the first two convolutional layers (producing 32 output channels) led to an epoch time exceeding 20 minutes, with further degradation in performance.

**Accuracy**: Models using the KKLayer tend to underperform compared to standard CNNs, with lower accuracy on benchmark datasets.

## Advantages

**Theoretical Exploration**: Provides a new perspective on feature extraction using principles from physics and signal processing.
**Learnable Parameters**: Introduces flexibility through learnable transformations in the frequency domain.

## Disadvantages

**Computational Complexity**: Extremely slow training, especially when scaling up the number of channels.
**No Performance Gains**: Fails to improve, and often degrades, model accuracy compared to standard convolutional layers.
**Resource Intensive**: Requires significantly more computational resources without corresponding benefits.

## Discontinuation

Due to its inefficiency and lack of practical utility, further work on the KKLayer has been **discontinued**. It remains a theoretical exercise, and its implementation is preserved only for reference and educational purposes.

## Conclusion

While the KKLayer offers an interesting theoretical exploration of alternative feature extraction mechanisms, it is not suitable for practical use. The classical convolutional layer remains superior in terms of both performance and computational efficiency.