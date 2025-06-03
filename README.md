# 🧠🔍 CNN Benchmark – Custom CNN vs. AlexNet (v3)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/CNN_vs_AlexNet%20(3).ipynb)

This Jupyter notebook explores **convolutional neural network** design by comparing a **lightweight custom CNN** against a **downsized AlexNet** on two classic image classification datasets. The project demonstrates how architectural choices affect model performance, training efficiency, and generalization capabilities.

## 📊 Dataset Overview

| Dataset | Samples | Classes | Image Size | Description |
|---------|---------|---------|------------|-------------|
| **MNIST** | 70,000 | 10 | 28 × 28 px | Handwritten digits (0-9) |
| **CIFAR-10** | 60,000 | 10 | 32 × 32 px | Natural images (airplane, car, bird, etc.) |

> **Notebook file:** `CNN_vs_AlexNet.ipynb`  
> **Goal:** Demonstrate how architecture depth, parameter count, and data augmentation influence accuracy, training speed, and per-class performance.

---

## 🚩 Quick Results Summary

### Performance Comparison (Default Training Run)

| Dataset | Model | Test Accuracy* | Parameters | Training Time** |
|---------|-------|---------------|------------|-----------------|
| **MNIST** | Custom CNN | **≈ 99.3%** | ~85K | ~2 min |
|           | AlexNet-style | **≈ 99.1%** | ~2.3M | ~4 min |
| **CIFAR-10** | Custom CNN (+ aug.) | **≈ 82.0%** | ~1M | ~12 min |
|             | AlexNet-style | **≈ 78.5%** | ~2M+ | ~15 min |

\*Results may vary ±0.5% due to weight initialization and hardware differences  
\*\*Approximate times on NVIDIA T4 GPU

### Key Insights
- 🎯 **Efficiency Winner**: Custom CNN achieves better accuracy with fewer parameters
- ⚡ **Speed Advantage**: Lightweight architecture trains 2-3x faster
- 📈 **Augmentation Impact**: More beneficial for custom CNN on CIFAR-10
- 🔍 **Resolution Matters**: AlexNet's large kernels are overkill for 32px images

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| **1. Load & Preview Data** | Fetch MNIST & CIFAR-10 via `keras.datasets` and inspect sample images |
| **2. Data Preprocessing** | Normalize pixel values, reshape tensors, one-hot encode labels |
| **3. CNN Fundamentals Recap** | Quick review of convolution, pooling, activations, and dropout |
| **4. Custom CNN Architecture** | Build lightweight model with 3-4 conv blocks (~1M parameters) |
| **5. AlexNet Adaptation** | Modify AlexNet for 32×32 inputs with adjusted filters and strides |
| **6. Training Workflows** | Implementation with and without data augmentation (`ImageDataGenerator`) |
| **7. Performance Evaluation** | Accuracy/loss curves, confusion matrices, per-class analysis |
| **8. Results Discussion** | Analysis of when lean models outperform deeper networks |

---

## 🔧 Installation & Setup
### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/DariusCornescu/CNN_-_AlexNet
   cd your-repo
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

<details>
<summary>📦 requirements.txt (click to expand)</summary>

```txt
tensorflow>=2.11.0
numpy>=1.21.0
matplotlib>=3.6.0
pandas>=1.5.0
scikit-learn>=1.2.0
seaborn>=0.12.0
jupyter>=1.0.0
ipywidgets>=8.0.0
```

</details>

### Hardware Requirements
- **GPU Training**: ~15 minutes on NVIDIA T4, RTX 30-series
- **CPU Training**: ~40-60 minutes (not recommended for full training)
- **Memory**: 4GB RAM minimum, 8GB+ recommended

---

## 🚀 Running the Notebook

### Local Execution
```bash
jupyter notebook "CNN_vs_AlexNet.ipynb"
```

### Cloud Platforms
- **Google Colab**: Click the Colab badge above for one-click access
- **Kaggle Notebooks**: Upload and run directly
- **Azure ML**: Compatible with Azure Machine Learning Studio

### Execution Steps
1. Open the notebook in your preferred environment
2. Select **"Runtime → Run all"** or **"Kernel → Restart & Run All"**
3. Monitor progress: MNIST completes in ~2 minutes, CIFAR-10 in 10-15 minutes on GPU
4. Review results in output cells: training curves, confusion matrices, and performance summaries

---
## 📊 Detailed Analysis

### Architecture Comparison

#### Custom CNN Features
- **Lightweight Design**: 3-4 convolutional blocks
- **Efficient Parameters**: ~1M total parameters
- **Optimized for Small Images**: Kernel sizes adapted for 28×28 and 32×32 inputs
- **Modern Techniques**: Batch normalization, dropout regularization

#### AlexNet Adaptation Features
- **Classic Architecture**: Based on ImageNet-winning design
- **Scaled Down**: Modified for smaller input dimensions
- **Higher Capacity**: 2M+ parameters
- **Traditional Approach**: ReLU activations, max pooling

### Performance Insights

#### MNIST Results
- Both models achieve excellent performance (99%+ accuracy)
- Custom CNN slightly outperforms with fewer parameters
- Training converges faster for lightweight architecture
- Minimal benefit from data augmentation on simple dataset

#### CIFAR-10 Results
- Custom CNN shows superior generalization (82% vs 78.5%)
- Data augmentation provides significant boost for custom model
- AlexNet struggles with limited input resolution
- Per-class analysis reveals specific weaknesses (frog ↔ deer confusion)

---

## 📈 Reproducing Visualizations

The notebook includes several utility functions for generating publication-ready plots:

### Training Analysis
```python
# Plot training history
plot_history(history)       # Accuracy & loss vs. epochs

# Compare multiple models
compare_models([model1, model2], histories=[hist1, hist2])
```

### Classification Results
```python
# Confusion matrix heatmap
plot_confusion_matrix(cm, class_names=['cat', 'dog', ...])

# Per-class accuracy breakdown
per_class_bar(accuracy_dict, title="CIFAR-10 Per-Class Results")
```

### Architecture Visualization
```python
# Model summary visualization
visualize_architecture(model, save_path='artifacts/model_arch.png')

# Parameter comparison
plot_parameter_comparison(models, names)
```

Call these functions after each `model.fit()` or `model.evaluate()` step to regenerate all visualizations.

---

## 🔬 Key Findings & Takeaways

### Architectural Insights
1. **Input Resolution Matters**: AlexNet's large kernels (11×11, 5×5) are excessive for 32×32 pixel images
2. **Parameter Efficiency**: Fewer parameters can achieve better results with proper design
3. **Depth vs. Width**: Balanced architecture often outperforms very deep or very wide networks
4. **Modern vs. Classic**: Contemporary techniques (batch norm, dropout) improve older architectures

### Training Observations
1. **Data Augmentation Impact**: More beneficial for smaller models and complex datasets
2. **Convergence Speed**: Lightweight models train faster and more reliably
3. **Overfitting Patterns**: Larger models show more tendency to overfit on limited data
4. **Learning Rate Sensitivity**: Different architectures require different optimization strategies

### Practical Applications
1. **Resource-Constrained Environments**: Custom CNN ideal for mobile/edge deployment
2. **Rapid Prototyping**: Faster training enables quicker iteration cycles
3. **Transfer Learning**: Results inform architecture choices for similar image sizes
4. **Educational Value**: Clear demonstration of architecture design principles

---

## 🛠️ Extending the Project

### Suggested Enhancements

#### Additional Architectures
- [ ] VGG-style networks with varying depths
- [ ] ResNet implementation with skip connections
- [ ] MobileNet for ultra-efficient inference
- [ ] Vision Transformer (ViT) for comparison

#### Advanced Techniques
- [ ] Learning rate scheduling and warm-up
- [ ] Advanced data augmentation (CutMix, MixUp)
- [ ] Model ensembling strategies
- [ ] Quantization for deployment optimization

#### New Experiments
- [ ] Fashion-MNIST dataset comparison
- [ ] CIFAR-100 for increased complexity
- [ ] Custom dataset integration
- [ ] Cross-dataset transfer learning

#### Analysis Extensions
- [ ] Gradient visualization and analysis
- [ ] Feature map visualization
- [ ] Adversarial robustness testing
- [ ] Computational complexity analysis (FLOPs)

---

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or adding new features, your help is appreciated.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Update documentation as needed
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Contribution Guidelines
- Follow existing code style and conventions
- Add appropriate comments and docstrings
- Include tests for new functionality
- Update README if adding new features
- Ensure all existing tests pass

---

## 🙏 Acknowledgements

### Datasets
- **MNIST**: Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner (1998). *Gradient-based learning applied to document recognition*
- **CIFAR-10**: A. Krizhevsky (2009). *Learning multiple layers of features from tiny images*

### Frameworks & Tools
- **TensorFlow/Keras**: Google Brain Team and Keras contributors
- **NumPy**: NumPy development team
- **Matplotlib/Seaborn**: Visualization library contributors
- **Jupyter**: Project Jupyter contributors

### Inspiration
- **AlexNet**: A. Krizhevsky, I. Sutskever, and G. E. Hinton (2012). *ImageNet Classification with Deep Convolutional Neural Networks*
- **CNN Architectures**: Various research papers and open-source implementations

### Infrastructure
- **Google Colab**: Free GPU access for experimentation
- **Kaggle**: Dataset hosting and notebook platform
- **GitHub**: Version control and collaboration platform

---

## 📚 References & Further Reading

### Essential Papers
1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*
2. Krizhevsky, A., et al. (2012). ImageNet classification with deep convolutional neural networks. *NIPS*
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv*

### Tutorials & Resources
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/)
- [Deep Learning Book - Chapter 9](https://www.deeplearningbook.org/contents/convnets.html)

### Related Projects
- [PyTorch CNN Examples](https://github.com/pytorch/examples/tree/master/mnist)
- [Keras Application Zoo](https://keras.io/api/applications/)
- [Papers With Code - Image Classification](https://paperswithcode.com/task/image-classification)

---

## 📜 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

```
MIT License

Copyright (c) 2025 [Cornescu Darius]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---


> *Happy training! 🚀 Feel free to open issues or pull requests with improvements, suggestions, or questions. Your feedback helps make this project better for everyone.*

---

*Last updated: June 2025*
