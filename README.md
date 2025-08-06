# Deep Learning Portfolio: Neural Networks and Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive collection of deep learning projects demonstrating expertise in neural networks, computer vision, and natural language processing. This portfolio showcases implementations from foundational concepts to state-of-the-art architectures.

## 🚀 Project Overview

This repository contains six distinct projects that demonstrate a complete understanding of modern deep learning techniques:

### 🧠 **Foundational Neural Networks**
- **MNIST Handwritten Digit Recognition** - Classical neural network implementation
- **PySmorch Framework** - Custom neural network library built from scratch

### 🔍 **Computer Vision**
- **Binary Image Classification** - Transfer learning with ResNet on Ants vs Bees dataset
- **Pet Image Segmentation** - Semantic segmentation using Fully Convolutional Networks

### 📝 **Natural Language Processing**
- **LSTM Sentiment Analysis** - Bidirectional LSTM for movie review classification
- **Self-Attention & Text Generation** - Transformer mechanisms and LLM inference

---

## 📋 Projects Detail

### 1. 🔢 [MNIST Handwritten Digit Recognition](./MNIST-NN.ipynb)
**Classic neural network for digit classification**

- **Architecture**: 3-layer MLP (784→50→50→10)
- **Dataset**: MNIST (70k handwritten digits)
- **Techniques**: ReLU activation, SGD optimization
- **Results**: 95.28% test accuracy
- **Skills**: Neural network fundamentals, PyTorch basics

```python
# Key Features
✓ Data visualization and preprocessing
✓ Multi-layer perceptron implementation
✓ Training curves and performance analysis
✓ GPU acceleration support
```

### 2. ⚙️ [PySmorch: Custom Neural Network Framework](./PySmorch_Framework_and_CNN_Architectures.ipynb)
**Complete neural network library implementation from scratch**

- **Components**: Linear layers, ReLU, Bias, Cross-entropy loss
- **Features**: Automatic differentiation, gradient checking
- **Validation**: Numerical gradient verification
- **Comparison**: PyTorch CNN architectures (SimpleConvNet vs ResNet32)
- **Skills**: Deep learning fundamentals, mathematical implementation

```python
# Framework Components
✓ Forward/backward propagation from scratch
✓ Gradient checking utilities
✓ Layer-wise implementation (Linear, ReLU, Bias, CrossEntropy)
✓ Modern CNN architecture comparison
✓ CIFAR-10 classification experiments
```

### 3. 🐜🐝 [Binary Classification: Ants vs Bees](./Binary_Classification_Ants_vs_Bees.ipynb)
**Transfer learning comparison study**

- **Model**: ResNet-18 with ImageNet pretraining
- **Dataset**: Hymenoptera (240 images, 2 classes)
- **Strategies**: Training from scratch vs Transfer learning vs Fine-tuning
- **Results**: Fine-tuning > Transfer learning > From scratch
- **Skills**: Transfer learning, small dataset optimization

```python
# Training Strategies Compared
✓ Training from scratch (baseline)
✓ Transfer learning (frozen features)
✓ Fine-tuning (all parameters)
✓ Data augmentation techniques
✓ Performance visualization
```

### 4. 🐱🐶 [Pet Image Segmentation](./Pet_Image_Segmentation_FCN.ipynb)
**Semantic segmentation with Fully Convolutional Networks**

- **Architecture**: FCN with ResNet-50 backbone
- **Dataset**: Oxford-IIIT Pet Dataset (37 breeds, pixel-level annotations)
- **Task**: Foreground/background segmentation
- **Metrics**: IoU (Intersection over Union)
- **Skills**: Segmentation, advanced computer vision

```python
# Segmentation Pipeline
✓ Pixel-level annotation handling
✓ FCN architecture implementation
✓ IoU metric computation
✓ Training vs validation comparisons
✓ Visual segmentation results
```

### 5. 🎬 [LSTM Sentiment Analysis](./LSTM_Sentiment_Analysis_IMDB.ipynb)
**Bidirectional LSTM for movie review classification**

- **Architecture**: 2-layer Bidirectional LSTM (300 hidden units)
- **Dataset**: IMDB Movie Reviews (50k reviews)
- **Features**: Packed sequences, dropout regularization
- **Preprocessing**: Custom tokenization, vocabulary building
- **Skills**: RNN architectures, sequence modeling, NLP

```python
# NLP Pipeline
✓ Text tokenization and preprocessing
✓ Vocabulary building with UNK/PAD tokens
✓ Bidirectional LSTM implementation
✓ Packed sequences for efficiency
✓ ~10.1M parameters optimization
```

### 6. 🤖 [Self-Attention & Text Generation](./Self_Attention_and_Text_Generation.ipynb)
**Transformer mechanisms and large language models**

- **Part I**: Self-attention implementation (single-head & multi-head)
- **Part II**: LLM text generation with 4-bit quantization
- **Features**: Attention visualization, memory optimization
- **Technologies**: Transformers, BitsAndBytes, GPU acceleration
- **Skills**: Attention mechanisms, modern NLP, model optimization

```python
# Advanced NLP Techniques
✓ Single-head and multi-head attention
✓ Scaled dot-product attention mathematics
✓ Large language model inference
✓ 4-bit quantization for memory efficiency
✓ Interactive text generation
```

---

## 🛠️ Technical Stack

### **Core Frameworks**
- **PyTorch**: Deep learning framework for all implementations
- **torchvision**: Computer vision utilities and pretrained models
- **transformers**: State-of-the-art NLP models and tokenizers

### **Specialized Libraries**
- **segmentation-models-pytorch**: Advanced segmentation architectures
- **datasets**: Efficient dataset loading and preprocessing
- **BitsAndBytes**: Model quantization and memory optimization

### **Data & Visualization**
- **numpy, pandas**: Data manipulation and analysis
- **matplotlib, seaborn**: Visualization and plotting
- **PIL**: Image processing utilities

### **Development Tools**
- **Jupyter**: Interactive development environment
- **CUDA**: GPU acceleration support
- **Git**: Version control and collaboration

---

## 📊 Key Achievements

### **Performance Metrics**
- 🎯 **95.28% accuracy** on MNIST digit classification
- 🔄 **Successful gradient checking** on custom neural network framework
- 📈 **Superior transfer learning performance** on small datasets
- 🎨 **High-quality semantic segmentation** with IoU metrics
- 💬 **Effective sentiment classification** on large text corpus
- ⚡ **Memory-efficient LLM inference** with 4-bit quantization

### **Technical Expertise Demonstrated**
- ✅ **Mathematical Implementation**: Built neural networks from scratch with proper gradients
- ✅ **Computer Vision**: Transfer learning, segmentation, data augmentation
- ✅ **Natural Language Processing**: RNN architectures, attention mechanisms, LLMs
- ✅ **Model Optimization**: Quantization, memory management, GPU utilization
- ✅ **Best Practices**: Code organization, documentation, reproducibility

---

## 🚀 Getting Started

### **Prerequisites**
```bash
Python 3.8+
CUDA-compatible GPU (recommended)
```

### **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd deep-learning-portfolio

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets segmentation-models-pytorch
pip install jupyter matplotlib seaborn pandas numpy
```

### **Usage**
```bash
# Launch Jupyter Notebook
jupyter notebook

# Open any project notebook:
# - MNIST-NN.ipynb
# - PySmorch_Framework_and_CNN_Architectures.ipynb
# - Binary_Classification_Ants_vs_Bees.ipynb
# - Pet_Image_Segmentation_FCN.ipynb
# - LSTM_Sentiment_Analysis_IMDB.ipynb
# - Self_Attention_and_Text_Generation.ipynb
```

---

## 📁 Project Structure

```
deep-learning-portfolio/
├── README.md                                    # This file
├── MNIST-NN.ipynb                             # Neural network fundamentals
├── PySmorch_Framework_and_CNN_Architectures.ipynb  # Custom framework + CNNs
├── Binary_Classification_Ants_vs_Bees.ipynb   # Transfer learning study
├── Pet_Image_Segmentation_FCN.ipynb           # Semantic segmentation
├── LSTM_Sentiment_Analysis_IMDB.ipynb         # Sequence modeling
└── Self_Attention_and_Text_Generation.ipynb   # Transformer & LLMs
```

---

## 🎓 Learning Outcomes

This portfolio demonstrates comprehensive knowledge across the deep learning spectrum:

### **Foundational Concepts**
- Neural network mathematics and implementation
- Gradient computation and backpropagation
- Loss functions and optimization algorithms

### **Computer Vision**
- Convolutional neural networks and architectures
- Transfer learning strategies and fine-tuning
- Semantic segmentation and pixel-level predictions
- Data augmentation and regularization techniques

### **Natural Language Processing**
- Recurrent neural networks and LSTM architectures
- Attention mechanisms and transformer models
- Text preprocessing and tokenization
- Large language model deployment and optimization

### **Advanced Techniques**
- Model quantization and memory optimization
- GPU acceleration and parallel computing
- Hyperparameter tuning and model selection
- Performance evaluation and visualization

---

## 🤝 Contributing

This portfolio represents a comprehensive study of deep learning techniques. Each notebook is self-contained and includes detailed explanations, making it suitable for:

- 📚 **Educational purposes** and learning deep learning concepts
- 💼 **Technical interviews** and portfolio demonstrations  
- 🔬 **Research baselines** and experimental starting points
- 👥 **Collaborative projects** and team learning

---

## 📞 Contact

**Yanjie Chen**
- 📧 Email: [yc4594@columbia.edu]
- 💼 LinkedIn: [http://www.linkedin.com/in/yanjiechen]
- 🐱 GitHub: [YanjieChen9117]

---

*This portfolio showcases practical implementations of cutting-edge deep learning techniques, from foundational neural networks to state-of-the-art transformer architectures. Each project demonstrates both theoretical understanding and practical implementation skills essential for modern AI development.*
