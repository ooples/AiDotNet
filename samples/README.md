# AiDotNet Samples

This directory contains complete, runnable examples demonstrating the full range of AiDotNet capabilities. Each sample is a standalone .NET project that you can build and run.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ooples/AiDotNet.git
cd AiDotNet/samples

# Run any sample
cd getting-started/HelloWorld
dotnet run
```

---

## Sample Index

### Getting Started

| Sample | Description | Difficulty |
|--------|-------------|------------|
| [HelloWorld](getting-started/HelloWorld/) | Your first AiDotNet model - XOR neural network | Beginner |
| [BasicClassification](getting-started/BasicClassification/) | Iris flower classification with Random Forest | Beginner |
| [BasicRegression](getting-started/BasicRegression/) | House price prediction with linear regression | Beginner |

---

### Classification

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [SentimentAnalysis](classification/BinaryClassification/SentimentAnalysis/) | Movie review sentiment with Naive Bayes | Text preprocessing, TF-IDF |
| [SpamDetection](classification/BinaryClassification/SpamDetection/) | Email spam classifier | Binary classification, evaluation metrics |
| [IrisClassification](classification/MultiClassification/IrisClassification/) | Multi-class classification on Iris dataset | Cross-validation, confusion matrix |

---

### Regression

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [PricePrediction](regression/PricePrediction/) | House price prediction with ensemble methods | Feature engineering, Gradient Boosting |
| [DemandForecasting](regression/DemandForecasting/) | Product demand forecasting | Multiple regression, regularization |

---

### Clustering

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [CustomerSegmentation](clustering/CustomerSegmentation/) | Customer segmentation with K-Means | Silhouette score, cluster visualization |
| [AnomalyDetection](clustering/AnomalyDetection/) | Outlier detection with DBSCAN | Density-based clustering |

---

### Computer Vision

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [YOLOv8Detection](computer-vision/ObjectDetection/YOLOv8Detection/) | Real-time object detection with YOLOv8 | GPU acceleration, visualization |
| [ImageClassification](computer-vision/ImageClassification/) | Image classification with CNN | Transfer learning, data augmentation |
| [OCR](computer-vision/OCR/) | Text extraction from images | Scene text recognition |

---

### Natural Language Processing

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [TextClassification](nlp/TextClassification/) | News article categorization | Transformers, embeddings |
| [Embeddings](nlp/Embeddings/) | Generate text embeddings | Sentence transformers, similarity search |
| [BasicRAG](nlp/RAG/BasicRAG/) | Build a Q&A system with RAG | Vector store, retriever, generator |
| [GraphRAG](nlp/RAG/GraphRAG/) | Knowledge graph-enhanced RAG | Entity extraction, graph traversal |

---

### LLM Fine-tuning

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [LoRA](llm-fine-tuning/LoRA/) | Fine-tune with Low-Rank Adaptation | Parameter-efficient training |
| [QLoRA](llm-fine-tuning/QLoRA/) | 4-bit quantized LoRA fine-tuning | Memory-efficient, INT4 quantization |

---

### Audio Processing

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [Whisper](audio/SpeechRecognition/Whisper/) | Speech-to-text with Whisper | Multi-language, timestamps |
| [TextToSpeech](audio/TextToSpeech/) | Generate speech from text | VITS, multiple voices |

---

### Video Processing

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [VideoGeneration](video/VideoGeneration/) | Generate videos from text | Stable Video Diffusion |

---

### Reinforcement Learning

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [CartPole](reinforcement-learning/CartPole/) | Balance a pole with PPO | Policy gradient, reward shaping |
| [DeepQLearning](reinforcement-learning/DeepQLearning/) | DQN on classic control tasks | Experience replay, target network |

---

### Time Series

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [Forecasting](time-series/Forecasting/) | Sales forecasting with N-BEATS | Deep learning, multi-horizon |
| [AnomalyDetection](time-series/AnomalyDetection/) | Detect anomalies in time series | DeepANT, threshold detection |

---

### Advanced Topics

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [DDP](advanced/DistributedTraining/DDP/) | Distributed Data Parallel training | Multi-GPU, NCCL backend |
| [FSDP](advanced/DistributedTraining/FSDP/) | Fully Sharded Data Parallel | Memory-efficient, large models |
| [MetaLearning](advanced/MetaLearning/) | Few-shot learning with MAML | Task adaptation |
| [SelfSupervisedLearning](advanced/SelfSupervisedLearning/) | Pre-training with SimCLR | Contrastive learning |

---

### Deployment

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [ModelServing](deployment/ModelServing/) | Serve models via REST API | AiDotNet.Serving, batching |
| [Quantization](deployment/Quantization/) | Quantize models for inference | INT8, performance optimization |

---

### End-to-End Applications

| Sample | Description | Key Features |
|--------|-------------|--------------|
| [ImageClassificationApp](end-to-end/ImageClassificationApp/) | Full web app for image classification | ASP.NET Core, model serving |

---

## Running Samples

### Prerequisites

- .NET 8.0 SDK or later
- (Optional) NVIDIA GPU with CUDA for GPU-accelerated samples

### Build and Run

Each sample is a standalone project:

```bash
cd samples/<category>/<sample-name>
dotnet restore
dotnet run
```

### GPU Acceleration

Samples that support GPU will automatically use it if available. To force CPU:

```csharp
.ConfigureGpuAcceleration(new GpuAccelerationConfig { Enabled = false })
```

---

## Sample Project Structure

Each sample follows a consistent structure:

```
SampleName/
├── README.md           # Sample documentation
├── SampleName.csproj   # Project file with dependencies
├── Program.cs          # Main entry point
├── Data/               # Sample data (if needed)
└── Output/             # Generated outputs (gitignored)
```

---

## Contributing Samples

We welcome new sample contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

When creating a new sample:
1. Follow the existing directory structure
2. Include a README.md with clear instructions
3. Keep samples focused on one feature/concept
4. Include sample data or instructions to download it
5. Test that the sample runs successfully

---

## Need Help?

- [API Documentation](../docs/)
- [Main README](../README.md)
- [GitHub Issues](https://github.com/ooples/AiDotNet/issues)
- [GitHub Discussions](https://github.com/ooples/AiDotNet/discussions)
