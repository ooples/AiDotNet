# AiDotNet

<div align="center">

### The Most Comprehensive AI/ML Framework for .NET

**4,300+ implementations across 60+ feature categories - bringing cutting-edge AI to the .NET ecosystem**

[![Build Status](https://github.com/ooples/AiDotNet/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/ooples/AiDotNet/actions/workflows/sonarcloud.yml)
[![CodeQL](https://github.com/ooples/AiDotNet/security/code-scanning/badge.svg)](https://github.com/ooples/AiDotNet/security/code-scanning)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/59242b2fb53c4ffc871212d346de752f)](https://app.codacy.com/gh/ooples/AiDotNet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![NuGet](https://img.shields.io/nuget/v/AiDotNet.svg)](https://www.nuget.org/packages/AiDotNet/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Neural Networks](https://img.shields.io/badge/Neural_Networks-100+-blue)
![Classical ML](https://img.shields.io/badge/Classical_ML-106+-green)
![Computer Vision](https://img.shields.io/badge/Vision-50+-orange)
![Audio](https://img.shields.io/badge/Audio-90+-purple)
![Video](https://img.shields.io/badge/Video-34+-red)
![RL Agents](https://img.shields.io/badge/RL_Agents-80+-yellow)
![Diffusion](https://img.shields.io/badge/Diffusion-20+-pink)
![RAG](https://img.shields.io/badge/RAG-50+-cyan)
![LoRA](https://img.shields.io/badge/LoRA-37+-teal)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-ff6f00)
![Multi-GPU](https://img.shields.io/badge/Multi--GPU-DDP|FSDP|ZeRO-00bcd4)

[Getting Started](#installation) •
[Samples](https://github.com/ooples/AiDotNet/tree/master/samples) •
[Documentation](https://ooples.github.io/AiDotNet/) •
[API Reference](https://ooples.github.io/AiDotNet/api/) •
[Contributing](#contributing)

</div>

---

## Why AiDotNet?

| Feature | AiDotNet | TorchSharp | TensorFlow.NET | ML.NET | Accord.NET |
|---------|----------|------------|----------------|--------|------------|
| Neural Network Architectures | **100+** | 50+ | 30+ | ~10 | ~20 |
| Classical ML Algorithms | **106+** | None | None | ~30 | ~50 |
| Computer Vision Models | **50+** | Via PyTorch | Via TF | Limited | Limited |
| Audio Processing | **90+** | Limited | Limited | None | Basic |
| Reinforcement Learning | **80+ agents** | Manual | Limited | None | None |
| Diffusion Models | **20+** | Manual | None | None | None |
| LoRA Fine-tuning | **37 adapters** | Manual | None | None | None |
| RAG Components | **50+** | None | None | None | None |
| Distributed Training | **DDP, FSDP, ZeRO** | DDP only | MirroredStrategy | None | None |
| HuggingFace Integration | **Native** | Partial | Partial | None | None |
| GPU Acceleration | **CUDA, OpenCL** | Via LibTorch | Via TF Runtime | Limited | None |
| Pure .NET (No Runtime) | **Yes** | No (LibTorch) | No (TF Runtime) | Yes | Yes |
| Startup Time | **Fast** | Slow | Slow | Fast | Fast |
| Memory<T>/Span<T> Support | **Full** | Limited | Limited | Limited | None |

---

## Quick Start by Task

### What do you want to build?

| Task | Quick Link | Description |
|------|------------|-------------|
| **Classify data** | [Classification](#classification) | Binary, multi-class, image classification |
| **Predict values** | [Regression](#regression) | Price prediction, forecasting |
| **Group similar items** | [Clustering](#clustering) | Customer segmentation, anomaly detection |
| **Detect objects in images** | [Computer Vision](#computer-vision) | YOLO, DETR, Mask R-CNN |
| **Process audio/speech** | [Audio](#audio-processing) | Whisper, TTS, music generation |
| **Build a chatbot with RAG** | [RAG](#retrieval-augmented-generation) | Vector stores, retrievers, rerankers |
| **Fine-tune LLMs** | [LoRA Fine-tuning](#lora-fine-tuning) | QLoRA, DoRA, AdaLoRA |
| **Generate images** | [Diffusion Models](#diffusion-models) | Stable Diffusion, DALL-E 3 |
| **Train RL agents** | [Reinforcement Learning](#reinforcement-learning) | DQN, PPO, SAC, multi-agent |
| **Forecast time series** | [Time Series](#time-series) | ARIMA, Prophet, N-BEATS |
| **Scale training** | [Distributed Training](#distributed-training) | Multi-GPU, multi-node |

---

## Installation

```bash
dotnet add package AiDotNet
```

**Requirements:** .NET 10.0 / .NET 8.0+ (or .NET Framework 4.7.1+)

---

## Hello World Example

```csharp
using AiDotNet;

// Build and train a model in one fluent chain
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetwork<double>(inputSize: 4, hiddenSize: 16, outputSize: 3))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .ConfigurePreprocessing()  // Auto-applies StandardScaler + Imputer
    .BuildAsync(features, labels);

// Make predictions using the facade pattern
var prediction = result.Predict(newSample);
```

---

## Complete Feature Reference

### Neural Networks (100+ Architectures)

<details>
<summary><strong>Click to expand all neural network types</strong></summary>

#### Core Architectures
- `NeuralNetwork` - Feedforward networks
- `ConvolutionalNeuralNetwork` - CNNs for images
- `RecurrentNeuralNetwork` - RNNs for sequences
- `LSTMNeuralNetwork` - Long Short-Term Memory
- `GRUNeuralNetwork` - Gated Recurrent Units
- `TransformerArchitecture` - Attention-based models
- `ResNetNetwork` - Residual networks
- `DenseNetNetwork` - Densely connected networks
- `EfficientNetNetwork` - Efficient scaling
- `MobileNetV2Network`, `MobileNetV3Network` - Mobile-optimized

#### Generative Models
- `GenerativeAdversarialNetwork` (GAN)
- `DCGAN`, `ConditionalGAN`, `CycleGAN`
- `ProgressiveGAN`, `BigGAN`, `StyleGAN`
- `ACGAN`, `InfoGAN`, `Pix2Pix`
- `Autoencoder`, `VariationalAutoencoder`

#### Graph Neural Networks
- `GraphNeuralNetwork` (GNN)
- `GraphAttentionNetwork` (GAT)
- `GraphSAGENetwork`
- `GraphIsomorphismNetwork` (GIN)
- `GraphGenerationModel`

#### Specialized Architectures
- `CapsuleNetwork` - Capsule networks
- `SpikingNeuralNetwork` - Neuromorphic computing
- `QuantumNeuralNetwork` - Quantum ML
- `HyperbolicNeuralNetwork` - Hyperbolic geometry
- `MixtureOfExpertsNeuralNetwork` - MoE
- `NeuralTuringMachine` - NTM
- `DifferentiableNeuralComputer` - DNC
- `MemoryNetwork` - Memory-augmented
- `EchoStateNetwork` - Reservoir computing
- `LiquidStateMachine` - Liquid state machines
- `HopfieldNetwork` - Associative memory
- `RestrictedBoltzmannMachine` - RBM
- `DeepBeliefNetwork` - DBN
- `DeepBoltzmannMachine` - DBM
- `RadialBasisFunctionNetwork` - RBF
- `SelfOrganizingMap` - SOM
- `ExtremeLearningMachine` - ELM
- `NEAT` - Neuroevolution

#### Vision-Language Models
- `ClipNeuralNetwork` - CLIP
- `BlipNeuralNetwork`, `Blip2NeuralNetwork` - BLIP
- `LLaVANeuralNetwork` - LLaVA
- `FlamingoNeuralNetwork` - Flamingo
- `Gpt4VisionNeuralNetwork` - GPT-4V

#### Attention Mechanisms
- `AttentionNetwork`
- `FlashAttention` - Memory-efficient attention
- `MultiHeadAttention`

</details>

**Example:**
```csharp
var model = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new ConvolutionalNeuralNetwork<double>(
        inputChannels: 3,
        numClasses: 1000,
        architecture: CNNArchitecture.ResNet50))
    .ConfigureOptimizer(new AdamWOptimizer<double>(learningRate: 0.001))
    .ConfigureMixedPrecision()  // FP16 training
    .ConfigureGpuAcceleration()
    .BuildAsync(trainImages, trainLabels);
```

---

### Classification (28+ Algorithms)

<details>
<summary><strong>Click to expand all classification algorithms</strong></summary>

#### Ensemble Methods
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `AdaBoostClassifier`
- `ExtraTreesClassifier`
- `BaggingClassifier`
- `StackingClassifier`
- `VotingClassifier`

#### Naive Bayes
- `GaussianNaiveBayes`
- `MultinomialNaiveBayes`
- `BernoulliNaiveBayes`
- `ComplementNaiveBayes`
- `CategoricalNaiveBayes`

#### Linear Models
- `LogisticRegression`
- `RidgeClassifier`
- `SGDClassifier`
- `PassiveAggressiveClassifier`
- `PerceptronClassifier`

#### Support Vector Machines
- `LinearSupportVectorClassifier`
- `SupportVectorClassifier`

#### Discriminant Analysis
- `LinearDiscriminantAnalysis`
- `QuadraticDiscriminantAnalysis`

#### Multi-label/Multi-output
- `OneVsRestClassifier`
- `OneVsOneClassifier`
- `ClassifierChain`
- `MultiOutputClassifier`

#### Neighbors
- `KNeighborsClassifier`

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing(pipeline => pipeline
        .Add(new StandardScaler<double>())
        .Add(new SimpleImputer<double>()))
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);

Console.WriteLine($"Accuracy: {result.CrossValidationResult?.MeanAccuracy:P2}");
```

---

### Regression (41+ Algorithms)

<details>
<summary><strong>Click to expand all regression algorithms</strong></summary>

#### Linear Models
- `MultipleRegression`
- `PolynomialRegression`
- `RidgeRegression`
- `LassoRegression`
- `ElasticNetRegression`
- `BayesianRegression`
- `OrthogonalRegression`

#### Tree-Based
- `DecisionTreeRegression`
- `GradientBoostingRegression`
- `AdaBoostR2Regression`
- `ExtremelyRandomizedTreesRegression`
- `M5ModelTreeRegression`
- `ConditionalInferenceTreeRegression`

#### Kernel Methods
- `GaussianProcessRegression`
- `KernelRidgeRegression`
- `SupportVectorRegression`

#### Specialized
- `IsotonicRegression`
- `LocallyWeightedRegression`
- `PartialLeastSquaresRegression`
- `MultivariateRegression`
- `GeneralizedAdditiveModelRegression`

#### Probabilistic
- `PoissonRegression`
- `NegativeBinomialRegression`
- `QuantileRegression`
- `RobustRegression`

#### Neural Network Based
- `MultilayerPerceptronRegression`
- `NeuralNetworkRegression`

#### Optimization-Based
- `GeneticAlgorithmRegression`

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegression<double>(
        nEstimators: 200,
        maxDepth: 5,
        learningRate: 0.1))
    .ConfigureHyperparameterOptimizer(
        new BayesianOptimizer<double>(),
        searchSpace: new HyperparameterSearchSpace()
            .AddContinuous("learning_rate", 0.01, 0.3)
            .AddInteger("max_depth", 3, 10),
        trials: 50)
    .BuildAsync(features, targets);
```

---

### Clustering (20+ Algorithms)

<details>
<summary><strong>Click to expand all clustering algorithms</strong></summary>

#### Centroid-Based
- `KMeansClustering`
- `KMedoidsClustering`
- `MiniBatchKMeans`

#### Density-Based
- `DBSCAN`
- `HDBSCAN`
- `OPTICS`
- `MeanShift`
- `Denclue`

#### Hierarchical
- `AgglomerativeClustering`
- `BIRCH`
- `CURE`

#### Model-Based
- `GaussianMixtureClustering`
- `BayesianGaussianMixture`

#### Spectral
- `SpectralClustering`

#### Self-Organizing
- `GMeans`
- `XMeans`

#### Distance Metrics
- `EuclideanDistance`
- `ManhattanDistance`
- `CosineDistance`
- `MahalanobisDistance`
- `ChebyshevDistance`
- `MinkowskiDistance`

#### Validation Metrics
- `SilhouetteScore`
- `DaviesBouldinIndex`
- `CalinskiHarabaszIndex`
- `DunnIndex`
- `AdjustedRandIndex`

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new HDBSCAN<double>(minClusterSize: 15, minSamples: 5))
    .ConfigureAutoML(new ClusteringAutoML<double>())  // Auto-tune parameters
    .BuildAsync(features);

Console.WriteLine($"Clusters found: {result.Model.Labels.Distinct().Count()}");
Console.WriteLine($"Silhouette Score: {result.ClusteringMetrics?.SilhouetteScore:F3}");
```

---

### Computer Vision (50+ Models)

<details>
<summary><strong>Click to expand all computer vision capabilities</strong></summary>

#### Object Detection
- **YOLO Family**: YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- **Transformer-Based**: DETR, Deformable DETR, DINO
- **Two-Stage**: Faster R-CNN, Cascade R-CNN
- **Anchor-Free**: FCOS, CenterNet

#### Instance Segmentation
- Mask R-CNN
- YOLACT
- SOLOv2
- Segment Anything (SAM)

#### Semantic Segmentation
- DeepLabV3+
- UNet, UNet++
- PSPNet
- HRNet

#### Object Tracking
- SORT
- DeepSORT
- ByteTrack
- OC-SORT

#### OCR & Scene Text
- `SceneTextReader`
- Text detection + recognition
- Multi-language support

#### Pose Estimation
- OpenPose
- HRNet-Pose
- ViTPose

#### 3D Vision
- PointNet, PointNet++
- MeshCNN
- NeRF (Neural Radiance Fields)

</details>

**Example:**
```csharp
var builder = new AiModelBuilder<float, Tensor<float>, DetectionResult[]>()
    .ConfigureObjectDetector(new YOLOv8Detector<float>(
        modelSize: YOLOModelSize.Medium,
        confidenceThreshold: 0.5f))
    .ConfigureVisualization(new VisualizationOptions { DrawLabels = true });

var result = await builder.BuildAsync();
var detections = result.Model.Detect(image);

foreach (var det in detections)
    Console.WriteLine($"{det.Label}: {det.Confidence:P1} at {det.BoundingBox}");
```

---

### Audio Processing (90+ Models)

<details>
<summary><strong>Click to expand all audio capabilities</strong></summary>

#### Speech Recognition
- **Whisper**: whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large
- **Wav2Vec2**: Multiple languages
- **HuBERT**
- **Conformer**

#### Text-to-Speech
- VITS
- FastSpeech2
- Tacotron2
- XTTS (multi-speaker, cross-lingual)

#### Music Generation
- MusicGen
- AudioGen
- Riffusion

#### Audio Classification
- Audio event detection
- Music genre classification
- Environmental sound classification

#### Speaker Analysis
- Speaker identification
- Speaker verification
- Speaker diarization

#### Audio Enhancement
- Noise reduction
- Echo cancellation
- Speech enhancement

#### Source Separation
- Vocals/instruments separation
- Multi-track separation

#### Voice Activity Detection
- WebRTC VAD
- Silero VAD

#### Emotion Recognition
- Speech emotion classification

#### Music Analysis
- Beat detection
- Chord recognition
- Key detection
- Tempo estimation

</details>

**Example:**
```csharp
// Speech-to-Text with Whisper
var whisper = new WhisperModel<float>(WhisperModelSize.Medium, language: "en");
var transcription = await whisper.TranscribeAsync(audioFile);
Console.WriteLine(transcription.Text);

// Text-to-Speech
var tts = new VITSModel<float>(voice: "en-US-female");
var audio = await tts.SynthesizeAsync("Hello, world!");
await audio.SaveAsync("output.wav");
```

---

### Video Processing (34+ Models)

<details>
<summary><strong>Click to expand all video capabilities</strong></summary>

#### Video Generation
- Stable Video Diffusion
- AnimateDiff
- VideoCrafter
- Text2Video

#### Action Recognition
- SlowFast
- TimeSformer
- Video Swin Transformer

#### Video Understanding
- CLIP4Clip
- Video captioning

#### Optical Flow
- RAFT
- FlowNet

#### Video Object Detection
- Video object tracking
- Multi-object tracking

</details>

**Example:**
```csharp
var videoGen = new StableVideoDiffusion<float>();
var video = await videoGen.GenerateAsync(
    prompt: "A cat playing piano",
    numFrames: 24,
    fps: 8);
await video.SaveAsync("output.mp4");
```

---

### Reinforcement Learning (80+ Agents)

<details>
<summary><strong>Click to expand all RL agents</strong></summary>

#### Value-Based
- `DQNAgent` - Deep Q-Network
- `DoubleDQNAgent` - Double DQN
- `DuelingDQNAgent` - Dueling architecture
- `RainbowDQNAgent` - Rainbow (all improvements)
- `QLearningAgent`, `DoubleQLearningAgent`
- `SARSAAgent`, `ExpectedSARSAAgent`
- `NStepQLearningAgent`, `NStepSARSAAgent`

#### Policy Gradient
- `REINFORCEAgent`
- `A2CAgent` - Advantage Actor-Critic
- `A3CAgent` - Asynchronous A3C
- `PPOAgent` - Proximal Policy Optimization

#### Actor-Critic
- `DDPGAgent` - Deep Deterministic PG
- `TD3Agent` - Twin Delayed DDPG
- `SACAgent` - Soft Actor-Critic

#### Model-Based
- `DreamerAgent` - World models
- `MuZeroAgent` - MuZero
- `DynaQAgent`, `DynaQPlusAgent`
- `PrioritizedSweepingAgent`

#### Multi-Agent
- `MADDPGAgent` - Multi-Agent DDPG
- `QMIXAgent` - QMIX

#### Offline RL
- `CQLAgent` - Conservative Q-Learning
- `IQLAgent` - Implicit Q-Learning

#### Monte Carlo
- `MonteCarloExploringStartsAgent`
- `FirstVisitMonteCarloAgent`
- `EveryVisitMonteCarloAgent`
- `OffPolicyMonteCarloAgent`

#### Planning
- `MCTSNode` - Monte Carlo Tree Search
- `PolicyIterationAgent`
- `ModifiedPolicyIterationAgent`

#### Bandits
- `EpsilonGreedyBanditAgent`
- `UCBBanditAgent`
- `ThompsonSamplingAgent`
- `GradientBanditAgent`

#### Sequence Models
- `DecisionTransformerAgent`

</details>

**Example:**
```csharp
var env = new CartPoleEnvironment();
var agent = new PPOAgent<double>(
    stateSize: env.ObservationSpace,
    actionSize: env.ActionSpace,
    hiddenSize: 64,
    learningRate: 3e-4);

// Training loop
for (int episode = 0; episode < 1000; episode++)
{
    var state = env.Reset();
    double totalReward = 0;

    while (!env.IsDone)
    {
        var action = agent.SelectAction(state);
        var (nextState, reward, done) = env.Step(action);
        agent.Store(state, action, reward, nextState, done);
        state = nextState;
        totalReward += reward;
    }

    agent.Train();
    Console.WriteLine($"Episode {episode}: Reward = {totalReward}");
}
```

---

### Time Series (30+ Models)

<details>
<summary><strong>Click to expand all time series models</strong></summary>

#### Classical Statistical
- `ARModel` - Autoregressive
- `MAModel` - Moving Average
- `ARMAModel` - ARMA
- `ARIMAModel` - ARIMA
- `SARIMAModel` - Seasonal ARIMA
- `ARIMAXModel` - ARIMAX with exogenous variables
- `GARCHModel` - Volatility modeling
- `VARModel`, `VARMAModel` - Vector models

#### Exponential Smoothing
- `ExponentialSmoothingModel`
- `HoltWintersModel`
- `TBATSModel`

#### Deep Learning
- `NBEATSModel` - N-BEATS
- `NHiTSModel` - N-HiTS
- `DeepARModel` - DeepAR
- `TemporalFusionTransformer`
- `InformerModel` - Informer
- `AutoformerModel` - Autoformer
- `ChronosFoundationModel` - Chronos

#### Anomaly Detection
- `DeepANT`
- `LSTMVAE`
- `TimeSeriesIsolationForest`

#### Specialized
- `ProphetModel` - Facebook Prophet
- `StateSpaceModel`
- `BayesianStructuralTimeSeriesModel`
- `SpectralAnalysisModel`

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NBEATSModel<double>(
        stackTypes: new[] { StackType.Trend, StackType.Seasonality, StackType.Generic },
        horizonSize: 24))
    .ConfigureTrainingPipeline(new TrainingPipelineConfiguration<double>
    {
        EarlyStopping = new EarlyStoppingConfig { Patience = 10 }
    })
    .BuildAsync(historicalData);

var forecast = result.Model.Forecast(steps: 24);
```

---

### Retrieval-Augmented Generation (50+ Components)

<details>
<summary><strong>Click to expand all RAG components</strong></summary>

#### Vector Stores
- In-memory vector store
- FAISS integration
- Pinecone, Weaviate, Qdrant adapters
- Chroma, Milvus support

#### Embedding Models
- OpenAI embeddings
- HuggingFace embeddings
- Sentence Transformers
- BGE, ColBERT

#### Retrievers
- Dense retriever
- Sparse retriever (BM25)
- Hybrid retriever
- Multi-query retriever
- Self-query retriever

#### Rerankers
- Cross-encoder reranker
- ColBERT reranker
- Cohere reranker

#### Document Processing
- Text splitters (recursive, semantic)
- Document loaders
- Chunking strategies

#### Graph RAG
- Knowledge graph construction
- Entity extraction
- Relation extraction
- Graph traversal retrieval

#### Query Processing
- Query expansion
- Query rewriting
- HyDE (Hypothetical Document Embeddings)

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<float, string, string>()
    .ConfigureRetrievalAugmentedGeneration(
        retriever: new HybridRetriever<float>(
            denseRetriever: new DenseRetriever<float>(embeddingModel),
            sparseRetriever: new BM25Retriever(),
            alpha: 0.7f),
        reranker: new CrossEncoderReranker<float>(),
        generator: new LLMGenerator<float>(llmClient),
        queryProcessors: new[] { new QueryExpander() })
    .BuildAsync();

var answer = await result.Model.QueryAsync("What is the capital of France?", documents);
```

---

### LoRA Fine-tuning (37+ Adapters)

<details>
<summary><strong>Click to expand all LoRA variants</strong></summary>

#### Standard LoRA
- `LoRAAdapter` - Original LoRA
- `LoRAPlusAdapter` - LoRA+

#### Quantized
- `QLoRAAdapter` - 4-bit quantization
- `QALoRAAdapter` - Quantization-aware

#### Memory Efficient
- `DoRAAdapter` - Weight-Decomposed
- `VeRAAdapter` - Very efficient
- `NOLAAdapter` - Noise-optimized

#### Rank Adaptation
- `AdaLoRAAdapter` - Adaptive rank
- `DyLoRAAdapter` - Dynamic rank
- `ReLoRAAdapter` - Recursive

#### Specialized
- `LoHaAdapter` - Hadamard product
- `LoKrAdapter` - Kronecker product
- `MoRAAdapter` - Mixture of ranks
- `LongLoRAAdapter` - Long context
- `GraphConvolutionalLoRAAdapter` - For GNNs

#### Advanced
- `PiSSAAdapter` - Principal singular values
- `FloraAdapter` - Floating-point
- `DeltaLoRAAdapter` - Delta updates
- `LoftQAdapter` - Quantization-aware init
- `ChainLoRAAdapter` - Chained adapters

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<float, string, string>()
    .ConfigureLoRA(new QLoRAConfiguration<float>
    {
        Rank = 16,
        Alpha = 32,
        TargetModules = new[] { "q_proj", "v_proj", "k_proj", "o_proj" },
        QuantizationBits = 4,
        UseDoubleQuantization = true
    })
    .ConfigureFineTuning(new FineTuningConfiguration<float>
    {
        BaseModel = "meta-llama/Llama-2-7b-hf",
        TrainingArguments = new TrainingArguments
        {
            NumEpochs = 3,
            BatchSize = 4,
            GradientAccumulationSteps = 8
        }
    })
    .BuildAsync(trainingData);
```

---

### Diffusion Models (20+ Models)

<details>
<summary><strong>Click to expand all diffusion models</strong></summary>

#### Image Generation
- `StableDiffusionModel`
- `SDXLModel` - Stable Diffusion XL
- `DallE3Model` - DALL-E 3
- `PixArtModel` - PixArt

#### Image Editing
- `ControlNetModel`
- `IPAdapterModel`
- Inpainting, Outpainting

#### Audio Generation
- `AudioLDMModel`, `AudioLDM2Model`
- `MusicGenModel`
- `RiffusionModel`
- `DiffWaveModel`

#### Video Generation
- `StableVideoDiffusion`
- `AnimateDiffModel`
- `VideoCrafterModel`

#### 3D Generation
- `DreamFusionModel`
- `MVDreamModel`
- `ShapEModel`
- `PointEModel`
- `Zero123Model`

#### Architectures
- `DDPMModel` - Denoising Diffusion
- `ConsistencyModel` - Fast inference
- Various schedulers (DDIM, PNDM, etc.)

</details>

**Example:**
```csharp
var diffusion = new SDXLModel<float>();
var image = await diffusion.GenerateAsync(
    prompt: "A photorealistic cat astronaut on Mars",
    negativePrompt: "blurry, low quality",
    width: 1024,
    height: 1024,
    numInferenceSteps: 30,
    guidanceScale: 7.5f);

await image.SaveAsync("output.png");
```

---

### Distributed Training

<details>
<summary><strong>Click to expand distributed training options</strong></summary>

#### Data Parallelism
- `DDPModel` - Distributed Data Parallel
- `DDPOptimizer`

#### Model Parallelism
- `PipelineParallelModel`
- `TensorParallelModel`

#### Fully Sharded
- `FSDPModel` - Fully Sharded Data Parallel
- `ZeROOptimizer` (Stage 1, 2, 3)
- `HybridShardedModel`

#### Communication
- NCCL backend (NVIDIA)
- Gloo backend
- MPI backend

#### Optimization
- `GradientCompressionOptimizer`
- `AsyncSGDOptimizer`
- `LocalSGDOptimizer`
- `ElasticOptimizer` - Fault-tolerant

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(largeModel)
    .ConfigureDistributedTraining(
        strategy: DistributedStrategy.FSDP,
        backend: new NCCLCommunicationBackend(),
        configuration: new FSDPConfiguration
        {
            ShardingStrategy = ShardingStrategy.FullShard,
            MixedPrecision = true,
            ActivationCheckpointing = true
        })
    .ConfigureGpuAcceleration(new GpuAccelerationConfig { DeviceIds = new[] { 0, 1, 2, 3 } })
    .BuildAsync(trainData);
```

---

### Meta-Learning (18+ Algorithms)

<details>
<summary><strong>Click to expand meta-learning methods</strong></summary>

#### Optimization-Based
- `MAMLAlgorithm` - Model-Agnostic Meta-Learning
- `iMAMLAlgorithm` - Implicit MAML
- `ReptileAlgorithm`
- `MetaSGDAlgorithm`
- `ANILAlgorithm` - Almost No Inner Loop
- `BOILAlgorithm` - Body Only Inner Loop

#### Metric-Based
- `ProtoNetsAlgorithm` - Prototypical Networks
- `MatchingNetworksAlgorithm`
- `RelationNetworkAlgorithm`

#### Memory-Based
- `MANNAlgorithm` - Memory-Augmented NN
- `NTMAlgorithm` - Neural Turing Machine

#### Hybrid
- `LEOAlgorithm` - Latent Embedding Optimization
- `CNAPAlgorithm` - Conditional Neural Adaptive Processes
- `TADAMAlgorithm` - Task-Dependent Adaptive Metric
- `MetaOptNetAlgorithm`
- `GNNMetaAlgorithm` - Graph-based

</details>

**Example:**
```csharp
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureMetaLearning(new MAMLAlgorithm<float>(
        innerLearningRate: 0.01f,
        outerLearningRate: 0.001f,
        innerSteps: 5))
    .BuildAsync(metaTrainTasks);

// Few-shot learning on new task
var adapted = result.MetaLearner.Adapt(supportSet, numSteps: 10);
var predictions = adapted.Predict(querySet);
```

---

### Self-Supervised Learning (10+ Methods)

<details>
<summary><strong>Click to expand SSL methods</strong></summary>

#### Contrastive
- `SimCLR`
- `MoCo` - Momentum Contrast
- `BYOL` - Bootstrap Your Own Latent
- `BarlowTwins`
- `SwAV` - Swapping Assignments

#### Masked
- `MAE` - Masked Autoencoder
- `BEiT`

#### Distillation
- `DINO` - Self-Distillation
- `iBOT`

#### Evaluation
- Linear probing
- KNN evaluation
- Transfer benchmarks

</details>

---

### Optimizers (42+)

<details>
<summary><strong>Click to expand all optimizers</strong></summary>

#### Adaptive Learning Rate
- `AdamOptimizer`, `AdamWOptimizer`
- `AdagradOptimizer`
- `AdaDeltaOptimizer`
- `AdaMaxOptimizer`
- `AMSGradOptimizer`
- `NadamOptimizer`

#### Momentum-Based
- `MomentumOptimizer`
- `NesterovAcceleratedGradientOptimizer`

#### Second-Order
- `BFGSOptimizer`, `LBFGSOptimizer`
- `NewtonMethodOptimizer`
- `LevenbergMarquardtOptimizer`
- `ConjugateGradientOptimizer`

#### Large-Scale
- `LAMBOptimizer` - Layer-wise Adaptive
- `LARSOptimizer` - Layer-wise Rate Scaling
- `LionOptimizer` - Evolved Sign Momentum

#### Regularized
- `FTRLOptimizer` - Follow-The-Regularized-Leader

#### Evolutionary
- `GeneticAlgorithmOptimizer`
- `ParticleSwarmOptimizer`
- `DifferentialEvolutionOptimizer`
- `CMAESOptimizer` - Covariance Matrix Adaptation
- `AntColonyOptimizer`

#### Specialized
- `BayesianOptimizer` - Hyperparameter tuning
- `NelderMeadOptimizer` - Simplex method
- `CoordinateDescentOptimizer`
- `ADMMOptimizer` - Alternating Direction

</details>

---

### Loss Functions (37+)

- **Classification**: CrossEntropy, BinaryCrossEntropy, FocalLoss, LabelSmoothing
- **Regression**: MSE, MAE, Huber, LogCosh, Quantile
- **Segmentation**: Dice, IoU, Tversky, Lovász
- **Metric Learning**: Triplet, Contrastive, ArcFace, CosFace, Circle
- **Generative**: Wasserstein, Hinge, LSGAN
- **Multi-task**: Uncertainty-weighted, GradNorm

---

### Additional Features

#### AutoML
- Hyperparameter optimization (Bayesian, Random, Grid)
- Neural Architecture Search (NAS)
- Feature selection
- Model selection

#### Federated Learning
- FedAvg, FedProx, FedNova
- Secure aggregation
- Differential privacy
- Client selection strategies

#### Tokenization (HuggingFace Compatible)
- BPE, WordPiece, Unigram, SentencePiece
- Pre-trained tokenizer loading
- Custom tokenizer training

#### Model Compression
- Pruning (magnitude, structured)
- Quantization (INT8, FP16, INT4)
- Knowledge distillation
- Low-rank factorization

#### Uncertainty Quantification
- Monte Carlo Dropout
- Deep Ensembles
- Bayesian Neural Networks
- Conformal Prediction

#### Data Augmentation
- **Image**: Flip, rotate, crop, color jitter, MixUp, CutMix, AutoAugment
- **Text**: Synonym replacement, back-translation, EDA
- **Audio**: Time stretch, pitch shift, noise injection
- **Tabular**: SMOTE, ADASYN

#### Experiment Tracking
- Experiment management
- Metric logging
- Checkpoint management
- Model registry

#### Prompt Engineering
- Prompt templates
- Chain-of-thought
- Few-shot learning
- Prompt optimization

---

## Samples

See the [samples/](samples/) directory for complete, runnable examples:

| Category | Sample | Description |
|----------|--------|-------------|
| Getting Started | [HelloWorld](samples/getting-started/HelloWorld/) | Your first AiDotNet model |
| Classification | [SentimentAnalysis](samples/classification/SentimentAnalysis/) | Text sentiment with NaiveBayes |
| Computer Vision | [ObjectDetection](samples/computer-vision/ObjectDetection/) | YOLOv8 object detection |
| Audio | [SpeechRecognition](samples/audio/SpeechRecognition/) | Whisper transcription |
| RAG | [BasicRAG](samples/nlp/RAG/BasicRAG/) | Build a Q&A system |
| RL | [CartPole](samples/reinforcement-learning/CartPole/) | Train a PPO agent |

---

## API Reference

Full API documentation is available at [ooples.github.io/AiDotNet](https://ooples.github.io/AiDotNet/).

Key namespaces:
- `AiDotNet` - Core builder and result types
- `AiDotNet.NeuralNetworks` - All neural network architectures
- `AiDotNet.Classification` - Classification algorithms
- `AiDotNet.Regression` - Regression algorithms
- `AiDotNet.Clustering` - Clustering algorithms
- `AiDotNet.ComputerVision` - Vision models
- `AiDotNet.Audio` - Audio processing
- `AiDotNet.ReinforcementLearning` - RL agents
- `AiDotNet.RetrievalAugmentedGeneration` - RAG components
- `AiDotNet.LoRA` - Fine-tuning adapters
- `AiDotNet.Diffusion` - Diffusion models
- `AiDotNet.DistributedTraining` - Distributed strategies
- `AiDotNet.MetaLearning` - Meta-learning algorithms
- `AiDotNet.TimeSeries` - Time series models
- `AiDotNet.Optimizers` - Optimization algorithms

---

## Platform Support

| Platform | Status |
|----------|--------|
| Windows | ✅ Full support |
| Linux | ✅ Full support |
| macOS | ✅ Full support |
| .NET 10.0 | ✅ Primary target |
| .NET 8.0+ | ✅ Supported |
| .NET Framework 4.7.1+ | ✅ Supported |

### GPU Acceleration

| Backend | Status |
|---------|--------|
| CUDA (NVIDIA) | ✅ Full support |
| OpenCL | ✅ Full support |
| Metal (Apple) | 🚧 Coming soon |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we especially welcome help:
- Additional model implementations
- Performance optimizations
- Documentation and examples
- Bug fixes and testing

---

## Community

- **Issues**: [Report bugs or request features](https://github.com/ooples/AiDotNet/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/ooples/AiDotNet/discussions)
- **Security**: [Report vulnerabilities](SECURITY.md)

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with care for the .NET AI/ML community**

[⭐ Star us on GitHub](https://github.com/ooples/AiDotNet) • [📦 NuGet Package](https://www.nuget.org/packages/AiDotNet/)

</div>
