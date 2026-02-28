# AiDotNet Meta-Learning Framework

Comprehensive guide to the meta-learning suite in AiDotNet. This framework provides 100+ algorithms spanning 12 categories, a full episodic data infrastructure, and synthetic benchmarks for rapid prototyping.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Algorithm Categories](#algorithm-categories)
  - [Optimization-Based](#optimization-based)
  - [Metric-Based](#metric-based)
  - [Memory-Based](#memory-based)
  - [Hybrid / Advanced](#hybrid--advanced)
  - [Neural Processes](#neural-processes)
  - [Foundation Model Era](#foundation-model-era)
  - [Bayesian Extensions](#bayesian-extensions)
  - [Cross-Domain Few-Shot](#cross-domain-few-shot)
  - [Meta-Reinforcement Learning](#meta-reinforcement-learning)
  - [Continual / Online Meta-Learning](#continual--online-meta-learning)
  - [Task Augmentation / Sampling](#task-augmentation--sampling)
  - [Transductive Methods](#transductive-methods)
  - [Hypernetwork Methods](#hypernetwork-methods)
- [Data Infrastructure](#data-infrastructure)
  - [IEpisode](#iepisode)
  - [IMetaDataset](#imetadataset)
  - [ITaskSampler](#itasksampler)
  - [Task Samplers](#task-samplers)
  - [Synthetic Datasets](#synthetic-datasets)
  - [Episode Caching](#episode-caching)
- [Choosing an Algorithm](#choosing-an-algorithm)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

```csharp
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;

// 1. Create a model
var model = new YourModel(inputDim: 3);

// 2. Configure algorithm options
var options = new MAMLOptions<double, Matrix<double>, Vector<double>>(model)
{
    InnerLearningRate = 0.01,
    OuterLearningRate = 0.001,
    InnerSteps = 5
};

// 3. Create the algorithm
var maml = new MAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);

// 4. Build a task batch and train
var task = new MetaLearningTask<double, Matrix<double>, Vector<double>>
{
    SupportSetX = supportX,
    SupportSetY = supportY,
    QuerySetX = queryX,
    QuerySetY = queryY,
    NumWays = 5,
    NumShots = 1,
    NumQueryPerClass = 15
};
var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task });

double loss = maml.MetaTrain(batch);

// 5. Adapt to a new task
var adapted = maml.Adapt(newTask);
var predictions = adapted.Predict(newTask.QuerySetX);
```

---

## Core Concepts

**Meta-learning** ("learning to learn") trains a model across many tasks so that it can adapt to new, unseen tasks with very few examples. The key abstractions are:

| Concept | Description |
|---------|-------------|
| **Task** | A single learning problem with a support set (training) and query set (evaluation). |
| **Episode** | A task plus metadata (domain, difficulty, loss history). |
| **N-way K-shot** | N classes, K examples per class in the support set. |
| **Meta-training** | Training across a batch of tasks to learn shared knowledge. |
| **Adaptation** | Fine-tuning on a single new task's support set. |
| **MetaLearnerBase** | Abstract base class all algorithms extend. Provides gradient computation, SPSA, cloning, etc. |

All algorithms implement `MetaTrain(TaskBatch)` and `Adapt(IMetaLearningTask)`, returning a finite loss and an adapted model respectively.

---

## Algorithm Categories

### Optimization-Based

Learn a parameter initialization that can be quickly fine-tuned to new tasks via gradient descent.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **MAML** | `MAML` | Bi-level optimization: inner loop adapts, outer loop updates initialization | Finn et al., 2017 |
| **Reptile** | `Reptile` | Move initialization toward task-adapted weights; no second-order gradients | Nichol et al., 2018 |
| **Meta-SGD** | `MetaSGD` | Learn per-parameter learning rates alongside initialization | Li et al., 2017 |
| **iMAML** | `iMAML` | Implicit differentiation avoids unrolling the inner loop | Rajeswaran et al., 2019 |
| **ANIL** | `ANIL` | Almost No Inner Loop: only adapt the head, freeze the body | Raghu et al., 2020 |
| **BOIL** | `BOIL` | Body Only update in Inner Loop: opposite of ANIL | Oh et al., 2021 |
| **LEO** | `LEO` | Latent Embedding Optimization: adapt in a low-dimensional latent space | Rusu et al., 2019 |
| **WarpGrad** | `WarpGrad` | Learn a warp layer that preconditions gradients | Flennerhag et al., 2020 |
| **ModularMeta** | `ModularMeta` | Module selection per task with routing network | Alet et al., 2018 |
| **CAVIA** | `CAVIA` | Context parameters appended to input, only adapt context | Zintgraf et al., 2019 |

### Metric-Based

Learn an embedding space where distances correspond to class similarity.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **ProtoNets** | `ProtoNets` | Classify by distance to class prototypes (mean embeddings) | Snell et al., 2017 |
| **MatchingNetworks** | `MatchingNetworks` | Attention over support set with full context embeddings | Vinyals et al., 2016 |
| **RelationNetwork** | `RelationNetwork` | Learn a distance metric as a neural network | Sung et al., 2018 |
| **TADAM** | `TADAM` | Task-dependent adaptive metric with task conditioning | Oreshkin et al., 2018 |

### Memory-Based

Use external memory to store and retrieve task-relevant information.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **MANN** | `MANN` | Memory-Augmented NN: external memory bank for few-shot | Santoro et al., 2016 |
| **NTM** | `NTM` | Neural Turing Machine with content/location-based addressing | Graves et al., 2014 |

### Hybrid / Advanced

Combine multiple meta-learning paradigms or introduce specialized architectures.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **CNAP** | `CNAP` | Conditional Neural Adaptive Processes with FiLM adaptation | Requeima et al., 2019 |
| **SEAL** | `SEAL` | Self-Augmented Learning with instance-specific augmentations | -- |
| **GNNMeta** | `GNNMeta` | Graph Neural Network propagates information between support/query | Garcia & Bruna, 2018 |
| **MetaOptNet** | `MetaOptNet` | Differentiable SVM/QP solver in the inner loop | Lee et al., 2019 |
| **DKT** | `DKT` | Deep Kernel Transfer with GP inference on learned features | Patacchiola et al., 2020 |
| **SUR** | `SUR` | Selecting Universal Representations from multiple feature extractors | Dvornik et al., 2020 |
| **TSA** | `TSA` | Task-Specific Adapters added to frozen backbone | Li et al., 2022 |
| **URL** | `URL` | Universal Representation Learning across domains | Li et al., 2021 |
| **FLUTE** | `FLUTE` | Few-shot Learning with Universal Task Embedding | Triantafillou et al., 2021 |

### Neural Processes

Function-distribution models that encode context into a latent variable and decode predictions.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **CNP** | `CNP` | Conditional NP: deterministic encoder aggregates context, decoder predicts | Garnelo et al., 2018a |
| **NP** | `NP` | Neural Process: adds latent path with KL regularization for uncertainty | Garnelo et al., 2018b |
| **ANP** | `ANP` | Attentive NP: cross-attention replaces mean aggregation for richer context | Kim et al., 2019 |
| **ConvCNP** | `ConvCNP` | Convolutional CNP: translation-equivariant encoder via convolution | Gordon et al., 2020 |
| **ConvNP** | `ConvNP` | Convolutional NP: latent variable + convolutional encoding | Foong et al., 2020 |
| **TNP** | `TNP` | Transformer NP: full self-attention over context and target tokens | Nguyen & Grover, 2023 |
| **SwinTNP** | `SwinTNP` | Shifted-window attention TNP for scalable sequence processing | -- |
| **EquivCNP** | `EquivCNP` | Equivariant CNP respecting symmetry groups in input space | Kawano et al., 2021 |
| **SteerCNP** | `SteerCNP` | Steerable CNP with group-equivariant convolution filters | Holderrieth et al., 2021 |
| **RCNP** | `RCNP` | Recurrent CNP: temporal/sequential aggregation of context | -- |
| **LBANP** | `LBANP` | Latent Bottleneck Attentive NP: information bottleneck between encoder and decoder | -- |

All Neural Process algorithms share `NeuralProcessBase` which provides encoder, decoder, and aggregation infrastructure.

### Foundation Model Era

Adapt large pre-trained models to few-shot tasks using parameter-efficient methods or generative priors.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **MetaLoRA** | `MetaLoRA` | Meta-learn low-rank adapter initialization for rapid fine-tuning | -- |
| **LoRARecycle** | `LoRARecycle` | Recycle and compose LoRA modules from a shared bank across tasks | -- |
| **ICMFusion** | `ICMFusion` | In-Context Model fusion: merge adapted parameters weighted by task similarity | -- |
| **MetaLoRABank** | `MetaLoRABank` | Bank of LoRA experts with attention-based routing per task | -- |
| **AutoLoRA** | `AutoLoRA` | Automatically search for optimal LoRA rank allocation per layer | -- |
| **MetaDiff** | `MetaDiff` | Meta-learning with diffusion model priors for parameter generation | -- |
| **MetaDM** | `MetaDM` | Denoising-based meta-learning: generate adapted parameters via denoising | -- |
| **MetaDDPM** | `MetaDDPM` | DDPM-based multi-step denoising of parameter perturbations | -- |

### Bayesian Extensions

Provide principled uncertainty quantification and PAC-Bayes generalization guarantees.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **PACOH** | `PACOH` | PAC-Optimal Hyper-posteriors: meta-learn priors with PAC-Bayes bounds | Rothfuss et al., 2021 |
| **MetaPACOH** | `MetaPACOH` | Extended PACOH with hierarchical prior structure | -- |
| **BMAML** | `BMAML` | Bayesian MAML: maintain particle ensemble over adapted parameters | Yoon et al., 2018 |
| **BayProNet** | `BayProNet` | Bayesian Prototypical Networks with Gaussian prototype distributions | -- |
| **FlexPACBayes** | `FlexPACBayes` | Flexible PAC-Bayes with learnable divergence and complexity terms | -- |

### Cross-Domain Few-Shot

Transfer meta-knowledge across distinct visual/semantic domains.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **MetaFDMixup** | `MetaFDMixup` | Feature-level domain mixup for cross-domain robustness | -- |
| **FreqPrior** | `FreqPrior` | Frequency-domain priors: separate domain-specific (high-freq) and shared (low-freq) features | -- |
| **MetaCollaborative** | `MetaCollaborative` | Multi-source collaborative training with domain-specific adapters | -- |
| **SDCL** | `SDCL` | Self-Distillation Continual Learning: teacher-student self-distillation | -- |
| **FreqPrompt** | `FreqPrompt` | Frequency-domain prompt tuning: learnable spectral prompts injected into features | -- |
| **OpenMAMLPlus** | `OpenMAMLPlus` | Open-set MAML with outlier detection and rejection capability | -- |

### Meta-Reinforcement Learning

Adapt reinforcement learning agents to new environments/rewards with minimal interaction.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **PEARL** | `PEARL` | Probabilistic Embeddings for Actor-critic RL: latent context variable inferred from experience | Rakelly et al., 2019 |
| **DREAM** | `DREAM` | Distributional reward estimation and meta-learning for reward shaping | -- |
| **DiscoRL** | `DiscoRL` | Discovery-driven RL: intrinsic motivation through prediction error | -- |
| **InContextRL** | `InContextRL` | In-context RL: feed episode history as context (no explicit adaptation) | -- |
| **HyperNetMetaRL** | `HyperNetMetaRL` | Hypernetwork generates policy weights from task description | -- |
| **ContextMetaRL** | `ContextMetaRL` | Context-conditioned policy with environment embedding | -- |

### Continual / Online Meta-Learning

Learn continuously across a stream of tasks without catastrophic forgetting.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **ACL** | `ACL` | Adaptive Continual Learning: per-task attention masks prevent interference | -- |
| **iTAML** | `iTAML` | Incremental Task-Agnostic Meta-Learning with knowledge distillation | Rajasegaran et al., 2020 |
| **MetaContinualAL** | `MetaContinualAL` | Meta-learned active learning policy for continual settings | -- |
| **MePo** | `MePo` | Memory Populations: maintain diverse parameter populations across tasks | -- |
| **OML** | `OML` | Online Meta-Learning: single-pass streaming with representation protection | Javed & White, 2019 |
| **MOCA** | `MOCA` | Meta-learning Online adaptation with Context Augmentation | -- |

### Task Augmentation / Sampling

Improve meta-training by generating, augmenting, or strategically sampling tasks.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **MetaTask** | `MetaTask` | Task-level augmentation: generate synthetic tasks from real ones | -- |
| **ATAML** | `ATAML` | Adaptive Task Augmentation ML: difficulty-aware task generation | -- |
| **MPTS** | `MPTS` | Meta-learning with Progressive Task Scheduling: curriculum over tasks | -- |
| **DynamicTaskSampling** | `DynamicTaskSampling` | Loss-weighted task sampling: oversample high-loss tasks | -- |
| **UnsupervisedMetaLearn** | `UnsupervisedMetaLearn` | Create pseudo-tasks from unlabeled data via clustering | -- |

### Transductive Methods

Exploit the structure of the entire query set (not just individual queries) during inference.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **GCDPLNet** | `GCDPLNet` | Graph-based Cross-Domain Prototype Learning: attention-based message passing on support-query graph | -- |
| **BayTransProto** | `BayTransProto` | Bayesian Transductive Prototypical Networks: posterior sampling over prototypes with query refinement | -- |
| **JMP** | `JMP` | Joint Multi-Phase: two-phase inner loop (feature adaptation then classifier refinement) | -- |
| **ETPN** | `ETPN` | Embedding-Transformed Prototypical Networks: learned sigmoid embedding transform | -- |
| **ActiveTransFSL** | `ActiveTransFSL` | Active Transductive FSL: gradient-based uncertainty to select informative query points | -- |

### Hypernetwork Methods

Use a secondary network to generate or modulate the primary model's parameters.

| Algorithm | Type Enum | Key Idea | Reference |
|-----------|-----------|----------|-----------|
| **TaskCondHyperNet** | `TaskCondHyperNet` | Chunked hypernetwork generates parameter deltas conditioned on task embedding | -- |
| **HyperCLIP** | `HyperCLIP` | CLIP-style contrastive alignment between task and parameter embeddings (InfoNCE) | -- |
| **RecurrentHyperNet** | `RecurrentHyperNet` | GRU cell processes gradients recurrently to generate per-parameter learning rate modulation | -- |
| **HyperNeRFMeta** | `HyperNeRFMeta` | NeRF-style positional encoding + task latent code for per-parameter-group modulation | -- |

---

## Data Infrastructure

The episodic data infrastructure provides industry-standard abstractions for meta-learning data pipelines.

### IEpisode

`IEpisode<T, TInput, TOutput>` wraps an `IMetaLearningTask` with rich metadata:

| Property | Type | Description |
|----------|------|-------------|
| `Task` | `IMetaLearningTask<T, TInput, TOutput>` | The underlying support/query data |
| `EpisodeId` | `int` | Auto-incrementing unique identifier |
| `Domain` | `string?` | Domain label for cross-domain settings |
| `Difficulty` | `double?` | Difficulty score in [0, 1] for curriculum learning |
| `LastLoss` | `double?` | Most recent evaluation loss for dynamic sampling |
| `SampleCount` | `int` | How many times this episode has been sampled |
| `CreatedTimestamp` | `long` | UTC creation time (milliseconds) |
| `EpisodeMetadata` | `Dictionary<string, object>?` | Arbitrary key-value metadata |

```csharp
var episode = new Episode<double, Matrix<double>, Vector<double>>(
    task: myTask,
    domain: "medical",
    difficulty: 0.7
);
```

### IMetaDataset

`IMetaDataset<T, TInput, TOutput>` is a high-level dataset that generates episodes on-the-fly:

```csharp
// Check feasibility
bool ok = dataset.SupportsConfiguration(numWays: 5, numShots: 1, numQueryPerClass: 15);

// Sample episodes
var episode = dataset.SampleEpisode(numWays: 5, numShots: 1, numQueryPerClass: 15);
var batch = dataset.SampleEpisodes(count: 16, numWays: 5, numShots: 1, numQueryPerClass: 15);

// Reproducibility
dataset.SetSeed(42);
```

### ITaskSampler

`ITaskSampler<T, TInput, TOutput>` controls the strategy for sampling tasks during meta-training:

```csharp
// Sample a batch of tasks
TaskBatch<T, TInput, TOutput> batch = sampler.SampleBatch(batchSize: 8);

// Single episode
IEpisode<T, TInput, TOutput> episode = sampler.SampleOne();

// Feedback loop for dynamic samplers
sampler.UpdateWithFeedback(episodes, losses);
```

### Task Samplers

| Sampler | Description |
|---------|-------------|
| `UniformTaskSampler` | Pure random sampling from the meta-dataset. Simple, effective baseline. |
| `BalancedTaskSampler` | Ensures every class appears equally often across sampled tasks. Prevents class imbalance. |
| `DynamicTaskSampler` | Loss-weighted sampling: tasks with higher loss are sampled more frequently. Self-adjusting curriculum. |
| `BatchEpisodeSampler` | Samples pre-built batches of episodes for efficient data loading. |

```csharp
// Uniform sampling
var sampler = new UniformTaskSampler<double, Matrix<double>, Vector<double>>(
    dataset, numWays: 5, numShots: 1, numQueryPerClass: 15);

// Dynamic (loss-weighted) sampling
var dynamic = new DynamicTaskSampler<double, Matrix<double>, Vector<double>>(
    dataset, numWays: 5, numShots: 1, numQueryPerClass: 15,
    temperature: 2.0);

// After each meta-training step, feed back losses
dynamic.UpdateWithFeedback(episodes, losses);
```

### Synthetic Datasets

Two built-in synthetic meta-datasets for prototyping and benchmarking:

**SineWaveMetaDataset** (regression): Each task is `y = A * sin(x + phase)` with random amplitude and phase. The standard MAML benchmark.

```csharp
var sineData = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(
    amplitudeRange: (0.1, 5.0),
    phaseRange: (0, Math.PI),
    xRange: (-5.0, 5.0),
    numClasses: 100,
    examplesPerClass: 20);
```

**GaussianClassificationMetaDataset** (classification): Each class is a Gaussian blob in feature space with random mean.

```csharp
var gaussianData = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
    featureDim: 10,
    numClasses: 50,
    examplesPerClass: 30,
    classSeparation: 3.0,
    clusterStdDev: 1.0);
```

### Episode Caching

`EpisodeCache<T, TInput, TOutput>` caches sampled episodes for replay and efficient reuse:

```csharp
var cache = new EpisodeCache<double, Matrix<double>, Vector<double>>(maxSize: 1000);
cache.Add(episode);
var cached = cache.Sample(count: 8);
```

### Backward Compatibility

`EpisodicDataLoaderTaskSamplerAdapter` bridges the existing `EpisodicDataLoaderBase` with the new `ITaskSampler` interface, so all pre-existing data loaders work seamlessly with the new infrastructure.

---

## Choosing an Algorithm

Use this decision tree to select the right algorithm family:

```
What is your scenario?
|
+-- Few-shot classification (standard)
|   +-- Need uncertainty? --> Bayesian (PACOH, BMAML, BayProNet)
|   +-- Simple baseline --> ProtoNets
|   +-- Best general-purpose --> MAML or ANIL
|   +-- Large pre-trained backbone --> Foundation Model Era (MetaLoRA, AutoLoRA)
|
+-- Few-shot regression / function fitting
|   +-- Standard benchmark --> Neural Processes (CNP, NP, ANP)
|   +-- Need translation equivariance --> ConvCNP, EquivCNP
|   +-- Scalable attention --> TNP, SwinTNP
|
+-- Cross-domain transfer
|   +-- Feature-level augmentation --> MetaFDMixup
|   +-- Frequency separation --> FreqPrior, FreqPrompt
|   +-- Multi-source --> MetaCollaborative
|   +-- Open-set (novel classes) --> OpenMAMLPlus
|
+-- Reinforcement learning
|   +-- Context-based --> PEARL, ContextMetaRL
|   +-- Reward shaping --> DREAM
|   +-- Policy generation --> HyperNetMetaRL
|   +-- In-context (no explicit adaptation) --> InContextRL
|
+-- Continual learning (streaming tasks)
|   +-- Prevent forgetting --> ACL, iTAML
|   +-- Active learning --> MetaContinualAL
|   +-- Online single-pass --> OML
|
+-- Want to improve meta-training
|   +-- Task augmentation --> MetaTask, ATAML
|   +-- Curriculum scheduling --> MPTS
|   +-- Dynamic sampling --> DynamicTaskSampling
|   +-- No labels available --> UnsupervisedMetaLearn
|
+-- Exploit query set structure
|   +-- Graph-based --> GCDPLNet
|   +-- Bayesian transductive --> BayTransProto
|   +-- Multi-phase adaptation --> JMP
|
+-- Parameter generation
    +-- Task-conditioned weights --> TaskCondHyperNet
    +-- Contrastive alignment --> HyperCLIP
    +-- Recurrent gradient processing --> RecurrentHyperNet
    +-- NeRF-style conditioning --> HyperNeRFMeta
```

### Quick Comparison by Compute Cost

| Cost Level | Algorithms |
|------------|-----------|
| **Low** | ProtoNets, Reptile, ANIL, CNP, ETPN |
| **Medium** | MAML, MetaSGD, NP, ANP, PACOH, FreqPrior |
| **High** | iMAML, LEO, BMAML, TNP, MetaDiff, GCDPLNet |

---

## Advanced Usage

### Custom Options

Every algorithm has a dedicated options class with algorithm-specific parameters plus standard `IMetaLearnerOptions<T>` properties:

```csharp
// Standard properties available on all options
var options = new ANPOptions<double, Matrix<double>, Vector<double>>(model)
{
    // Standard meta-learning parameters
    InnerLearningRate = 0.01,
    OuterLearningRate = 0.001,
    InnerSteps = 5,

    // ANP-specific parameters
    NumHeads = 4,
    LatentDim = 128,
    CrossAttentionDim = 64,
    UseSelfAttention = true
};
```

### Multi-Step Training Loop

```csharp
var algorithm = new PACOHAlgorithm<double, Matrix<double>, Vector<double>>(options);
var sampler = new DynamicTaskSampler<double, Matrix<double>, Vector<double>>(
    dataset, numWays: 5, numShots: 5, numQueryPerClass: 15);

for (int epoch = 0; epoch < 1000; epoch++)
{
    var batch = sampler.SampleBatch(batchSize: 4);
    double loss = algorithm.MetaTrain(batch);

    // Feed back losses for dynamic sampling
    // sampler.UpdateWithFeedback(episodes, losses);

    if (epoch % 100 == 0)
        Console.WriteLine($"Epoch {epoch}: loss = {loss:F4}");
}

// Evaluate on held-out task
var testTask = testDataset.SampleEpisode(5, 1, 15).Task;
var adapted = algorithm.Adapt(testTask);
var predictions = adapted.Predict(testTask.QuerySetX);
```

### Algorithm Type Enum

Every algorithm exposes its type via `algorithm.AlgorithmType`, returning a value from `MetaLearningAlgorithmType`. This enables runtime algorithm selection and serialization:

```csharp
MetaLearningAlgorithmType type = algorithm.AlgorithmType;
// e.g., MetaLearningAlgorithmType.PACOH
```

### SPSA for Non-Differentiable Parameters

Many algorithms use Simultaneous Perturbation Stochastic Approximation (SPSA) for auxiliary parameters that lack analytical gradients. This is handled internally by `MetaLearnerBase.UpdateAuxiliaryParamsSPSA()`.

---

## File Layout

```
src/
  MetaLearning/
    MetaLearnerBase.cs              # Abstract base class
    MetaLearningAlgorithmType.cs    # Enum of all algorithm types
    Algorithms/
      MAMLAlgorithm.cs              # Optimization-based
      ProtoNetsAlgorithm.cs         # Metric-based
      CNPAlgorithm.cs               # Neural Processes
      NeuralProcessBase.cs          # Shared NP infrastructure
      MetaLoRAAlgorithm.cs          # Foundation Model Era
      PACOHAlgorithm.cs             # Bayesian Extensions
      MetaFDMixupAlgorithm.cs       # Cross-Domain
      PEARLAlgorithm.cs             # Meta-RL
      ACLAlgorithm.cs               # Continual/Online
      MetaTaskAlgorithm.cs          # Task Augmentation
      GCDPLNetAlgorithm.cs          # Transductive
      TaskCondHyperNetAlgorithm.cs  # Hypernetwork
      AdaptedMetaModel.cs           # Wrapper for adapted models
      ...
    Options/
      MAMLOptions.cs
      ProtoNetsOptions.cs
      CNPOptions.cs
      ...
    Data/
      Episode.cs                    # IEpisode implementation
      MetaDatasetBase.cs            # Abstract IMetaDataset base
      SineWaveMetaDataset.cs        # Regression benchmark
      GaussianClassificationMetaDataset.cs  # Classification benchmark
      UniformTaskSampler.cs         # Random sampling
      BalancedTaskSampler.cs        # Class-balanced sampling
      DynamicTaskSampler.cs         # Loss-weighted sampling
      BatchEpisodeSampler.cs        # Batch episode access
      EpisodeCache.cs               # Episode replay buffer
      EpisodicDataLoaderTaskSamplerAdapter.cs  # Legacy bridge
      TaskBatch.cs                  # Batch wrapper
  Interfaces/
    IMetaLearningTask.cs
    IEpisode.cs
    IMetaDataset.cs
    ITaskSampler.cs

tests/
  IntegrationTests/MetaLearning/
    NeuralProcessAlgorithmTests.cs
    FoundationModelMetaTests.cs
    BayesianMetaTests.cs
    CrossDomainMetaTests.cs
    MetaRLTests.cs
    ContinualMetaTests.cs
    AdvancedMetaTests.cs
    DataInfrastructureTests.cs
```
