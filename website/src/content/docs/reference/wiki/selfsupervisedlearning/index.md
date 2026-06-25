---
title: "Self Supervised Learning"
description: "All 62 public types in the AiDotNet.selfsupervisedlearning namespace, organized by kind."
section: "API Reference"
---

**62** public types in this namespace, organized by kind.

## Models & Types (40)

| Type | Summary |
|:-----|:--------|
| [`BYOLLoss<T>`](/docs/reference/wiki/selfsupervisedlearning/byolloss/) | BYOL (Bootstrap Your Own Latent) Loss - a simple cosine similarity loss. |
| [`BYOL<T>`](/docs/reference/wiki/selfsupervisedlearning/byol/) | BYOL: Bootstrap Your Own Latent - Self-supervised learning without negative samples. |
| [`BarlowTwinsLoss<T>`](/docs/reference/wiki/selfsupervisedlearning/barlowtwinsloss/) | Barlow Twins Loss - redundancy reduction loss for self-supervised learning. |
| [`BarlowTwins<T>`](/docs/reference/wiki/selfsupervisedlearning/barlowtwins/) | Barlow Twins: Self-Supervised Learning via Redundancy Reduction. |
| [`BenchmarkResult<T>`](/docs/reference/wiki/selfsupervisedlearning/benchmarkresult/) | Result from a single benchmark evaluation. |
| [`BenchmarkSuite<T>`](/docs/reference/wiki/selfsupervisedlearning/benchmarksuite/) | Complete benchmark suite results. |
| [`CenteringMechanism<T>`](/docs/reference/wiki/selfsupervisedlearning/centeringmechanism/) | Centering mechanism for preventing collapse in self-distillation methods. |
| [`DINOLoss<T>`](/docs/reference/wiki/selfsupervisedlearning/dinoloss/) | DINO (Self-Distillation with No Labels) Loss for self-supervised learning. |
| [`DINO<T>`](/docs/reference/wiki/selfsupervisedlearning/dino/) | DINO: Self-Distillation with No Labels - a self-supervised method for Vision Transformers. |
| [`FineTuningResult<T>`](/docs/reference/wiki/selfsupervisedlearning/finetuningresult/) | Result from fine-tuning an SSL pretrained encoder. |
| [`InfoNCELoss<T>`](/docs/reference/wiki/selfsupervisedlearning/infonceloss/) | InfoNCE (Noise Contrastive Estimation) Loss for contrastive learning. |
| [`KNNEvaluator<T>`](/docs/reference/wiki/selfsupervisedlearning/knnevaluator/) | k-Nearest Neighbors (k-NN) evaluation for SSL representation quality. |
| [`LinearEvalResult<T>`](/docs/reference/wiki/selfsupervisedlearning/linearevalresult/) | Results from linear evaluation. |
| [`LinearEvaluator<T>`](/docs/reference/wiki/selfsupervisedlearning/linearevaluator/) | Linear evaluation protocol for assessing SSL representation quality. |
| [`LinearProjector<T>`](/docs/reference/wiki/selfsupervisedlearning/linearprojector/) | Linear projection head for self-supervised learning. |
| [`MAEReconstructionLoss<T>`](/docs/reference/wiki/selfsupervisedlearning/maereconstructionloss/) | MAE (Masked Autoencoder) Reconstruction Loss for self-supervised learning. |
| [`MAE<T>`](/docs/reference/wiki/selfsupervisedlearning/mae/) | MAE: Masked Autoencoder for Self-Supervised Vision Learning. |
| [`MLPProjector<T>`](/docs/reference/wiki/selfsupervisedlearning/mlpprojector/) | Multi-layer perceptron (MLP) projection head for self-supervised learning. |
| [`MemoryBank<T>`](/docs/reference/wiki/selfsupervisedlearning/memorybank/) | FIFO memory queue for storing embeddings in contrastive learning. |
| [`MoCoV2<T>`](/docs/reference/wiki/selfsupervisedlearning/mocov2/) | MoCo v2: Improved Baselines with Momentum Contrastive Learning. |
| [`MoCoV3<T>`](/docs/reference/wiki/selfsupervisedlearning/mocov3/) | MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers. |
| [`MoCo<T>`](/docs/reference/wiki/selfsupervisedlearning/moco/) | MoCo: Momentum Contrast for Unsupervised Visual Representation Learning. |
| [`MomentumEncoder<T>`](/docs/reference/wiki/selfsupervisedlearning/momentumencoder/) | Momentum-updated encoder for self-supervised learning methods. |
| [`NTXentLoss<T>`](/docs/reference/wiki/selfsupervisedlearning/ntxentloss/) | Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for contrastive learning. |
| [`SSLAugmentationContext<T>`](/docs/reference/wiki/selfsupervisedlearning/sslaugmentationcontext/) | Context for SSL augmentation operations. |
| [`SSLAugmentationPolicies<T>`](/docs/reference/wiki/selfsupervisedlearning/sslaugmentationpolicies/) | Provides standard augmentation policies for self-supervised learning methods. |
| [`SSLFineTuningPipeline<T>`](/docs/reference/wiki/selfsupervisedlearning/sslfinetuningpipeline/) | Pipeline for fine-tuning SSL pretrained encoders on downstream tasks. |
| [`SSLMetricReport<T>`](/docs/reference/wiki/selfsupervisedlearning/sslmetricreport/) | Complete SSL metrics report. |
| [`SSLMetrics<T>`](/docs/reference/wiki/selfsupervisedlearning/sslmetrics/) | Metrics for monitoring and evaluating self-supervised learning. |
| [`SSLPretrainingPipeline<T>`](/docs/reference/wiki/selfsupervisedlearning/sslpretrainingpipeline/) | High-level pipeline for SSL pretraining. |
| [`SSLResult<T>`](/docs/reference/wiki/selfsupervisedlearning/sslresult/) | Result from SSL pretraining containing the trained encoder and metrics. |
| [`SSLSession<T>`](/docs/reference/wiki/selfsupervisedlearning/sslsession/) | Manages a self-supervised learning training session. |
| [`SSLStepResult<T>`](/docs/reference/wiki/selfsupervisedlearning/sslstepresult/) | Result of a single SSL training step. |
| [`SSLTrainingHistory<T>`](/docs/reference/wiki/selfsupervisedlearning/ssltraininghistory/) | Training history from SSL pretraining. |
| [`SimCLR<T>`](/docs/reference/wiki/selfsupervisedlearning/simclr/) | SimCLR: A Simple Framework for Contrastive Learning of Visual Representations. |
| [`SimSiam<T>`](/docs/reference/wiki/selfsupervisedlearning/simsiam/) | SimSiam: Exploring Simple Siamese Representation Learning. |
| [`SymmetricProjector<T>`](/docs/reference/wiki/selfsupervisedlearning/symmetricprojector/) | Symmetric Projector Head for BYOL and SimSiam-style methods. |
| [`TemperatureScheduler`](/docs/reference/wiki/selfsupervisedlearning/temperaturescheduler/) | Schedules temperature parameters during self-supervised learning training. |
| [`TransferBenchmark<T>`](/docs/reference/wiki/selfsupervisedlearning/transferbenchmark/) | Transfer learning benchmark for evaluating SSL representations on downstream tasks. |
| [`iBOT<T>`](/docs/reference/wiki/selfsupervisedlearning/ibot/) | iBOT: Image BERT Pre-Training with Online Tokenizer - combining DINO with masked image modeling. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`SSLMethodBase<T>`](/docs/reference/wiki/selfsupervisedlearning/sslmethodbase/) | Abstract base class for self-supervised learning methods. |
| [`TeacherStudentSSL<T>`](/docs/reference/wiki/selfsupervisedlearning/teacherstudentssl/) | Base class for teacher-student self-supervised learning methods. |

## Interfaces (5)

| Type | Summary |
|:-----|:--------|
| [`IDetachedTensor<T>`](/docs/reference/wiki/selfsupervisedlearning/idetachedtensor/) | Marker interface for tensors that should not receive gradients. |
| [`IMemoryBank<T>`](/docs/reference/wiki/selfsupervisedlearning/imemorybank/) | Defines the contract for memory banks used in contrastive learning methods. |
| [`IMomentumEncoder<T>`](/docs/reference/wiki/selfsupervisedlearning/imomentumencoder/) | Defines the contract for momentum-updated encoders used in SSL methods. |
| [`IProjectorHead<T>`](/docs/reference/wiki/selfsupervisedlearning/iprojectorhead/) | Defines the contract for projection heads used in self-supervised learning. |
| [`ISSLMethod<T>`](/docs/reference/wiki/selfsupervisedlearning/isslmethod/) | Defines the contract for self-supervised learning methods. |

## Enums (5)

| Type | Summary |
|:-----|:--------|
| [`BenchmarkProtocol`](/docs/reference/wiki/selfsupervisedlearning/benchmarkprotocol/) | Supported evaluation protocols for transfer learning benchmarks. |
| [`FineTuningStrategy`](/docs/reference/wiki/selfsupervisedlearning/finetuningstrategy/) | Fine-tuning strategies for SSL pretrained encoders. |
| [`SSLCommunicationBackend`](/docs/reference/wiki/selfsupervisedlearning/sslcommunicationbackend/) | Communication backends for distributed SSL training. |
| [`SSLOptimizerType`](/docs/reference/wiki/selfsupervisedlearning/ssloptimizertype/) | Optimizer types optimized for SSL training. |
| [`TemperatureScheduleType`](/docs/reference/wiki/selfsupervisedlearning/temperaturescheduletype/) | Types of temperature scheduling strategies. |

## Structs (1)

| Type | Summary |
|:-----|:--------|
| [`DetachedTensor<T>`](/docs/reference/wiki/selfsupervisedlearning/detachedtensor/) | A wrapper that marks a tensor as detached from the computation graph. |

## Options & Configuration (8)

| Type | Summary |
|:-----|:--------|
| [`BYOLConfig`](/docs/reference/wiki/selfsupervisedlearning/byolconfig/) | BYOL-specific configuration settings. |
| [`BarlowTwinsConfig`](/docs/reference/wiki/selfsupervisedlearning/barlowtwinsconfig/) | Barlow Twins-specific configuration settings. |
| [`DINOConfig`](/docs/reference/wiki/selfsupervisedlearning/dinoconfig/) | DINO-specific configuration settings. |
| [`FineTuningConfig`](/docs/reference/wiki/selfsupervisedlearning/finetuningconfig/) | Configuration for fine-tuning SSL pretrained encoders. |
| [`MAEConfig`](/docs/reference/wiki/selfsupervisedlearning/maeconfig/) | MAE-specific configuration settings. |
| [`MoCoConfig`](/docs/reference/wiki/selfsupervisedlearning/mococonfig/) | MoCo-specific configuration settings. |
| [`SSLConfig`](/docs/reference/wiki/selfsupervisedlearning/sslconfig/) | Unified configuration for self-supervised learning with industry-standard defaults. |
| [`SSLDistributedConfig`](/docs/reference/wiki/selfsupervisedlearning/ssldistributedconfig/) | Configuration for distributed SSL training using DDP (Distributed Data Parallel). |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`StopGradient<T>`](/docs/reference/wiki/selfsupervisedlearning/stopgradient/) | Provides stop-gradient operations for self-supervised learning. |

