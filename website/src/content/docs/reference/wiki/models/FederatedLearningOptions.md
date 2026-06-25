---
title: "FederatedLearningOptions"
description: "Configuration options for federated learning training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated learning training.

## How It Works

This class contains all the configurable parameters needed to set up and run a federated learning system.

**For Beginners:** Options are like the settings panel for federated learning.
Just as you configure settings for a video game (difficulty, graphics quality, etc.),
these options let you configure how federated learning should work.

Key configuration areas:

- Client Management: How many clients, how to select them
- Training: Learning rates, epochs, batch sizes
- Privacy: Differential privacy parameters
- Communication: How often to aggregate, compression settings
- Convergence: When to stop training

For example, a typical configuration might be:

- 100 total clients (e.g., hospitals)
- Select 10 clients per round (10% participation)
- Each client trains for 5 local epochs
- Use privacy budget ε=1.0, δ=1e-5
- Run for maximum 100 rounds or until convergence

## Properties

| Property | Summary |
|:-----|:--------|
| `Adapters` | Gets or sets federated adapter options for parameter-efficient fine-tuning (LoRA, prompt tuning). |
| `AdvancedCompression` | Gets or sets advanced communication compression options (PowerSGD, sketching, error feedback). |
| `AggregationStrategy` | Gets or sets the aggregation strategy to use. |
| `AsyncFederatedLearning` | Gets or sets asynchronous federated learning options (FedAsync / FedBuff). |
| `BackdoorDefense` | Gets or sets backdoor attack detection and defense options. |
| `BatchSize` | Gets or sets the batch size for local training. |
| `ClientSelection` | Gets or sets client selection options (strategy and related parameters). |
| `ClientSelectionFraction` | Gets or sets the fraction of clients to select for each training round (0.0 to 1.0). |
| `Compression` | Gets or sets federated compression options. |
| `CompressionRatio` | Gets or sets the compression ratio (0.0 to 1.0) if compression is enabled. |
| `ContinualLearning` | Gets or sets federated continual learning options for preventing catastrophic forgetting. |
| `ContributionEvaluation` | Gets or sets client contribution evaluation options for measuring each client's value. |
| `ConvergenceThreshold` | Gets or sets the convergence threshold for early stopping. |
| `Decentralized` | Gets or sets decentralized (serverless) federated learning options. |
| `DifferentialPrivacyClipNorm` | Gets or sets the clipping norm used by differential privacy mechanisms. |
| `DifferentialPrivacyMode` | Gets or sets where differential privacy is applied (local, central, or both). |
| `Distillation` | Gets or sets federated knowledge distillation options (FedMD, FedDF, FedGEN). |
| `DriftDetection` | Gets or sets federated concept drift detection and adaptation options. |
| `EnablePersonalization` | Gets or sets whether to enable personalization. |
| `Fairness` | Gets or sets fairness constraint options for equitable model performance across client groups. |
| `GraphLearning` | Gets or sets federated graph learning options. |
| `HeterogeneityCorrection` | Gets or sets federated heterogeneity correction options (SCAFFOLD / FedNova / FedDyn). |
| `HomomorphicEncryption` | Gets or sets homomorphic encryption options for federated aggregation (CKKS/BFV). |
| `LearningRate` | Gets or sets the learning rate for local client training. |
| `LocalEpochs` | Gets or sets the number of local training epochs each client performs per round. |
| `MaxRounds` | Gets or sets the maximum number of federated learning rounds to execute. |
| `MetaLearning` | Gets or sets federated meta-learning options (Per-FedAvg / FedMAML / Reptile-style). |
| `MinRoundsBeforeConvergence` | Gets or sets the minimum number of rounds to train before checking convergence. |
| `Mode` | Gets or sets the federated learning mode (horizontal or vertical). |
| `MultiPartyComputation` | Gets or sets Multi-Party Computation options for secure operations beyond summation. |
| `NumberOfClients` | Gets or sets the total number of clients participating in federated learning. |
| `Personalization` | Gets or sets personalization options (preferred configuration for FedPer/FedRep/Ditto/pFedMe/clustered). |
| `PersonalizationLayerFraction` | Gets or sets the fraction of model layers to keep personalized (not aggregated). |
| `PrivacyAccountant` | Gets or sets which privacy accountant to use for reporting privacy spend. |
| `PrivacyDelta` | Gets or sets the delta (δ) parameter for differential privacy (failure probability). |
| `PrivacyEpsilon` | Gets or sets the epsilon (ε) parameter for differential privacy (privacy budget). |
| `PrivateSetIntersection` | Gets or sets Private Set Intersection options for entity alignment. |
| `ProximalMu` | Gets or sets the proximal term coefficient for FedProx algorithm. |
| `RandomSeed` | Gets or sets a random seed for reproducibility. |
| `RobustAggregation` | Gets or sets options for robust aggregation strategies (Median, TrimmedMean, Krum, MultiKrum, Bulyan). |
| `SecureAggregation` | Gets or sets secure aggregation configuration options. |
| `ServerOptimizer` | Gets or sets server-side federated optimization options (FedOpt family). |
| `TrustedExecutionEnvironment` | Gets or sets Trusted Execution Environment options for hardware-backed secure aggregation. |
| `Unlearning` | Gets or sets federated unlearning options (right to be forgotten). |
| `UseCompression` | Gets or sets whether to use gradient compression to reduce communication costs. |
| `UseDifferentialPrivacy` | Gets or sets whether to use differential privacy. |
| `UseSecureAggregation` | Gets or sets whether to use secure aggregation. |
| `Verification` | Gets or sets verification options for zero-knowledge proof-based update validation. |
| `VerticalLearning` | Gets or sets vertical federated learning options (split learning, entity alignment, etc.). |

