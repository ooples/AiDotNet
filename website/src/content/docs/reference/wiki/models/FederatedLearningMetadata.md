---
title: "FederatedLearningMetadata"
description: "Contains metadata and metrics about federated learning training progress and results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains metadata and metrics about federated learning training progress and results.

## How It Works

This class tracks various metrics throughout the federated learning process to help
monitor training progress, diagnose issues, and evaluate model quality.

**For Beginners:** Metadata is like a training diary that records what happened
during federated learning - how long it took, how accurate the model became, which
clients participated, etc.

Think of this as a comprehensive training report containing:

- Performance metrics: Accuracy, loss, convergence
- Resource usage: Time, communication costs
- Participation: Which clients contributed
- Privacy tracking: Privacy budget consumption

For example, after training you might see:

- Total rounds: 50 (out of max 100)
- Final accuracy: 92.5%
- Training time: 2 hours
- Total clients participated: 100
- Privacy budget used: ε=5.0 (out of 10.0 total)

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyHistory` | Gets or sets the history of accuracy values across all rounds. |
| `AggregationStrategyUsed` | Gets or sets the aggregation strategy used during training. |
| `AsyncModeUsed` | Gets or sets the asynchronous federated learning mode used (if any). |
| `AverageClientsPerRound` | Gets or sets the average number of clients selected per round. |
| `AverageRoundTimeSeconds` | Gets or sets the average time per round in seconds. |
| `CompressionEnabled` | Gets or sets whether compression was enabled for client updates. |
| `CompressionStrategyUsed` | Gets or sets the compression strategy used for client updates (if enabled). |
| `Converged` | Gets or sets whether training converged before reaching maximum rounds. |
| `ConvergenceRound` | Gets or sets the round at which convergence was detected. |
| `CumulativeContributionScores` | Gets or sets the cumulative contribution scores across all rounds. |
| `DifferentialPrivacyEnabled` | Gets or sets whether differential privacy was enabled. |
| `DriftDetectedRounds` | Gets or sets the number of rounds in which global drift was detected. |
| `DriftDetectionMethodUsed` | Gets or sets the drift detection method used. |
| `FairnessConstraintUsed` | Gets or sets the fairness constraint type used. |
| `FairnessConstraintsEnabled` | Gets or sets whether fairness constraints were applied during training. |
| `FinalAccuracy` | Gets or sets the final global model accuracy on validation data. |
| `FinalLoss` | Gets or sets the final global model loss value. |
| `HeterogeneityCorrectionUsed` | Gets or sets the heterogeneity correction method used (if any). |
| `HomomorphicEncryptionEnabled` | Gets or sets whether homomorphic encryption was enabled. |
| `HomomorphicEncryptionModeUsed` | Gets or sets the HE mode used (HEOnly or Hybrid). |
| `HomomorphicEncryptionProviderUsed` | Gets or sets the HE provider used. |
| `HomomorphicEncryptionSchemeUsed` | Gets or sets the HE scheme used (CKKS or BFV). |
| `IdentifiedFreeRiders` | Gets or sets the IDs of clients identified as free-riders. |
| `LossHistory` | Gets or sets the history of loss values across all rounds. |
| `MetaLearningEnabled` | Gets or sets whether federated meta-learning was enabled. |
| `MetaLearningInnerEpochsUsed` | Gets or sets the inner (client) adaptation epochs used for meta-learning. |
| `MetaLearningRateUsed` | Gets or sets the meta learning rate used by the server update rule. |
| `MetaLearningStrategyUsed` | Gets or sets the federated meta-learning strategy used (Reptile/PerFedAvg/FedMAML). |
| `Notes` | Gets or sets additional notes or observations about the training run. |
| `PersonalizationEnabled` | Gets or sets whether personalization was enabled. |
| `PersonalizationLocalAdaptationEpochs` | Gets or sets the number of post-aggregation local adaptation epochs (if configured). |
| `PersonalizationStrategyUsed` | Gets or sets the personalization strategy used (FedPer, FedRep, Ditto, pFedMe, Clustered). |
| `PersonalizedParameterFraction` | Gets or sets the fraction of parameters treated as personalized for head-split strategies. |
| `PrivacyAccountantUsed` | Gets or sets which privacy accountant was used for reporting. |
| `ReportedDelta` | Gets or sets the delta used when reporting `ReportedEpsilonAtDelta`. |
| `ReportedEpsilonAtDelta` | Gets or sets the reported epsilon at the reported delta (when supported by the accountant). |
| `RoundMetrics` | Gets or sets the per-round detailed metrics. |
| `RoundsCompleted` | Gets or sets the number of federated learning rounds completed. |
| `SecureAggregationEnabled` | Gets or sets whether secure aggregation was enabled. |
| `SecureAggregationMinimumUploaderCountUsed` | Gets or sets the minimum uploader count used by secure aggregation (dropout-resilient modes). |
| `SecureAggregationModeUsed` | Gets or sets which secure aggregation mode was used. |
| `SecureAggregationReconstructionThresholdUsed` | Gets or sets the reconstruction threshold used by secure aggregation (dropout-resilient modes). |
| `ServerOptimizerUsed` | Gets or sets the server-side federated optimizer used (FedOpt family). |
| `TotalClientsParticipated` | Gets or sets the total number of clients that participated across all rounds. |
| `TotalCommunicationMB` | Gets or sets the total communication cost in megabytes. |
| `TotalPrivacyBudgetConsumed` | Gets or sets the total privacy budget (epsilon) consumed. |
| `TotalPrivacyDeltaConsumed` | Gets or sets the total privacy delta consumed (basic composition reporting). |
| `TotalTrainingTimeSeconds` | Gets or sets the total training time in seconds. |
| `UnlearningPerformed` | Gets or sets whether federated unlearning was performed. |
| `UnlearningRequestsProcessed` | Gets or sets the number of unlearning requests processed. |

## Fields

| Field | Summary |
|:-----|:--------|
| `MetadataKey` | The recommended key for storing federated learning metadata inside `Properties`. |

