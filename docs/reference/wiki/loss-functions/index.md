---
title: "Loss Functions"
description: "Every Loss Functions type in AiDotNet, auto-generated with compile-checked examples."
section: "Reference"
---

Every Loss Functions type in AiDotNet — each with a beginner-friendly explanation and, where the snippet compiles against the live library, a runnable example.

| Type | Summary |
|:-----|:--------|
| [`BinaryCrossEntropyLoss`](./BinaryCrossEntropyLoss.md) | Implements the Binary Cross Entropy loss function for binary classification problems. |
| [`BinaryCrossEntropyWithLogitsLoss`](./BinaryCrossEntropyWithLogitsLoss.md) | Implements Binary Cross-Entropy loss that accepts raw logits (not probabilities). |
| [`BornRuleMseLoss`](./BornRuleMseLoss.md) | Mean-squared error measured in *probability* space for models whose final layer emits quantum *amplitudes*: `loss = mean((predicted² − target)²)`. |
| [`CategoricalCrossEntropyLoss`](./CategoricalCrossEntropyLoss.md) | Implements the Categorical Cross Entropy loss function for multi-class classification. |
| [`CharbonnierLoss`](./CharbonnierLoss.md) | Implements the Charbonnier loss function, a smooth approximation of L1 loss. |
| [`ContrastiveLoss`](./ContrastiveLoss.md) | Implements the Contrastive Loss function for learning similarity metrics. |
| [`CosineSimilarityLoss`](./CosineSimilarityLoss.md) | Implements the Cosine Similarity Loss between two vectors. |
| [`CrossEntropyLoss`](./CrossEntropyLoss.md) | Implements the Cross Entropy loss function for multi-class classification problems. |
| [`CrossEntropyWithLogitsLoss`](./CrossEntropyWithLogitsLoss.md) | Implements Cross-Entropy loss that accepts raw logits (not probabilities). |
| [`CTCLoss`](./CTCLoss.md) | Implements the Connectionist Temporal Classification (CTC) loss function for sequence-to-sequence learning. |
| [`DiceLoss`](./DiceLoss.md) | Implements the Dice loss function, commonly used for image segmentation tasks. |
| [`ElasticNetLoss`](./ElasticNetLoss.md) | Implements the Elastic Net Loss function, which combines Mean Squared Error with L1 and L2 regularization. |
| [`ExponentialLoss`](./ExponentialLoss.md) | Implements the Exponential Loss function, commonly used in boosting algorithms. |
| [`FocalLoss`](./FocalLoss.md) | Implements the Focal Loss function, which gives more weight to hard-to-classify examples. |
| [`HingeLoss`](./HingeLoss.md) | Implements the Hinge loss function commonly used in support vector machines. |
| [`HuberLoss`](./HuberLoss.md) | Implements the Huber loss function, which combines properties of both MSE and MAE. |
| [`JaccardLoss`](./JaccardLoss.md) | Implements the Jaccard loss function, commonly used for measuring dissimilarity between sets. |
| [`KullbackLeiblerDivergence`](./KullbackLeiblerDivergence.md) | Implements the Kullback-Leibler Divergence, a measure of how one probability distribution differs from another. |
| [`LogCoshLoss`](./LogCoshLoss.md) | Implements the Log-Cosh loss function, a smooth approximation of Mean Absolute Error. |
| [`LossFunctionBase`](./LossFunctionBase.md) | Base class for loss function implementations. |
| [`MarginLoss`](./MarginLoss.md) | Implements the Margin loss function, specifically designed for Capsule Networks. |
| [`MeanAbsoluteErrorLoss`](./MeanAbsoluteErrorLoss.md) | Implements the Mean Absolute Error (MAE) loss function. |
| [`MeanBiasErrorLoss`](./MeanBiasErrorLoss.md) | Implements the Mean Bias Error (MBE) loss function. |
| [`MeanSquaredErrorLoss`](./MeanSquaredErrorLoss.md) | Implements the Mean Squared Error (MSE) loss function. |
| [`ModifiedHuberLoss`](./ModifiedHuberLoss.md) | Implements the Modified Huber Loss function, a smoother version of the hinge loss. |
| [`NoiseContrastiveEstimationLoss`](./NoiseContrastiveEstimationLoss.md) | Implements the Noise Contrastive Estimation (NCE) loss function for efficient training with large output spaces. |
| [`OrdinalRegressionLoss`](./OrdinalRegressionLoss.md) | Implements the Ordinal Regression Loss function for predicting ordered categories. |
| [`PairwiseRankingLoss`](./PairwiseRankingLoss.md) | Implements the pairwise RankNet learning-to-rank loss with an optional tail-weighting knob. |
| [`PerceptualLoss`](./PerceptualLoss.md) | Implements the Perceptual Loss function for comparing high-level features of images. |
| [`PoissonLoss`](./PoissonLoss.md) | Implements the Poisson loss function for count data modeling. |
| [`QuantileLoss`](./QuantileLoss.md) | Implements the Quantile loss function for quantile regression. |
| [`QuantumLoss`](./QuantumLoss.md) | Represents a quantum-specific loss function for quantum neural networks. |
| [`RealESRGANLoss`](./RealESRGANLoss.md) | Combined loss function for Real-ESRGAN super-resolution training. |
| [`RootMeanSquaredErrorLoss`](./RootMeanSquaredErrorLoss.md) | Implements the Root Mean Squared Error (RMSE) loss function. |
| [`RotationPredictionLoss`](./RotationPredictionLoss.md) | Self-supervised loss function based on rotation prediction for images. |
| [`ScaleInvariantDepthLoss`](./ScaleInvariantDepthLoss.md) | Scale-invariant depth loss function for depth estimation training. |
| [`SparseCategoricalCrossEntropyLoss`](./SparseCategoricalCrossEntropyLoss.md) | Implements the Sparse Categorical Cross Entropy loss function for multi-class classification with integer labels. |
| [`SquaredHingeLoss`](./SquaredHingeLoss.md) | Implements the Squared Hinge Loss function for binary classification problems. |
| [`TripletLoss`](./TripletLoss.md) | Implements the Triplet Loss function for learning similarity embeddings. |
| [`WassersteinLoss`](./WassersteinLoss.md) | Implements the Wasserstein loss function used in Wasserstein Generative Adversarial Networks (WGAN). |
| [`WeightedCrossEntropyLoss`](./WeightedCrossEntropyLoss.md) | Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance. |
