---
title: "Loss Functions"
description: "All 41 public types in the AiDotNet.lossfunctions namespace, organized by kind."
section: "API Reference"
---

**41** public types in this namespace, organized by kind.

## Models & Types (40)

| Type | Summary |
|:-----|:--------|
| [`BinaryCrossEntropyLoss<T>`](/docs/reference/wiki/lossfunctions/binarycrossentropyloss/) | Implements the Binary Cross Entropy loss function for binary classification problems. |
| [`BinaryCrossEntropyWithLogitsLoss<T>`](/docs/reference/wiki/lossfunctions/binarycrossentropywithlogitsloss/) | Implements Binary Cross-Entropy loss that accepts raw logits (not probabilities). |
| [`BornRuleMseLoss<T>`](/docs/reference/wiki/lossfunctions/bornrulemseloss/) | Mean-squared error measured in *probability* space for models whose final layer emits quantum *amplitudes*: `loss = mean((predicted² − target)²)`. |
| [`CTCLoss<T>`](/docs/reference/wiki/lossfunctions/ctcloss/) | Implements the Connectionist Temporal Classification (CTC) loss function for sequence-to-sequence learning. |
| [`CategoricalCrossEntropyLoss<T>`](/docs/reference/wiki/lossfunctions/categoricalcrossentropyloss/) | Implements the Categorical Cross Entropy loss function for multi-class classification. |
| [`CharbonnierLoss<T>`](/docs/reference/wiki/lossfunctions/charbonnierloss/) | Implements the Charbonnier loss function, a smooth approximation of L1 loss. |
| [`ContrastiveLoss<T>`](/docs/reference/wiki/lossfunctions/contrastiveloss/) | Implements the Contrastive Loss function for learning similarity metrics. |
| [`CosineSimilarityLoss<T>`](/docs/reference/wiki/lossfunctions/cosinesimilarityloss/) | Implements the Cosine Similarity Loss between two vectors. |
| [`CrossEntropyLoss<T>`](/docs/reference/wiki/lossfunctions/crossentropyloss/) | Implements the Cross Entropy loss function for multi-class classification problems. |
| [`CrossEntropyWithLogitsLoss<T>`](/docs/reference/wiki/lossfunctions/crossentropywithlogitsloss/) | Implements Cross-Entropy loss that accepts raw logits (not probabilities). |
| [`DiceLoss<T>`](/docs/reference/wiki/lossfunctions/diceloss/) | Implements the Dice loss function, commonly used for image segmentation tasks. |
| [`ElasticNetLoss<T>`](/docs/reference/wiki/lossfunctions/elasticnetloss/) | Implements the Elastic Net Loss function, which combines Mean Squared Error with L1 and L2 regularization. |
| [`ExponentialLoss<T>`](/docs/reference/wiki/lossfunctions/exponentialloss/) | Implements the Exponential Loss function, commonly used in boosting algorithms. |
| [`FocalLoss<T>`](/docs/reference/wiki/lossfunctions/focalloss/) | Implements the Focal Loss function, which gives more weight to hard-to-classify examples. |
| [`HingeLoss<T>`](/docs/reference/wiki/lossfunctions/hingeloss/) | Implements the Hinge loss function commonly used in support vector machines. |
| [`HuberLoss<T>`](/docs/reference/wiki/lossfunctions/huberloss/) | Implements the Huber loss function, which combines properties of both MSE and MAE. |
| [`JaccardLoss<T>`](/docs/reference/wiki/lossfunctions/jaccardloss/) | Implements the Jaccard loss function, commonly used for measuring dissimilarity between sets. |
| [`KullbackLeiblerDivergence<T>`](/docs/reference/wiki/lossfunctions/kullbackleiblerdivergence/) | Implements the Kullback-Leibler Divergence, a measure of how one probability distribution differs from another. |
| [`LogCoshLoss<T>`](/docs/reference/wiki/lossfunctions/logcoshloss/) | Implements the Log-Cosh loss function, a smooth approximation of Mean Absolute Error. |
| [`MarginLoss<T>`](/docs/reference/wiki/lossfunctions/marginloss/) | Implements the Margin loss function, specifically designed for Capsule Networks. |
| [`MeanAbsoluteErrorLoss<T>`](/docs/reference/wiki/lossfunctions/meanabsoluteerrorloss/) | Implements the Mean Absolute Error (MAE) loss function. |
| [`MeanBiasErrorLoss<T>`](/docs/reference/wiki/lossfunctions/meanbiaserrorloss/) | Implements the Mean Bias Error (MBE) loss function. |
| [`MeanSquaredErrorLoss<T>`](/docs/reference/wiki/lossfunctions/meansquarederrorloss/) | Implements the Mean Squared Error (MSE) loss function. |
| [`ModifiedHuberLoss<T>`](/docs/reference/wiki/lossfunctions/modifiedhuberloss/) | Implements the Modified Huber Loss function, a smoother version of the hinge loss. |
| [`NoiseContrastiveEstimationLoss<T>`](/docs/reference/wiki/lossfunctions/noisecontrastiveestimationloss/) | Implements the Noise Contrastive Estimation (NCE) loss function for efficient training with large output spaces. |
| [`OrdinalRegressionLoss<T>`](/docs/reference/wiki/lossfunctions/ordinalregressionloss/) | Implements the Ordinal Regression Loss function for predicting ordered categories. |
| [`PairwiseRankingLoss<T>`](/docs/reference/wiki/lossfunctions/pairwiserankingloss/) | Implements the pairwise RankNet learning-to-rank loss with an optional tail-weighting knob. |
| [`PerceptualLoss<T>`](/docs/reference/wiki/lossfunctions/perceptualloss/) | Implements the Perceptual Loss function for comparing high-level features of images. |
| [`PoissonLoss<T>`](/docs/reference/wiki/lossfunctions/poissonloss/) | Implements the Poisson loss function for count data modeling. |
| [`QuantileLoss<T>`](/docs/reference/wiki/lossfunctions/quantileloss/) | Implements the Quantile loss function for quantile regression. |
| [`QuantumLoss<T>`](/docs/reference/wiki/lossfunctions/quantumloss/) | Represents a quantum-specific loss function for quantum neural networks. |
| [`RealESRGANLoss<T>`](/docs/reference/wiki/lossfunctions/realesrganloss/) | Combined loss function for Real-ESRGAN super-resolution training. |
| [`RootMeanSquaredErrorLoss<T>`](/docs/reference/wiki/lossfunctions/rootmeansquarederrorloss/) | Implements the Root Mean Squared Error (RMSE) loss function. |
| [`RotationPredictionLoss<T>`](/docs/reference/wiki/lossfunctions/rotationpredictionloss/) | Self-supervised loss function based on rotation prediction for images. |
| [`ScaleInvariantDepthLoss<T>`](/docs/reference/wiki/lossfunctions/scaleinvariantdepthloss/) | Scale-invariant depth loss function for depth estimation training. |
| [`SparseCategoricalCrossEntropyLoss<T>`](/docs/reference/wiki/lossfunctions/sparsecategoricalcrossentropyloss/) | Implements the Sparse Categorical Cross Entropy loss function for multi-class classification with integer labels. |
| [`SquaredHingeLoss<T>`](/docs/reference/wiki/lossfunctions/squaredhingeloss/) | Implements the Squared Hinge Loss function for binary classification problems. |
| [`TripletLoss<T>`](/docs/reference/wiki/lossfunctions/tripletloss/) | Implements the Triplet Loss function for learning similarity embeddings. |
| [`WassersteinLoss<T>`](/docs/reference/wiki/lossfunctions/wassersteinloss/) | Implements the Wasserstein loss function used in Wasserstein Generative Adversarial Networks (WGAN). |
| [`WeightedCrossEntropyLoss<T>`](/docs/reference/wiki/lossfunctions/weightedcrossentropyloss/) | Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`LossFunctionBase<T>`](/docs/reference/wiki/lossfunctions/lossfunctionbase/) | Base class for loss function implementations. |

