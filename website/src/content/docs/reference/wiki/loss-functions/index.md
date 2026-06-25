---
title: "Loss Functions"
description: "Every Loss Functions type in AiDotNet, auto-generated with compile-checked examples."
section: "Reference"
---

Every Loss Functions type in AiDotNet — each with a beginner-friendly explanation and, where the snippet compiles against the live library, a runnable example.

| Type | Summary |
|:-----|:--------|
| [`BinaryCrossEntropyLoss`](/docs/reference/wiki/loss-functions/binarycrossentropyloss/) | Implements the Binary Cross Entropy loss function for binary classification problems. |
| [`BinaryCrossEntropyWithLogitsLoss`](/docs/reference/wiki/loss-functions/binarycrossentropywithlogitsloss/) | Implements Binary Cross-Entropy loss that accepts raw logits (not probabilities). |
| [`BornRuleMseLoss`](/docs/reference/wiki/loss-functions/bornrulemseloss/) | Mean-squared error measured in *probability* space for models whose final layer emits quantum *amplitudes*: `loss = mean((predicted² − target)²)`. |
| [`CategoricalCrossEntropyLoss`](/docs/reference/wiki/loss-functions/categoricalcrossentropyloss/) | Implements the Categorical Cross Entropy loss function for multi-class classification. |
| [`CharbonnierLoss`](/docs/reference/wiki/loss-functions/charbonnierloss/) | Implements the Charbonnier loss function, a smooth approximation of L1 loss. |
| [`ContrastiveLoss`](/docs/reference/wiki/loss-functions/contrastiveloss/) | Implements the Contrastive Loss function for learning similarity metrics. |
| [`CosineSimilarityLoss`](/docs/reference/wiki/loss-functions/cosinesimilarityloss/) | Implements the Cosine Similarity Loss between two vectors. |
| [`CrossEntropyLoss`](/docs/reference/wiki/loss-functions/crossentropyloss/) | Implements the Cross Entropy loss function for multi-class classification problems. |
| [`CrossEntropyWithLogitsLoss`](/docs/reference/wiki/loss-functions/crossentropywithlogitsloss/) | Implements Cross-Entropy loss that accepts raw logits (not probabilities). |
| [`CTCLoss`](/docs/reference/wiki/loss-functions/ctcloss/) | Implements the Connectionist Temporal Classification (CTC) loss function for sequence-to-sequence learning. |
| [`DiceLoss`](/docs/reference/wiki/loss-functions/diceloss/) | Implements the Dice loss function, commonly used for image segmentation tasks. |
| [`ElasticNetLoss`](/docs/reference/wiki/loss-functions/elasticnetloss/) | Implements the Elastic Net Loss function, which combines Mean Squared Error with L1 and L2 regularization. |
| [`ExponentialLoss`](/docs/reference/wiki/loss-functions/exponentialloss/) | Implements the Exponential Loss function, commonly used in boosting algorithms. |
| [`FocalLoss`](/docs/reference/wiki/loss-functions/focalloss/) | Implements the Focal Loss function, which gives more weight to hard-to-classify examples. |
| [`HingeLoss`](/docs/reference/wiki/loss-functions/hingeloss/) | Implements the Hinge loss function commonly used in support vector machines. |
| [`HuberLoss`](/docs/reference/wiki/loss-functions/huberloss/) | Implements the Huber loss function, which combines properties of both MSE and MAE. |
| [`JaccardLoss`](/docs/reference/wiki/loss-functions/jaccardloss/) | Implements the Jaccard loss function, commonly used for measuring dissimilarity between sets. |
| [`KullbackLeiblerDivergence`](/docs/reference/wiki/loss-functions/kullbackleiblerdivergence/) | Implements the Kullback-Leibler Divergence, a measure of how one probability distribution differs from another. |
| [`LogCoshLoss`](/docs/reference/wiki/loss-functions/logcoshloss/) | Implements the Log-Cosh loss function, a smooth approximation of Mean Absolute Error. |
| [`LossFunctionBase`](/docs/reference/wiki/loss-functions/lossfunctionbase/) | Base class for loss function implementations. |
| [`MarginLoss`](/docs/reference/wiki/loss-functions/marginloss/) | Implements the Margin loss function, specifically designed for Capsule Networks. |
| [`MeanAbsoluteErrorLoss`](/docs/reference/wiki/loss-functions/meanabsoluteerrorloss/) | Implements the Mean Absolute Error (MAE) loss function. |
| [`MeanBiasErrorLoss`](/docs/reference/wiki/loss-functions/meanbiaserrorloss/) | Implements the Mean Bias Error (MBE) loss function. |
| [`MeanSquaredErrorLoss`](/docs/reference/wiki/loss-functions/meansquarederrorloss/) | Implements the Mean Squared Error (MSE) loss function. |
| [`ModifiedHuberLoss`](/docs/reference/wiki/loss-functions/modifiedhuberloss/) | Implements the Modified Huber Loss function, a smoother version of the hinge loss. |
| [`NoiseContrastiveEstimationLoss`](/docs/reference/wiki/loss-functions/noisecontrastiveestimationloss/) | Implements the Noise Contrastive Estimation (NCE) loss function for efficient training with large output spaces. |
| [`OrdinalRegressionLoss`](/docs/reference/wiki/loss-functions/ordinalregressionloss/) | Implements the Ordinal Regression Loss function for predicting ordered categories. |
| [`PairwiseRankingLoss`](/docs/reference/wiki/loss-functions/pairwiserankingloss/) | Implements the pairwise RankNet learning-to-rank loss with an optional tail-weighting knob. |
| [`PerceptualLoss`](/docs/reference/wiki/loss-functions/perceptualloss/) | Implements the Perceptual Loss function for comparing high-level features of images. |
| [`PoissonLoss`](/docs/reference/wiki/loss-functions/poissonloss/) | Implements the Poisson loss function for count data modeling. |
| [`QuantileLoss`](/docs/reference/wiki/loss-functions/quantileloss/) | Implements the Quantile loss function for quantile regression. |
| [`QuantumLoss`](/docs/reference/wiki/loss-functions/quantumloss/) | Represents a quantum-specific loss function for quantum neural networks. |
| [`RealESRGANLoss`](/docs/reference/wiki/loss-functions/realesrganloss/) | Combined loss function for Real-ESRGAN super-resolution training. |
| [`RootMeanSquaredErrorLoss`](/docs/reference/wiki/loss-functions/rootmeansquarederrorloss/) | Implements the Root Mean Squared Error (RMSE) loss function. |
| [`RotationPredictionLoss`](/docs/reference/wiki/loss-functions/rotationpredictionloss/) | Self-supervised loss function based on rotation prediction for images. |
| [`ScaleInvariantDepthLoss`](/docs/reference/wiki/loss-functions/scaleinvariantdepthloss/) | Scale-invariant depth loss function for depth estimation training. |
| [`SparseCategoricalCrossEntropyLoss`](/docs/reference/wiki/loss-functions/sparsecategoricalcrossentropyloss/) | Implements the Sparse Categorical Cross Entropy loss function for multi-class classification with integer labels. |
| [`SquaredHingeLoss`](/docs/reference/wiki/loss-functions/squaredhingeloss/) | Implements the Squared Hinge Loss function for binary classification problems. |
| [`TripletLoss`](/docs/reference/wiki/loss-functions/tripletloss/) | Implements the Triplet Loss function for learning similarity embeddings. |
| [`WassersteinLoss`](/docs/reference/wiki/loss-functions/wassersteinloss/) | Implements the Wasserstein loss function used in Wasserstein Generative Adversarial Networks (WGAN). |
| [`WeightedCrossEntropyLoss`](/docs/reference/wiki/loss-functions/weightedcrossentropyloss/) | Implements the Weighted Cross Entropy loss function for classification problems with uneven class importance. |
