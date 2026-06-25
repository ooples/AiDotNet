---
title: "LossType"
description: "Defines different loss functions used to measure how well a model's predictions match the actual values."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines different loss functions used to measure how well a model's predictions match the actual values.

## For Beginners

Loss functions are like a "scoring system" for AI models. They measure the difference
between what the model predicted and what the correct answer actually is. A lower loss means the model
is making better predictions. During training, the optimizer adjusts the model's parameters to minimize
this loss value, gradually improving the model's accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `BinaryCrossEntropy` | Binary Cross Entropy - cross entropy specialized for two-class (yes/no) problems. |
| `CTC` | CTC Loss - Connectionist Temporal Classification for sequence-to-sequence alignment. |
| `CategoricalCrossEntropy` | Categorical Cross Entropy - cross entropy for multi-class classification with one-hot encoded labels. |
| `Charbonnier` | Charbonnier Loss - a differentiable variant of L1 loss, popular in image restoration. |
| `Contrastive` | Contrastive Loss - learns embeddings by pulling similar pairs together and pushing dissimilar pairs apart. |
| `CosineSimilarity` | Cosine Similarity Loss - measures the cosine of the angle between prediction and target vectors. |
| `CrossEntropy` | Cross Entropy - measures the difference between two probability distributions. |
| `Dice` | Dice Loss - measures overlap between predicted and actual segmentation masks. |
| `ElasticNet` | Elastic Net Loss - combines L1 (Lasso) and L2 (Ridge) regularization penalties. |
| `Exponential` | Exponential Loss - used in boosting algorithms like AdaBoost. |
| `Focal` | Focal Loss - gives more weight to hard-to-classify examples, reducing the impact of easy ones. |
| `Hinge` | Hinge Loss - used for "maximum-margin" classification, especially with Support Vector Machines. |
| `Huber` | Huber Loss - combines MSE for small errors and MAE for large errors. |
| `Jaccard` | Jaccard Loss - also known as Intersection over Union (IoU) loss for segmentation tasks. |
| `KullbackLeiblerDivergence` | Kullback-Leibler Divergence - measures how one probability distribution diverges from another. |
| `LogCosh` | Log-Cosh Loss - the logarithm of the hyperbolic cosine of the prediction error. |
| `Margin` | Margin Loss - used in capsule networks with separate margins for positive and negative classes. |
| `MeanAbsoluteError` | Mean Absolute Error - measures the average absolute difference between predictions and actual values. |
| `MeanBiasError` | Mean Bias Error - measures the average bias (systematic over/under-prediction) of the model. |
| `MeanSquaredError` | Mean Squared Error - measures the average squared difference between predictions and actual values. |
| `ModifiedHuber` | Modified Huber Loss - a smooth approximation combining hinge loss and quadratic loss. |
| `NoiseContrastiveEstimation` | Noise Contrastive Estimation Loss - approximates softmax for large vocabulary problems. |
| `OrdinalRegression` | Ordinal Regression Loss - preserves the ordering relationship between classes. |
| `Perceptual` | Perceptual Loss - compares feature representations from a pretrained network. |
| `Poisson` | Poisson Loss - appropriate for count data that follows a Poisson distribution. |
| `Quantile` | Quantile Loss - predicts specific percentiles rather than the mean. |
| `Quantum` | Quantum Loss - a loss function inspired by quantum computing principles. |
| `RealESRGAN` | RealESRGAN Loss - a composite loss for Real-ESRGAN super-resolution models. |
| `RootMeanSquaredError` | Root Mean Squared Error - the square root of the average squared differences. |
| `RotationPrediction` | Rotation Prediction Loss - a self-supervised loss that predicts image rotation angles. |
| `ScaleInvariantDepth` | Scale Invariant Depth Loss - measures depth prediction quality regardless of absolute scale. |
| `SparseCategoricalCrossEntropy` | Sparse Categorical Cross Entropy - like categorical cross entropy but with integer labels instead of one-hot. |
| `SquaredHinge` | Squared Hinge Loss - a smoother variant of hinge loss that squares the penalty. |
| `Triplet` | Triplet Loss - learns embeddings using anchor, positive, and negative examples. |
| `Wasserstein` | Wasserstein Loss - based on the Earth Mover's Distance, commonly used in GANs. |
| `WeightedCrossEntropy` | Weighted Cross Entropy Loss - cross entropy with class-specific weights. |

