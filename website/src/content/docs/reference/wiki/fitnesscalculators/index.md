---
title: "Fitness Calculators"
description: "All 28 public types in the AiDotNet.fitnesscalculators namespace, organized by kind."
section: "API Reference"
---

**28** public types in this namespace, organized by kind.

## Models & Types (27)

| Type | Summary |
|:-----|:--------|
| [`AdjustedRSquaredFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/adjustedrsquaredfitnesscalculator/) | A fitness calculator that uses the Adjusted R-Squared metric to evaluate model performance. |
| [`BinaryCrossEntropyLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/binarycrossentropylossfitnesscalculator/) | A fitness calculator that uses Binary Cross-Entropy Loss to evaluate model performance for binary classification problems. |
| [`CategoricalCrossEntropyLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/categoricalcrossentropylossfitnesscalculator/) | A fitness calculator that uses Categorical Cross-Entropy Loss to evaluate model performance for multi-class classification problems. |
| [`CompressionAwareFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/compressionawarefitnesscalculator/) | A fitness calculator that considers both model accuracy and compression effectiveness. |
| [`ContrastiveLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/contrastivelossfitnesscalculator/) | A fitness calculator that uses Contrastive Loss to evaluate model performance for similarity learning tasks. |
| [`CosineSimilarityLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/cosinesimilaritylossfitnesscalculator/) | A fitness calculator that uses Cosine Similarity Loss to evaluate model performance for tasks where the direction of vectors matters more than their magnitude. |
| [`CrossEntropyLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/crossentropylossfitnesscalculator/) | A fitness calculator that uses Cross Entropy Loss to evaluate model performance for classification tasks. |
| [`DiceLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/dicelossfitnesscalculator/) | A fitness calculator that uses Dice Loss to evaluate model performance for image segmentation and other tasks where overlap between predictions and actual values is important. |
| [`ElasticNetLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/elasticnetlossfitnesscalculator/) | A fitness calculator that uses Elastic Net Loss to evaluate model performance while encouraging simpler models through regularization. |
| [`ExponentialLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/exponentiallossfitnesscalculator/) | A fitness calculator that uses Exponential Loss to evaluate model performance, particularly for classification tasks. |
| [`FocalLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/focallossfitnesscalculator/) | A fitness calculator that uses Focal Loss to evaluate model performance, particularly for imbalanced classification problems. |
| [`HingeLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/hingelossfitnesscalculator/) | A fitness calculator that uses Hinge Loss to evaluate model performance, particularly for binary classification and support vector machines. |
| [`HuberLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/huberlossfitnesscalculator/) | A fitness calculator that uses Huber Loss to evaluate model performance, combining the best aspects of Mean Squared Error and Mean Absolute Error. |
| [`JaccardLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/jaccardlossfitnesscalculator/) | A fitness calculator that uses Jaccard Loss to evaluate model performance, particularly for segmentation and classification tasks. |
| [`KullbackLeiblerDivergenceFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/kullbackleiblerdivergencefitnesscalculator/) | A fitness calculator that uses Kullback-Leibler Divergence to evaluate model performance, particularly for probability distributions. |
| [`LogCoshLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/logcoshlossfitnesscalculator/) | A fitness calculator that uses Log-Cosh Loss to evaluate model performance, particularly for regression tasks. |
| [`MeanAbsoluteErrorFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/meanabsoluteerrorfitnesscalculator/) | A fitness calculator that uses Mean Absolute Error (MAE) to evaluate model performance, particularly for regression tasks. |
| [`MeanSquaredErrorFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/meansquarederrorfitnesscalculator/) | A fitness calculator that uses Mean Squared Error (MSE) to evaluate model performance, particularly for regression tasks. |
| [`ModifiedHuberLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/modifiedhuberlossfitnesscalculator/) | A fitness calculator that uses Modified Huber Loss to evaluate model performance, particularly for classification tasks. |
| [`OrdinalRegressionLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/ordinalregressionlossfitnesscalculator/) | A fitness calculator that uses Ordinal Regression Loss to evaluate model performance, particularly for ordinal classification tasks. |
| [`PoissonLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/poissonlossfitnesscalculator/) | A fitness calculator that uses Poisson Loss to evaluate model performance, particularly for count-based prediction tasks. |
| [`QuantileLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/quantilelossfitnesscalculator/) |  |
| [`RSquaredFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/rsquaredfitnesscalculator/) | A fitness calculator that uses R-Squared (R²) to evaluate model performance. |
| [`RootMeanSquaredErrorFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/rootmeansquarederrorfitnesscalculator/) | A fitness calculator that uses Root Mean Squared Error (RMSE) to evaluate model performance. |
| [`SquaredHingeLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/squaredhingelossfitnesscalculator/) | A fitness calculator that uses Squared Hinge Loss to evaluate model performance, particularly for binary classification tasks. |
| [`TripletLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/tripletlossfitnesscalculator/) | A fitness calculator that uses Triplet Loss to evaluate model performance, particularly for similarity learning and embedding tasks. |
| [`WeightedCrossEntropyLossFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/weightedcrossentropylossfitnesscalculator/) | A fitness calculator that uses Weighted Cross Entropy Loss to evaluate model performance, particularly for classification problems with imbalanced classes. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`FitnessCalculatorBase<T, TInput, TOutput>`](/docs/reference/wiki/fitnesscalculators/fitnesscalculatorbase/) | Base class for all fitness calculators that evaluate how well a model performs. |

