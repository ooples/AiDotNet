---
title: "Uncertainty Quantification"
description: "All 11 public types in the AiDotNet.uncertaintyquantification namespace, organized by kind."
section: "API Reference"
---

**11** public types in this namespace, organized by kind.

## Models & Types (7)

| Type | Summary |
|:-----|:--------|
| [`BayesianNeuralNetwork<T>`](/docs/reference/wiki/uncertaintyquantification/bayesianneuralnetwork/) | Implements a Bayesian Neural Network that provides uncertainty estimates with predictions. |
| [`ConformalClassifier<T>`](/docs/reference/wiki/uncertaintyquantification/conformalclassifier/) | Implements Conformal Prediction for classification tasks. |
| [`DeepEnsemble<T>`](/docs/reference/wiki/uncertaintyquantification/deepensemble/) | Implements Deep Ensembles for uncertainty estimation. |
| [`ExpectedCalibrationError<T>`](/docs/reference/wiki/uncertaintyquantification/expectedcalibrationerror/) |  |
| [`MCDropoutNeuralNetwork<T>`](/docs/reference/wiki/uncertaintyquantification/mcdropoutneuralnetwork/) | Implements Monte Carlo Dropout for uncertainty estimation. |
| [`SplitConformalPredictor<T>`](/docs/reference/wiki/uncertaintyquantification/splitconformalpredictor/) | Implements Split Conformal Prediction for regression tasks. |
| [`TemperatureScaling<T>`](/docs/reference/wiki/uncertaintyquantification/temperaturescaling/) |  |

## Layers (2)

| Type | Summary |
|:-----|:--------|
| [`BayesianDenseLayer<T>`](/docs/reference/wiki/uncertaintyquantification/bayesiandenselayer/) | Implements a Bayesian dense (fully-connected) layer using variational inference. |
| [`MCDropoutLayer<T>`](/docs/reference/wiki/uncertaintyquantification/mcdropoutlayer/) | Implements Monte Carlo Dropout layer for uncertainty estimation in neural networks. |

## Interfaces (2)

| Type | Summary |
|:-----|:--------|
| [`IBayesianLayer<T>`](/docs/reference/wiki/uncertaintyquantification/ibayesianlayer/) | Defines the contract for Bayesian neural network layers that support probabilistic inference. |
| [`IUncertaintyEstimator<T>`](/docs/reference/wiki/uncertaintyquantification/iuncertaintyestimator/) | Defines the contract for models that can estimate prediction uncertainty. |

