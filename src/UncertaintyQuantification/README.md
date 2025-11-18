# Uncertainty Quantification Module

This module provides comprehensive uncertainty quantification and Bayesian neural network capabilities for AiDotNet, implementing features requested in Issue #418.

## Overview

Uncertainty quantification is critical for reliable AI systems, especially in safety-critical applications. This module provides several approaches to estimate and calibrate prediction uncertainty.

## Features

### 1. Bayesian Neural Networks

#### Monte Carlo Dropout (`MCDropoutNeuralNetwork<T>`)
- **Easiest to implement**: Works with existing architectures by adding `MCDropoutLayer`
- **How it works**: Keeps dropout active during inference and samples multiple predictions
- **Use case**: Quick uncertainty estimates without retraining
- **Example**:
```csharp
var architecture = new NeuralNetworkArchitecture<float>()
    .AddLayer(new DenseLayer<float>(inputSize, 128))
    .AddLayer(new MCDropoutLayer<float>(0.3, mcMode: true))
    .AddLayer(new DenseLayer<float>(128, outputSize));

var model = new MCDropoutNeuralNetwork<float>(architecture, numSamples: 50);
var (mean, uncertainty) = model.PredictWithUncertainty(input);
```

#### Bayesian Neural Network (`BayesianNeuralNetwork<T>`)
- **Most principled approach**: Uses variational inference (Bayes by Backprop)
- **How it works**: Learns probability distributions over weights instead of point estimates
- **Use case**: When you need well-calibrated uncertainty estimates and can afford retraining
- **Example**:
```csharp
var architecture = new NeuralNetworkArchitecture<float>()
    .AddLayer(new BayesianDenseLayer<float>(inputSize, 128))
    .AddLayer(new BayesianDenseLayer<float>(128, outputSize));

var model = new BayesianNeuralNetwork<float>(architecture, numSamples: 30);

// During training, add KL divergence to loss
var klDivergence = model.ComputeKLDivergence();
var totalLoss = dataLoss + beta * klDivergence;
```

#### Deep Ensemble (`DeepEnsemble<T>`)
- **Most reliable uncertainty**: Multiple independent models
- **How it works**: Train multiple models with different initializations
- **Use case**: When accuracy is paramount and computational cost is acceptable
- **Example**:
```csharp
var models = new List<INeuralNetwork<float>>();
for (int i = 0; i < 5; i++)
{
    var model = new NeuralNetwork<float>(architecture);
    model.Train(trainingData); // With different random seed
    models.Add(model);
}

var ensemble = new DeepEnsemble<float>(models);
var (mean, uncertainty) = ensemble.PredictWithUncertainty(input);
```

### 2. Uncertainty Types

All Bayesian approaches support decomposing uncertainty into:

- **Aleatoric Uncertainty**: Irreducible randomness in the data
  - Example: Sensor noise, inherent variability
  - Cannot be reduced by collecting more data

- **Epistemic Uncertainty**: Model's lack of knowledge
  - Example: Insufficient training data, out-of-distribution inputs
  - Can be reduced by collecting more training data

```csharp
var aleatoricUncertainty = model.EstimateAleatoricUncertainty(input);
var epistemicUncertainty = model.EstimateEpistemicUncertainty(input);
```

### 3. Calibration Methods

#### Temperature Scaling (`TemperatureScaling<T>`)
- **Purpose**: Calibrate neural network probabilities to match true frequencies
- **How it works**: Learns a temperature parameter that scales logits before softmax
- **When to use**: Post-training calibration for classification models
- **Example**:
```csharp
var tempScaling = new TemperatureScaling<float>();

// Calibrate on validation set
tempScaling.Calibrate(validationLogits, validationLabels);

// Apply during inference
var scaledLogits = tempScaling.ScaleLogits(logits);
var calibratedProbs = Softmax(scaledLogits);
```

#### Expected Calibration Error (`ExpectedCalibrationError<T>`)
- **Purpose**: Measure how well-calibrated your model is
- **Interpretation**:
  - ECE < 0.05: Well-calibrated
  - ECE 0.05-0.10: Moderately calibrated
  - ECE > 0.10: Poorly calibrated (needs calibration)
- **Example**:
```csharp
var ece = new ExpectedCalibrationError<float>(numBins: 10);
var calibrationError = ece.Compute(probabilities, predictions, trueLabels);

// Get reliability diagram for visualization
var diagram = ece.GetReliabilityDiagram(probabilities, predictions, trueLabels);
```

### 4. Conformal Prediction

#### Split Conformal Prediction (`SplitConformalPredictor<T>`)
- **Guarantee**: Prediction intervals contain true value with specified probability
- **Key advantage**: Works with ANY model, provides distribution-free guarantees
- **Use case**: When you need statistically valid uncertainty bounds
- **Example**:
```csharp
var conformal = new SplitConformalPredictor<float>(trainedModel);

// Calibrate on held-out calibration set
conformal.Calibrate(calibrationInputs, calibrationTargets);

// Get prediction interval with 90% coverage guarantee
var (prediction, lower, upper) = conformal.PredictWithInterval(input, confidenceLevel: 0.9);
// Interpretation: "90% confident true value is between lower and upper"

// Verify coverage on test set
var coverage = conformal.EvaluateCoverage(testInputs, testTargets, 0.9);
// Should be ≥ 0.90
```

#### Conformal Classification (`ConformalClassifier<T>`)
- **Guarantee**: Prediction sets contain true class with specified probability
- **Key advantage**: Automatically detects when model is uncertain (returns larger sets)
- **Example**:
```csharp
var conformal = new ConformalClassifier<float>(classifier, numClasses: 10);

// Calibrate
conformal.Calibrate(calibrationInputs, calibrationLabels);

// Get prediction set
var predictionSet = conformal.PredictSet(input, confidenceLevel: 0.9);
// Example outputs:
//   {2} - confident it's class 2
//   {2, 5} - uncertain between classes 2 and 5
//   {0, 1, 2, 3, 4, 5} - very uncertain (defer to human expert)
```

## Implementation Details

### Layers

1. **`MCDropoutLayer<T>`**: Dropout layer that can stay active during inference
2. **`BayesianDenseLayer<T>`**: Fully-connected layer with weight distributions

### Interfaces

1. **`IUncertaintyEstimator<T>`**: Contract for models that provide uncertainty estimates
2. **`IBayesianLayer<T>`**: Contract for Bayesian layers supporting weight sampling

## Success Criteria (from Issue #418)

- [x] Reliable uncertainty estimates achieved
- [x] ECE calibration metric implemented (target: < 0.05)
- [x] Conformal prediction with guaranteed coverage
- [x] Comprehensive implementations for:
  - [x] Monte Carlo Dropout
  - [x] Variational Inference (Bayes by Backprop)
  - [x] Deep Ensembles
  - [x] Temperature Scaling
  - [x] Expected Calibration Error
  - [x] Split Conformal Prediction
  - [x] Conformal Classification

## Testing

Comprehensive unit tests are provided in `tests/AiDotNet.Tests/UnitTests/UncertaintyQuantification/`:
- `MCDropoutLayerTests.cs`
- `TemperatureScalingTests.cs`
- `ExpectedCalibrationErrorTests.cs`

## References

1. **Monte Carlo Dropout**: Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation"
2. **Bayes by Backprop**: Blundell, C., et al. (2015). "Weight Uncertainty in Neural Networks"
3. **Deep Ensembles**: Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation"
4. **Temperature Scaling**: Guo, C., et al. (2017). "On Calibration of Modern Neural Networks"
5. **Conformal Prediction**: Vovk, V., et al. (2005). "Algorithmic Learning in a Random World"

## Future Enhancements

- Laplace Approximation
- SWAG (Stochastic Weight Averaging-Gaussian)
- Isotonic Regression calibration
- Platt Scaling
- Adaptive Conformal Inference
- Cross-Conformal Prediction
