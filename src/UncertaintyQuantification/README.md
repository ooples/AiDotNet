# Uncertainty Quantification

Uncertainty quantification (UQ) augments standard point predictions with an uncertainty estimate, which is critical for risk-aware and safety-critical systems.

## Overview

This module is designed to integrate with AiDotNet’s facade workflow:

- Build/train via `AiModelBuilder`
- Run inference via `AiModelResult`

The primary facade entrypoint is `AiModelResult.PredictWithUncertainty(...)`, which returns a `UncertaintyPredictionResult` containing a point prediction, optional variance, and additional diagnostics (e.g., entropy / mutual information for classification-like outputs).

## Quick Start (Facade)

### 1. Configure and build a model

```csharp
using AiDotNet;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

var architecture = new NeuralNetworkArchitecture<double>(inputFeatures: 2, outputSize: 1);
var model = new NeuralNetworkModel<double>(architecture);

var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureUncertaintyQuantification(new UncertaintyQuantificationOptions
    {
        Method = AiDotNet.Enums.UncertaintyQuantificationMethod.MonteCarloDropout,
        NumSamples = 30,
        MonteCarloDropoutRate = 0.1
    });

var x = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
{
    { 0.0, 0.0 },
    { 0.0, 1.0 },
    { 1.0, 0.0 },
    { 1.0, 1.0 }
}));

var y = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 }));

var result = builder.Build(x, y);
```

### 2. Predict with uncertainty

```csharp
var uq = result.PredictWithUncertainty(x);
var mean = uq.Prediction;
var variance = uq.Variance;
```

## Notes on Behavior

- The uncertainty output is a variance estimate (same shape as the prediction).
- When output denormalization is affine (e.g., MinMax, ZScore, RobustScaling), variance is scaled accordingly. For non-linear normalization transforms, variance is returned in normalized space.
- If uncertainty quantification is not configured (or cannot be applied), `PredictWithUncertainty` returns the deterministic prediction with an all-zero variance tensor.

## Advanced Building Blocks

This module also contains reusable components for advanced users and internal integrations:

- Bayesian neural network layers (e.g., `BayesianDenseLayer<T>`)
- Calibration utilities (e.g., `TemperatureScaling<T>`, `ExpectedCalibrationError<T>`)
- Conformal prediction utilities (e.g., `SplitConformalPredictor<T>`, `ConformalClassifier<T>`)

## Testing

- Unit tests: `tests/AiDotNet.Tests/UnitTests/UncertaintyQuantification/`
- Facade integration tests: `tests/AiDotNet.Tests/IntegrationTests/UncertaintyQuantificationFacadeTests.cs`
