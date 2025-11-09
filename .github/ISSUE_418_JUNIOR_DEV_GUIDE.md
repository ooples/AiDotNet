# Issue #418: Junior Developer Implementation Guide

## Understanding Uncertainty Quantification and Bayesian Neural Networks

**Goal**: Implement methods to quantify how confident the model is about its predictions, critical for production AI systems where knowing when the model is uncertain can prevent catastrophic failures.

---

## Key Concepts for Beginners

### What is Uncertainty?

**Two Types of Uncertainty**:

1. **Aleatoric Uncertainty** (Data Uncertainty):
   - Uncertainty inherent in the data itself
   - Example: Image is blurry, so classification is inherently uncertain
   - Cannot be reduced by adding more training data
   - Example: Predicting tomorrow's weather has inherent randomness

2. **Epistemic Uncertainty** (Model Uncertainty):
   - Uncertainty due to lack of knowledge/data
   - Example: Model hasn't seen enough examples of this class
   - Can be reduced with more training data
   - Example: Model uncertainty about rare medical conditions

### What are Bayesian Neural Networks?

**Traditional Neural Network**:
```
Weight W = 0.5 (single fixed value)
```

**Bayesian Neural Network**:
```
Weight W ~ Normal(mean=0.5, std=0.1) (distribution)
```

Instead of learning single weight values, we learn distributions over weights. This naturally provides uncertainty estimates.

---

## Phase 1: MC Dropout for Uncertainty Estimation

### AC 1.1: Add DropoutLayer.EnableTestTimeDropout Property

**What is MC Dropout?**
- During training: Dropout randomly zeros neurons (prevents overfitting)
- During testing: Usually dropout is disabled
- **MC Dropout trick**: Keep dropout enabled during testing, run multiple forward passes, get different predictions each time, use variation as uncertainty estimate

**File**: `src/Layers/DropoutLayer.cs`

**Step 1**: Add property to control test-time dropout

```csharp
// File: src/Layers/DropoutLayer.cs
namespace AiDotNet.Layers;

public class DropoutLayer<T> : LayerBase<T>
{
    private readonly double _dropoutRate;
    private readonly Random _random;
    private Matrix<T>? _mask;

    // NEW: Property to enable dropout during inference (for MC Dropout)
    /// <summary>
    /// When true, applies dropout during inference for uncertainty estimation (MC Dropout).
    /// Default is false (standard behavior: dropout only during training).
    /// </summary>
    public bool EnableTestTimeDropout { get; set; } = false;

    public DropoutLayer(double dropoutRate)
    {
        if (dropoutRate < 0.0 || dropoutRate >= 1.0)
            throw new ArgumentException("Dropout rate must be in [0, 1)", nameof(dropoutRate));

        _dropoutRate = dropoutRate;
        _random = new Random();
    }

    public override Matrix<T> Forward(Matrix<T> input, bool training = false)
    {
        var numOps = NumericOperations<T>.Instance;

        // Apply dropout if training OR if test-time dropout is enabled
        bool shouldApplyDropout = training || EnableTestTimeDropout;

        if (!shouldApplyDropout)
        {
            // Standard inference: no dropout
            return input;
        }

        // Apply dropout
        int rows = input.Rows;
        int cols = input.Columns;

        _mask = new Matrix<T>(rows, cols);
        var output = new Matrix<T>(rows, cols);

        // Scale factor to maintain expected value
        T scale = numOps.FromDouble(1.0 / (1.0 - _dropoutRate));
        T zero = numOps.Zero;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                // Randomly drop with probability dropoutRate
                bool keep = _random.NextDouble() >= _dropoutRate;

                if (keep)
                {
                    // Keep and scale
                    _mask[r, c] = numOps.One;
                    output[r, c] = numOps.Multiply(input[r, c], scale);
                }
                else
                {
                    // Drop
                    _mask[r, c] = zero;
                    output[r, c] = zero;
                }
            }
        }

        return output;
    }

    public override Matrix<T> Backward(Matrix<T> gradient)
    {
        if (_mask == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        var numOps = NumericOperations<T>.Instance;
        var output = new Matrix<T>(gradient.Rows, gradient.Columns);

        T scale = numOps.FromDouble(1.0 / (1.0 - _dropoutRate));

        for (int r = 0; r < gradient.Rows; r++)
        {
            for (int c = 0; c < gradient.Columns; c++)
            {
                // Only propagate gradient where neurons weren't dropped
                output[r, c] = numOps.Multiply(
                    numOps.Multiply(gradient[r, c], _mask[r, c]),
                    scale
                );
            }
        }

        return output;
    }
}
```

**Step 2**: Create unit test

```csharp
// File: tests/UnitTests/Layers/DropoutLayerTests.cs
namespace AiDotNet.Tests.Layers;

public class DropoutLayerTests
{
    [Fact]
    public void Forward_TrainingFalse_EnableTestTimeDropoutFalse_NoDropout()
    {
        // Arrange
        var layer = new DropoutLayer<double>(0.5);
        layer.EnableTestTimeDropout = false; // Disabled

        var input = new Matrix<double>(2, 3);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 3; c++)
                input[r, c] = 1.0;

        // Act
        var output = layer.Forward(input, training: false);

        // Assert - should be identical (no dropout)
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(1.0, output[r, c]);
            }
        }
    }

    [Fact]
    public void Forward_TrainingFalse_EnableTestTimeDropoutTrue_AppliesDropout()
    {
        // Arrange
        var layer = new DropoutLayer<double>(0.5);
        layer.EnableTestTimeDropout = true; // Enabled for MC Dropout

        var input = new Matrix<double>(10, 10);
        for (int r = 0; r < 10; r++)
            for (int c = 0; c < 10; c++)
                input[r, c] = 1.0;

        // Act
        var output = layer.Forward(input, training: false);

        // Assert - should have some zeros (dropout applied)
        int zeroCount = 0;
        for (int r = 0; r < 10; r++)
        {
            for (int c = 0; c < 10; c++)
            {
                if (output[r, c] == 0.0)
                    zeroCount++;
            }
        }

        // With 50% dropout on 100 elements, expect roughly 40-60 zeros
        Assert.InRange(zeroCount, 30, 70);
    }
}
```

---

### AC 1.2: Implement MCDropoutPredictor

**What does this do?**
Runs multiple forward passes with dropout enabled, collects predictions, computes mean and variance.

**File**: `src/Uncertainty/MCDropoutPredictor.cs`

```csharp
// File: src/Uncertainty/MCDropoutPredictor.cs
namespace AiDotNet.Uncertainty;

/// <summary>
/// Performs Monte Carlo Dropout for uncertainty estimation.
/// Runs multiple forward passes with dropout enabled and aggregates predictions.
/// </summary>
public class MCDropoutPredictor<T>
{
    private readonly IModel<T> _model;
    private readonly int _numSamples;

    /// <summary>
    /// Creates predictor with MC Dropout.
    /// </summary>
    /// <param name="model">Neural network model with dropout layers</param>
    /// <param name="numSamples">Number of forward passes (typically 10-100)</param>
    public MCDropoutPredictor(IModel<T> model, int numSamples = 50)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));

        if (numSamples < 2)
            throw new ArgumentException("Must use at least 2 samples", nameof(numSamples));

        _numSamples = numSamples;
    }

    /// <summary>
    /// Predict with uncertainty using MC Dropout.
    /// </summary>
    /// <param name="input">Input data</param>
    /// <returns>Prediction result with mean, variance, and epistemic uncertainty</returns>
    public UncertaintyPrediction<T> PredictWithUncertainty(Matrix<T> input)
    {
        // Enable test-time dropout for all dropout layers
        EnableTestTimeDropout(true);

        try
        {
            var numOps = NumericOperations<T>.Instance;

            // Collect predictions from multiple forward passes
            var predictions = new List<Matrix<T>>();

            for (int i = 0; i < _numSamples; i++)
            {
                var prediction = _model.Forward(input, training: false);
                predictions.Add(prediction);
            }

            // Compute mean prediction
            var mean = ComputeMean(predictions);

            // Compute variance (measure of uncertainty)
            var variance = ComputeVariance(predictions, mean);

            // Epistemic uncertainty = average variance
            double epistemicUncertainty = ComputeAverageValue(variance);

            return new UncertaintyPrediction<T>
            {
                Mean = mean,
                Variance = variance,
                EpistemicUncertainty = epistemicUncertainty,
                Predictions = predictions
            };
        }
        finally
        {
            // Restore normal inference mode
            EnableTestTimeDropout(false);
        }
    }

    private void EnableTestTimeDropout(bool enable)
    {
        // Find all dropout layers in the model and set EnableTestTimeDropout
        var layers = GetAllLayers(_model);

        foreach (var layer in layers)
        {
            if (layer is DropoutLayer<T> dropoutLayer)
            {
                dropoutLayer.EnableTestTimeDropout = enable;
            }
        }
    }

    private List<ILayer<T>> GetAllLayers(IModel<T> model)
    {
        var layers = new List<ILayer<T>>();

        // Use reflection to get Layers property from Sequential model
        var layersProperty = model.GetType().GetProperty("Layers");
        if (layersProperty != null)
        {
            var layersList = layersProperty.GetValue(model) as IEnumerable<ILayer<T>>;
            if (layersList != null)
            {
                layers.AddRange(layersList);
            }
        }

        return layers;
    }

    private Matrix<T> ComputeMean(List<Matrix<T>> predictions)
    {
        var numOps = NumericOperations<T>.Instance;

        int rows = predictions[0].Rows;
        int cols = predictions[0].Columns;

        var mean = new Matrix<T>(rows, cols);
        T count = numOps.FromDouble(_numSamples);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                T sum = numOps.Zero;

                foreach (var pred in predictions)
                {
                    sum = numOps.Add(sum, pred[r, c]);
                }

                mean[r, c] = numOps.Divide(sum, count);
            }
        }

        return mean;
    }

    private Matrix<T> ComputeVariance(List<Matrix<T>> predictions, Matrix<T> mean)
    {
        var numOps = NumericOperations<T>.Instance;

        int rows = predictions[0].Rows;
        int cols = predictions[0].Columns;

        var variance = new Matrix<T>(rows, cols);
        T count = numOps.FromDouble(_numSamples);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                T sumSquaredDiff = numOps.Zero;
                T meanVal = mean[r, c];

                foreach (var pred in predictions)
                {
                    T diff = numOps.Subtract(pred[r, c], meanVal);
                    T squaredDiff = numOps.Multiply(diff, diff);
                    sumSquaredDiff = numOps.Add(sumSquaredDiff, squaredDiff);
                }

                variance[r, c] = numOps.Divide(sumSquaredDiff, count);
            }
        }

        return variance;
    }

    private double ComputeAverageValue(Matrix<T> matrix)
    {
        var numOps = NumericOperations<T>.Instance;

        T sum = numOps.Zero;
        int count = matrix.Rows * matrix.Columns;

        for (int r = 0; r < matrix.Rows; r++)
        {
            for (int c = 0; c < matrix.Columns; c++)
            {
                sum = numOps.Add(sum, matrix[r, c]);
            }
        }

        T average = numOps.Divide(sum, numOps.FromDouble(count));
        return Convert.ToDouble(average);
    }
}

/// <summary>
/// Result of prediction with uncertainty quantification.
/// </summary>
public class UncertaintyPrediction<T>
{
    /// <summary>Mean prediction across all MC samples.</summary>
    public Matrix<T> Mean { get; set; } = new Matrix<T>(0, 0);

    /// <summary>Variance of predictions (higher = more uncertain).</summary>
    public Matrix<T> Variance { get; set; } = new Matrix<T>(0, 0);

    /// <summary>Average epistemic uncertainty (scalar value).</summary>
    public double EpistemicUncertainty { get; set; }

    /// <summary>All individual predictions from MC samples.</summary>
    public List<Matrix<T>> Predictions { get; set; } = new List<Matrix<T>>();
}
```

**Step 3**: Create unit test

```csharp
// File: tests/UnitTests/Uncertainty/MCDropoutPredictorTests.cs
namespace AiDotNet.Tests.Uncertainty;

public class MCDropoutPredictorTests
{
    [Fact]
    public void PredictWithUncertainty_ReturnsDifferentPredictions()
    {
        // Arrange - Create simple model with dropout
        var model = new Sequential<double>();
        model.Add(new DenseLayer<double>(10, 5));
        model.Add(new ActivationLayer<double>(new ReLU<double>()));
        model.Add(new DropoutLayer<double>(0.5)); // 50% dropout
        model.Add(new DenseLayer<double>(5, 2));

        var predictor = new MCDropoutPredictor<double>(model, numSamples: 10);

        var input = new Matrix<double>(1, 10);
        for (int i = 0; i < 10; i++)
            input[0, i] = i * 0.1;

        // Act
        var result = predictor.PredictWithUncertainty(input);

        // Assert
        Assert.NotNull(result.Mean);
        Assert.NotNull(result.Variance);
        Assert.Equal(10, result.Predictions.Count);
        Assert.True(result.EpistemicUncertainty >= 0); // Variance is non-negative

        // Predictions should vary due to dropout
        bool hasDifference = false;
        for (int i = 1; i < result.Predictions.Count; i++)
        {
            if (result.Predictions[i][0, 0] != result.Predictions[0][0, 0])
            {
                hasDifference = true;
                break;
            }
        }

        Assert.True(hasDifference, "Predictions should vary with dropout enabled");
    }
}
```

---

## Phase 2: Variational Inference for Bayesian NNs

### AC 2.1: Implement BayesianDenseLayer

**What is this?**
Instead of single weight values, we learn mean and variance for each weight. Sample from the distribution during forward pass.

**Mathematical Background**:
- Traditional: `W = 0.5` (deterministic)
- Bayesian: `W ~ Normal(μ=0.5, σ²=0.01)` (stochastic)
- Sample: `w_sample = μ + σ * ε` where `ε ~ Normal(0, 1)`

**File**: `src/Layers/BayesianDenseLayer.cs`

```csharp
// File: src/Layers/BayesianDenseLayer.cs
namespace AiDotNet.Layers;

/// <summary>
/// Bayesian fully-connected layer with weight uncertainty.
/// Learns distributions over weights instead of point estimates.
/// </summary>
public class BayesianDenseLayer<T> : LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly Random _random;

    // Weight distribution parameters
    private Matrix<T> _weightMean;        // Mean of weight distribution
    private Matrix<T> _weightLogVar;      // Log variance (for numerical stability)
    private Vector<T> _biasMean;          // Mean of bias distribution
    private Vector<T> _biasLogVar;        // Log variance of bias

    // Sampled weights for current forward pass
    private Matrix<T>? _sampledWeights;
    private Vector<T>? _sampledBias;

    private Matrix<T>? _lastInput;

    /// <summary>
    /// Number of samples for MC integration during training.
    /// </summary>
    public int NumSamples { get; set; } = 1;

    public BayesianDenseLayer(int inputSize, int outputSize)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _random = new Random();

        // Initialize weight distribution parameters
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        var numOps = NumericOperations<T>.Instance;

        _weightMean = new Matrix<T>(_outputSize, _inputSize);
        _weightLogVar = new Matrix<T>(_outputSize, _inputSize);
        _biasMean = new Vector<T>(_outputSize);
        _biasLogVar = new Vector<T>(_outputSize);

        // Xavier initialization for means
        double std = Math.Sqrt(2.0 / (_inputSize + _outputSize));

        for (int r = 0; r < _outputSize; r++)
        {
            for (int c = 0; c < _inputSize; c++)
            {
                // Initialize mean with small random values
                _weightMean[r, c] = numOps.FromDouble((_random.NextDouble() - 0.5) * 2 * std);

                // Initialize log variance to small negative value (small initial variance)
                _weightLogVar[r, c] = numOps.FromDouble(-5.0);
            }

            _biasMean[r] = numOps.Zero;
            _biasLogVar[r] = numOps.FromDouble(-5.0);
        }
    }

    public override Matrix<T> Forward(Matrix<T> input, bool training = false)
    {
        _lastInput = input;
        var numOps = NumericOperations<T>.Instance;

        // Sample weights from the learned distribution
        SampleWeights();

        // Standard forward pass with sampled weights
        // output = input * W^T + b
        var output = new Matrix<T>(input.Rows, _outputSize);

        for (int r = 0; r < input.Rows; r++)
        {
            for (int j = 0; j < _outputSize; j++)
            {
                T sum = _sampledBias![j];

                for (int k = 0; k < _inputSize; k++)
                {
                    sum = numOps.Add(sum,
                        numOps.Multiply(input[r, k], _sampledWeights![j, k]));
                }

                output[r, j] = sum;
            }
        }

        return output;
    }

    private void SampleWeights()
    {
        var numOps = NumericOperations<T>.Instance;

        _sampledWeights = new Matrix<T>(_outputSize, _inputSize);
        _sampledBias = new Vector<T>(_outputSize);

        // Reparameterization trick: w = μ + σ * ε, where ε ~ N(0,1)
        for (int r = 0; r < _outputSize; r++)
        {
            for (int c = 0; c < _inputSize; c++)
            {
                // Sample from standard normal
                double epsilon = SampleStandardNormal();

                // Compute std from log variance: σ = exp(0.5 * log(σ²))
                double logVar = Convert.ToDouble(_weightLogVar[r, c]);
                double std = Math.Exp(0.5 * logVar);

                // Reparameterization: w = μ + σ * ε
                double mean = Convert.ToDouble(_weightMean[r, c]);
                double sampled = mean + std * epsilon;

                _sampledWeights[r, c] = numOps.FromDouble(sampled);
            }

            // Sample bias
            double epsilonBias = SampleStandardNormal();
            double logVarBias = Convert.ToDouble(_biasLogVar[r]);
            double stdBias = Math.Exp(0.5 * logVarBias);
            double meanBias = Convert.ToDouble(_biasMean[r]);
            double sampledBias = meanBias + stdBias * epsilonBias;

            _sampledBias[r] = numOps.FromDouble(sampledBias);
        }
    }

    private double SampleStandardNormal()
    {
        // Box-Muller transform
        double u1 = _random.NextDouble();
        double u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    public override Matrix<T> Backward(Matrix<T> gradient)
    {
        if (_lastInput == null || _sampledWeights == null)
            throw new InvalidOperationException("Must call Forward before Backward");

        var numOps = NumericOperations<T>.Instance;

        // Compute gradient w.r.t. input: dL/dx = dL/dy * W
        var inputGradient = new Matrix<T>(_lastInput.Rows, _inputSize);

        for (int r = 0; r < _lastInput.Rows; r++)
        {
            for (int i = 0; i < _inputSize; i++)
            {
                T sum = numOps.Zero;

                for (int j = 0; j < _outputSize; j++)
                {
                    sum = numOps.Add(sum,
                        numOps.Multiply(gradient[r, j], _sampledWeights[j, i]));
                }

                inputGradient[r, i] = sum;
            }
        }

        // Compute gradients w.r.t. mean and log variance
        // This is complex - for now, store gradients for optimizer
        // Full implementation requires KL divergence gradient computation

        return inputGradient;
    }

    /// <summary>
    /// Computes KL divergence between learned distribution and prior.
    /// Used as regularization term in loss function.
    /// KL(q(w) || p(w)) where q is learned, p is prior N(0, 1)
    /// </summary>
    public T ComputeKLDivergence()
    {
        var numOps = NumericOperations<T>.Instance;
        T kl = numOps.Zero;

        // KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        for (int r = 0; r < _outputSize; r++)
        {
            for (int c = 0; c < _inputSize; c++)
            {
                double mu = Convert.ToDouble(_weightMean[r, c]);
                double logVar = Convert.ToDouble(_weightLogVar[r, c]);

                double klValue = 0.5 * (Math.Exp(logVar) + mu * mu - 1.0 - logVar);
                kl = numOps.Add(kl, numOps.FromDouble(klValue));
            }

            // Add bias KL
            double muBias = Convert.ToDouble(_biasMean[r]);
            double logVarBias = Convert.ToDouble(_biasLogVar[r]);
            double klBias = 0.5 * (Math.Exp(logVarBias) + muBias * muBias - 1.0 - logVarBias);
            kl = numOps.Add(kl, numOps.FromDouble(klBias));
        }

        return kl;
    }
}
```

**Step 2**: Create unit test

```csharp
// File: tests/UnitTests/Layers/BayesianDenseLayerTests.cs
namespace AiDotNet.Tests.Layers;

public class BayesianDenseLayerTests
{
    [Fact]
    public void Forward_ProducesDifferentOutputs_DueToSampling()
    {
        // Arrange
        var layer = new BayesianDenseLayer<double>(3, 2);
        var input = new Matrix<double>(1, 3);
        input[0, 0] = 1.0;
        input[0, 1] = 2.0;
        input[0, 2] = 3.0;

        // Act - Run forward pass multiple times
        var output1 = layer.Forward(input, training: true);
        var output2 = layer.Forward(input, training: true);
        var output3 = layer.Forward(input, training: true);

        // Assert - Outputs should differ due to weight sampling
        bool hasDifference = false;
        for (int c = 0; c < 2; c++)
        {
            if (output1[0, c] != output2[0, c] || output2[0, c] != output3[0, c])
            {
                hasDifference = true;
                break;
            }
        }

        Assert.True(hasDifference, "Bayesian layer should produce different outputs due to sampling");
    }

    [Fact]
    public void ComputeKLDivergence_ReturnsNonNegativeValue()
    {
        // Arrange
        var layer = new BayesianDenseLayer<double>(5, 3);

        // Act
        var kl = layer.ComputeKLDivergence();

        // Assert - KL divergence is always non-negative
        Assert.True(Convert.ToDouble(kl) >= 0.0);
    }
}
```

---

## Phase 3: Aleatoric Uncertainty Estimation

### AC 3.1: Implement AleatoricDenseLayer

**What is this?**
The network outputs BOTH a prediction AND its own estimate of data uncertainty.

**Architecture**:
```
Input → Hidden Layers → Split into two heads:
                        ├─ Mean prediction
                        └─ Log variance (uncertainty)
```

**File**: `src/Layers/AleatoricDenseLayer.cs`

```csharp
// File: src/Layers/AleatoricDenseLayer.cs
namespace AiDotNet.Layers;

/// <summary>
/// Dense layer that outputs both prediction and aleatoric uncertainty.
/// The network learns to estimate its own data uncertainty.
/// </summary>
public class AleatoricDenseLayer<T> : LayerBase<T>
{
    private readonly int _inputSize;
    private readonly int _outputSize;

    // Two separate weight matrices: one for mean, one for log variance
    private Matrix<T> _weightsMean;
    private Matrix<T> _weightsLogVar;
    private Vector<T> _biasMean;
    private Vector<T> _biasLogVar;

    private Matrix<T>? _lastInput;

    public AleatoricDenseLayer(int inputSize, int outputSize)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        InitializeParameters();
    }

    private void InitializeParameters()
    {
        var random = new Random();
        var numOps = NumericOperations<T>.Instance;

        double std = Math.Sqrt(2.0 / (_inputSize + _outputSize));

        _weightsMean = new Matrix<T>(_outputSize, _inputSize);
        _weightsLogVar = new Matrix<T>(_outputSize, _inputSize);
        _biasMean = new Vector<T>(_outputSize);
        _biasLogVar = new Vector<T>(_outputSize);

        for (int r = 0; r < _outputSize; r++)
        {
            for (int c = 0; c < _inputSize; c++)
            {
                _weightsMean[r, c] = numOps.FromDouble((random.NextDouble() - 0.5) * 2 * std);
                _weightsLogVar[r, c] = numOps.FromDouble((random.NextDouble() - 0.5) * 2 * std);
            }

            _biasMean[r] = numOps.Zero;
            _biasLogVar[r] = numOps.FromDouble(-1.0); // Initialize to low uncertainty
        }
    }

    public override Matrix<T> Forward(Matrix<T> input, bool training = false)
    {
        _lastInput = input;
        var numOps = NumericOperations<T>.Instance;

        // Compute mean prediction
        var mean = new Matrix<T>(input.Rows, _outputSize);
        var logVar = new Matrix<T>(input.Rows, _outputSize);

        for (int r = 0; r < input.Rows; r++)
        {
            // Mean prediction
            for (int j = 0; j < _outputSize; j++)
            {
                T sumMean = _biasMean[j];

                for (int k = 0; k < _inputSize; k++)
                {
                    sumMean = numOps.Add(sumMean,
                        numOps.Multiply(input[r, k], _weightsMean[j, k]));
                }

                mean[r, j] = sumMean;
            }

            // Log variance prediction
            for (int j = 0; j < _outputSize; j++)
            {
                T sumLogVar = _biasLogVar[j];

                for (int k = 0; k < _inputSize; k++)
                {
                    sumLogVar = numOps.Add(sumLogVar,
                        numOps.Multiply(input[r, k], _weightsLogVar[j, k]));
                }

                logVar[r, j] = sumLogVar;
            }
        }

        // Concatenate mean and log variance
        // Output format: [mean_0, ..., mean_n, logvar_0, ..., logvar_n]
        var output = new Matrix<T>(input.Rows, _outputSize * 2);

        for (int r = 0; r < input.Rows; r++)
        {
            for (int c = 0; c < _outputSize; c++)
            {
                output[r, c] = mean[r, c];                      // First half: means
                output[r, c + _outputSize] = logVar[r, c];      // Second half: log variances
            }
        }

        return output;
    }

    public override Matrix<T> Backward(Matrix<T> gradient)
    {
        // Standard backpropagation
        // Implementation similar to DenseLayer
        throw new NotImplementedException("Implement gradient computation");
    }
}
```

**Step 2**: Implement aleatoric loss function

```csharp
// File: src/Loss/AleatoricLoss.cs
namespace AiDotNet.Loss;

/// <summary>
/// Loss function for training networks with aleatoric uncertainty.
/// Minimizes: 0.5 * exp(-logvar) * (y - pred)² + 0.5 * logvar
/// This encourages the network to predict high variance when uncertain.
/// </summary>
public class AleatoricLoss<T> : ILoss<T>
{
    public T Compute(Matrix<T> predictions, Matrix<T> targets)
    {
        var numOps = NumericOperations<T>.Instance;

        int batchSize = predictions.Rows;
        int outputSize = predictions.Columns / 2; // Half mean, half logvar

        T totalLoss = numOps.Zero;

        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < outputSize; c++)
            {
                T pred = predictions[r, c];
                T logVar = predictions[r, c + outputSize];
                T target = targets[r, c];

                // Compute loss: 0.5 * exp(-logvar) * (y - pred)² + 0.5 * logvar
                T diff = numOps.Subtract(target, pred);
                T squaredDiff = numOps.Multiply(diff, diff);

                double logVarValue = Convert.ToDouble(logVar);
                double precision = Math.Exp(-logVarValue); // exp(-logvar)

                double lossValue = 0.5 * precision * Convert.ToDouble(squaredDiff) + 0.5 * logVarValue;

                totalLoss = numOps.Add(totalLoss, numOps.FromDouble(lossValue));
            }
        }

        // Average over batch and outputs
        T count = numOps.FromDouble(batchSize * outputSize);
        return numOps.Divide(totalLoss, count);
    }

    public Matrix<T> ComputeGradient(Matrix<T> predictions, Matrix<T> targets)
    {
        // Gradient computation for backpropagation
        throw new NotImplementedException("Implement gradient computation");
    }
}
```

---

## Phase 4: Integration and Visualization

### AC 4.1: Implement UncertaintyCalibration

**What is calibration?**
A well-calibrated model's predicted uncertainty matches actual error rates.
- If model says "80% confident" → should be correct 80% of the time
- Calibration plot: predicted confidence vs actual accuracy

**File**: `src/Uncertainty/UncertaintyCalibration.cs`

```csharp
// File: src/Uncertainty/UncertaintyCalibration.cs
namespace AiDotNet.Uncertainty;

/// <summary>
/// Evaluates uncertainty calibration and computes calibration metrics.
/// </summary>
public class UncertaintyCalibration<T>
{
    /// <summary>
    /// Compute Expected Calibration Error (ECE).
    /// Measures how well predicted confidence matches actual accuracy.
    /// </summary>
    public double ComputeECE(
        List<UncertaintyPrediction<T>> predictions,
        List<Vector<T>> targets,
        int numBins = 10)
    {
        if (predictions.Count != targets.Count)
            throw new ArgumentException("Predictions and targets must have same count");

        // Create bins for confidence levels
        var bins = new CalibrationBin[numBins];
        for (int i = 0; i < numBins; i++)
        {
            bins[i] = new CalibrationBin
            {
                MinConfidence = i / (double)numBins,
                MaxConfidence = (i + 1) / (double)numBins
            };
        }

        // Assign each prediction to a bin
        for (int i = 0; i < predictions.Count; i++)
        {
            var pred = predictions[i];
            var target = targets[i];

            // Get predicted class and confidence
            int predictedClass = GetPredictedClass(pred.Mean);
            double confidence = GetConfidence(pred.Mean);
            bool isCorrect = IsCorrect(pred.Mean, target);

            // Find appropriate bin
            int binIndex = Math.Min((int)(confidence * numBins), numBins - 1);

            bins[binIndex].Count++;
            bins[binIndex].TotalConfidence += confidence;
            bins[binIndex].TotalAccuracy += isCorrect ? 1.0 : 0.0;
        }

        // Compute ECE
        double ece = 0.0;
        int totalSamples = predictions.Count;

        foreach (var bin in bins)
        {
            if (bin.Count > 0)
            {
                double avgConfidence = bin.TotalConfidence / bin.Count;
                double avgAccuracy = bin.TotalAccuracy / bin.Count;
                double binWeight = bin.Count / (double)totalSamples;

                ece += binWeight * Math.Abs(avgConfidence - avgAccuracy);
            }
        }

        return ece;
    }

    private int GetPredictedClass(Matrix<T> prediction)
    {
        // Find index of maximum value in first row
        int maxIndex = 0;
        double maxValue = Convert.ToDouble(prediction[0, 0]);

        for (int c = 1; c < prediction.Columns; c++)
        {
            double value = Convert.ToDouble(prediction[0, c]);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = c;
            }
        }

        return maxIndex;
    }

    private double GetConfidence(Matrix<T> prediction)
    {
        // Use softmax max probability as confidence
        var softmax = ApplySoftmax(prediction);

        double maxProb = 0.0;
        for (int c = 0; c < softmax.Columns; c++)
        {
            double prob = Convert.ToDouble(softmax[0, c]);
            if (prob > maxProb)
                maxProb = prob;
        }

        return maxProb;
    }

    private bool IsCorrect(Matrix<T> prediction, Vector<T> target)
    {
        int predictedClass = GetPredictedClass(prediction);

        // Find true class (assuming one-hot encoding)
        int trueClass = 0;
        for (int i = 0; i < target.Length; i++)
        {
            if (Convert.ToDouble(target[i]) == 1.0)
            {
                trueClass = i;
                break;
            }
        }

        return predictedClass == trueClass;
    }

    private Matrix<T> ApplySoftmax(Matrix<T> input)
    {
        var numOps = NumericOperations<T>.Instance;
        var output = new Matrix<T>(input.Rows, input.Columns);

        for (int r = 0; r < input.Rows; r++)
        {
            // Find max for numerical stability
            double max = Convert.ToDouble(input[r, 0]);
            for (int c = 1; c < input.Columns; c++)
            {
                double val = Convert.ToDouble(input[r, c]);
                if (val > max)
                    max = val;
            }

            // Compute exp(x - max) and sum
            double sum = 0.0;
            for (int c = 0; c < input.Columns; c++)
            {
                double val = Convert.ToDouble(input[r, c]);
                sum += Math.Exp(val - max);
            }

            // Normalize
            for (int c = 0; c < input.Columns; c++)
            {
                double val = Convert.ToDouble(input[r, c]);
                double prob = Math.Exp(val - max) / sum;
                output[r, c] = numOps.FromDouble(prob);
            }
        }

        return output;
    }

    private class CalibrationBin
    {
        public double MinConfidence { get; set; }
        public double MaxConfidence { get; set; }
        public int Count { get; set; }
        public double TotalConfidence { get; set; }
        public double TotalAccuracy { get; set; }
    }
}
```

---

## Testing Strategy

### Unit Tests Required

1. **MC Dropout Tests**:
   - Test dropout enabled during inference
   - Verify predictions vary across samples
   - Check variance computation

2. **Bayesian Layer Tests**:
   - Verify weight sampling
   - Check KL divergence computation
   - Test different outputs per forward pass

3. **Aleatoric Tests**:
   - Verify dual output (mean + variance)
   - Test loss function gradient
   - Check uncertainty predictions

4. **Calibration Tests**:
   - Test ECE computation
   - Verify bin assignment
   - Check edge cases (perfect calibration, worst calibration)

---

## Common Pitfalls

1. **Forgetting to enable test-time dropout**:
   - Solution: Always set `EnableTestTimeDropout = true` before MC sampling

2. **Not enough MC samples**:
   - Too few: Unstable uncertainty estimates
   - Too many: Slow inference
   - Sweet spot: 30-100 samples

3. **Numerical instability with variance**:
   - Use log variance instead of variance
   - `σ = exp(0.5 * log_var)` is more stable than direct variance

4. **Miscalibrated uncertainty**:
   - Use temperature scaling post-training
   - Collect calibration dataset separate from test set

---

## Success Criteria Checklist

- [ ] MC Dropout predictor returns different predictions across samples
- [ ] Variance increases for out-of-distribution inputs
- [ ] Bayesian layer samples from weight distribution
- [ ] KL divergence loss properly regularizes Bayesian weights
- [ ] Aleatoric layer outputs both mean and uncertainty
- [ ] Aleatoric loss function encourages high variance when data is noisy
- [ ] ECE computation correctly measures calibration
- [ ] All unit tests pass with > 80% code coverage
- [ ] Documentation includes usage examples for all uncertainty methods

---

## Resources for Learning

1. **MC Dropout Paper**: "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
2. **Variational Inference**: "Weight Uncertainty in Neural Networks" (Blundell et al., 2015)
3. **Aleatoric vs Epistemic**: "What Uncertainties Do We Need in Bayesian Deep Learning?" (Kendall & Gal, 2017)
4. **Calibration**: "On Calibration of Modern Neural Networks" (Guo et al., 2017)

---

## Example Usage After Implementation

```csharp
// MC Dropout uncertainty estimation
var model = new Sequential<double>();
model.Add(new DenseLayer<double>(784, 256));
model.Add(new ActivationLayer<double>(new ReLU<double>()));
model.Add(new DropoutLayer<double>(0.3));
model.Add(new DenseLayer<double>(256, 10));

var mcPredictor = new MCDropoutPredictor<double>(model, numSamples: 50);
var result = mcPredictor.PredictWithUncertainty(testImage);

Console.WriteLine($"Prediction: {GetPredictedClass(result.Mean)}");
Console.WriteLine($"Uncertainty: {result.EpistemicUncertainty:F4}");

// If uncertainty > threshold, reject prediction or request human review
if (result.EpistemicUncertainty > 0.5)
{
    Console.WriteLine("High uncertainty - flagging for manual review");
}
```
