# Junior Developer Implementation Guide: Issue #409

## Overview
**Issue**: Quantization-Aware Training (QAT)
**Goal**: Train models with simulated quantization for efficient low-precision inference
**Difficulty**: Advanced
**Estimated Time**: 16-20 hours

## What is Quantization-Aware Training?

Quantization-Aware Training (QAT) simulates low-precision arithmetic during training:
- **Weights/Activations**: Stored as INT8/INT4 instead of FP32
- **Inference**: 4x-8x faster, 4x-8x smaller models
- **Accuracy**: Minimal loss compared to post-training quantization

### Why QAT vs. Post-Training Quantization?

**Post-Training Quantization (PTQ)**:
- Convert trained FP32 model to INT8 after training
- Fast but can lose 2-5% accuracy
- No model retraining needed

**Quantization-Aware Training (QAT)**:
- Simulate quantization during training
- Model learns to be robust to quantization noise
- Maintains near-FP32 accuracy with INT8 precision
- Requires retraining but yields better results

## Mathematical Background

### Uniform Quantization

```
Quantization maps continuous values to discrete levels:

q = round(x / scale) + zero_point

Dequantization (for gradient computation):
x_approx = scale * (q - zero_point)

Where:
    x = original FP32 value
    q = quantized integer value
    scale = (x_max - x_min) / (2^bits - 1)
    zero_point = offset for asymmetric quantization
```

### Symmetric vs. Asymmetric Quantization

**Symmetric (zero_point = 0)**:
```
scale = max(|x_max|, |x_min|) / (2^(bits-1) - 1)
q = round(x / scale)
Range: [-127, 127] for INT8
```

**Asymmetric (full range)**:
```
scale = (x_max - x_min) / (2^bits - 1)
zero_point = round(-x_min / scale)
q = round(x / scale) + zero_point
Range: [0, 255] for UINT8
```

### Straight-Through Estimator (STE)

Quantization is non-differentiable (round function has zero gradient everywhere).

**Solution**: Straight-Through Estimator approximates gradient:

```
Forward pass:  q = round(x / scale)
Backward pass: ∂q/∂x ≈ 1  (pretend round() is identity)

This allows gradients to flow through quantization nodes.
```

### Per-Tensor vs. Per-Channel Quantization

**Per-Tensor**: Single scale/zero-point for entire tensor
```
scale = (max(W) - min(W)) / 255
Faster but less accurate
```

**Per-Channel**: Different scale for each output channel
```
scale[i] = (max(W[i, :]) - min(W[i, :])) / 255
More accurate, commonly used for weights
```

## Understanding the Codebase

### Key Files to Create

**Core Interfaces:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IQuantizer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IQuantizationConfig.cs
```

**Implementations:**
```
C:\Users\cheat\source\repos\AiDotNet\src\Quantization\UniformQuantizer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Quantization\SymmetricQuantizer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Quantization\AsymmetricQuantizer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Quantization\FakeQuantizationLayer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Quantization\QuantizationAwareTrainer.cs
C:\Users\cheat\source\repos\AiDotNet\src\Quantization\QuantizationConfig.cs
```

**Test Files:**
```
C:\Users\cheat\source\repos\AiDotNet\tests\Quantization\QuantizationTests.cs
```

## Step-by-Step Implementation Guide

### Phase 1: Core Interfaces and Configuration

#### Step 1.1: Create IQuantizationConfig Interface

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IQuantizationConfig.cs
namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Configuration for quantization-aware training.
    /// </summary>
    public interface IQuantizationConfig
    {
        /// <summary>
        /// Number of bits for quantization (typically 8 for INT8).
        /// </summary>
        int Bits { get; set; }

        /// <summary>
        /// Quantization scheme: Symmetric or Asymmetric.
        /// </summary>
        QuantizationScheme Scheme { get; set; }

        /// <summary>
        /// Granularity: Per-tensor or per-channel.
        /// </summary>
        QuantizationGranularity Granularity { get; set; }

        /// <summary>
        /// Whether to quantize weights.
        /// </summary>
        bool QuantizeWeights { get; set; }

        /// <summary>
        /// Whether to quantize activations.
        /// </summary>
        bool QuantizeActivations { get; set; }

        /// <summary>
        /// Observer type for collecting statistics (min/max, percentile, etc.).
        /// </summary>
        ObserverType Observer { get; set; }

        /// <summary>
        /// Number of warmup batches before freezing quantization parameters.
        /// </summary>
        int WarmupBatches { get; set; }
    }

    public enum QuantizationScheme
    {
        /// <summary>Symmetric: zero-point at 0, range [-127, 127]</summary>
        Symmetric,
        /// <summary>Asymmetric: arbitrary zero-point, range [0, 255]</summary>
        Asymmetric
    }

    public enum QuantizationGranularity
    {
        /// <summary>Single scale/zero-point for entire tensor</summary>
        PerTensor,
        /// <summary>Different scale/zero-point per output channel</summary>
        PerChannel
    }

    public enum ObserverType
    {
        /// <summary>Use min/max of observed values</summary>
        MinMax,
        /// <summary>Use moving average of min/max</summary>
        MovingAverageMinMax,
        /// <summary>Use percentile (e.g., 0.01 and 99.99 percentile)</summary>
        Percentile
    }
}
```

#### Step 1.2: Create IQuantizer Interface

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IQuantizer.cs
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines quantization and dequantization operations.
    /// </summary>
    /// <typeparam name="T">Numeric type (usually double or float)</typeparam>
    public interface IQuantizer<T>
    {
        /// <summary>
        /// Quantizes a value to integer representation.
        /// </summary>
        /// <param name="value">Floating-point value to quantize</param>
        /// <returns>Quantized integer value</returns>
        int Quantize(T value);

        /// <summary>
        /// Dequantizes an integer back to floating-point.
        /// </summary>
        /// <param name="quantizedValue">Integer value</param>
        /// <returns>Dequantized floating-point value</returns>
        T Dequantize(int quantizedValue);

        /// <summary>
        /// Fake quantization: quantize then immediately dequantize.
        /// Used during training to simulate quantization effects.
        /// </summary>
        /// <param name="value">Original value</param>
        /// <returns>Value after quantize-dequantize cycle</returns>
        T FakeQuantize(T value);

        /// <summary>
        /// Fake quantize an entire matrix (element-wise).
        /// </summary>
        Matrix<T> FakeQuantize(Matrix<T> matrix);

        /// <summary>
        /// Fake quantize a tensor (element-wise).
        /// </summary>
        Tensor<T> FakeQuantize(Tensor<T> tensor);

        /// <summary>
        /// Updates quantization parameters (scale, zero-point) based on observed data.
        /// </summary>
        /// <param name="data">Data to observe for calibration</param>
        void Calibrate(Matrix<T> data);

        /// <summary>
        /// Gets the quantization scale factor.
        /// </summary>
        T Scale { get; }

        /// <summary>
        /// Gets the zero-point offset.
        /// </summary>
        int ZeroPoint { get; }

        /// <summary>
        /// Number of quantization bits.
        /// </summary>
        int Bits { get; }
    }
}
```

### Phase 2: Quantizer Implementations

#### Step 2.1: Implement SymmetricQuantizer

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Quantization\SymmetricQuantizer.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Quantization
{
    /// <summary>
    /// Symmetric quantization: zero-point at 0, range [-2^(n-1), 2^(n-1)-1].
    /// Commonly used for weights.
    /// </summary>
    public class SymmetricQuantizer<T> : IQuantizer<T>
    {
        private readonly INumericOperations<T> _numOps;
        private T _scale;
        private readonly int _qmin;
        private readonly int _qmax;

        public T Scale => _scale;
        public int ZeroPoint => 0; // Always 0 for symmetric
        public int Bits { get; }

        public SymmetricQuantizer(int bits = 8)
        {
            _numOps = NumericOperations<T>.Instance;
            Bits = bits;

            // For INT8: range is [-128, 127]
            _qmin = -(1 << (bits - 1));
            _qmax = (1 << (bits - 1)) - 1;

            // Initialize scale to 1.0 (will be calibrated)
            _scale = _numOps.One;
        }

        public void Calibrate(Matrix<T> data)
        {
            // Find maximum absolute value
            T maxAbs = _numOps.Zero;

            for (int i = 0; i < data.Rows; i++)
            {
                for (int j = 0; j < data.Columns; j++)
                {
                    var absVal = _numOps.Abs(data[i, j]);
                    if (_numOps.GreaterThan(absVal, maxAbs))
                        maxAbs = absVal;
                }
            }

            // Avoid division by zero
            if (_numOps.Equals(maxAbs, _numOps.Zero))
            {
                _scale = _numOps.One;
                return;
            }

            // scale = max_abs / q_max
            double maxAbsDouble = Convert.ToDouble(_numOps.ToDouble(maxAbs));
            double scaleDouble = maxAbsDouble / _qmax;

            _scale = _numOps.FromDouble(scaleDouble);
        }

        public int Quantize(T value)
        {
            // q = round(x / scale)
            double valueDouble = Convert.ToDouble(_numOps.ToDouble(value));
            double scaleDouble = Convert.ToDouble(_numOps.ToDouble(_scale));

            int quantized = (int)Math.Round(valueDouble / scaleDouble);

            // Clamp to valid range
            return Math.Max(_qmin, Math.Min(_qmax, quantized));
        }

        public T Dequantize(int quantizedValue)
        {
            // x_approx = scale * q
            double dequantized = quantizedValue * Convert.ToDouble(_numOps.ToDouble(_scale));
            return _numOps.FromDouble(dequantized);
        }

        public T FakeQuantize(T value)
        {
            // Quantize then dequantize (simulates quantization noise)
            int q = Quantize(value);
            return Dequantize(q);
        }

        public Matrix<T> FakeQuantize(Matrix<T> matrix)
        {
            var result = new Matrix<T>(matrix.Rows, matrix.Columns);

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = FakeQuantize(matrix[i, j]);
                }
            }

            return result;
        }

        public Tensor<T> FakeQuantize(Tensor<T> tensor)
        {
            var result = tensor.Clone();

            // Flatten and quantize each element
            var flatSize = 1;
            for (int i = 0; i < tensor.Rank; i++)
                flatSize *= tensor.Dimensions[i];

            for (int i = 0; i < flatSize; i++)
            {
                var indices = FlatIndexToMultiDim(i, tensor.Dimensions);
                var value = tensor[indices];
                var fakeQ = FakeQuantize(value);
                result[indices] = fakeQ;
            }

            return result;
        }

        private int[] FlatIndexToMultiDim(int flatIndex, int[] dimensions)
        {
            var indices = new int[dimensions.Length];
            int remaining = flatIndex;

            for (int dim = dimensions.Length - 1; dim >= 0; dim--)
            {
                indices[dim] = remaining % dimensions[dim];
                remaining /= dimensions[dim];
            }

            return indices;
        }
    }
}
```

#### Step 2.2: Implement AsymmetricQuantizer

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Quantization\AsymmetricQuantizer.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Quantization
{
    /// <summary>
    /// Asymmetric quantization: arbitrary zero-point, range [0, 2^n - 1].
    /// Commonly used for activations (often non-negative like ReLU).
    /// </summary>
    public class AsymmetricQuantizer<T> : IQuantizer<T>
    {
        private readonly INumericOperations<T> _numOps;
        private T _scale;
        private int _zeroPoint;
        private readonly int _qmin;
        private readonly int _qmax;

        public T Scale => _scale;
        public int ZeroPoint => _zeroPoint;
        public int Bits { get; }

        public AsymmetricQuantizer(int bits = 8)
        {
            _numOps = NumericOperations<T>.Instance;
            Bits = bits;

            // For UINT8: range is [0, 255]
            _qmin = 0;
            _qmax = (1 << bits) - 1;

            _scale = _numOps.One;
            _zeroPoint = 0;
        }

        public void Calibrate(Matrix<T> data)
        {
            // Find min and max values
            T minVal = data[0, 0];
            T maxVal = data[0, 0];

            for (int i = 0; i < data.Rows; i++)
            {
                for (int j = 0; j < data.Columns; j++)
                {
                    var val = data[i, j];
                    if (_numOps.LessThan(val, minVal))
                        minVal = val;
                    if (_numOps.GreaterThan(val, maxVal))
                        maxVal = val;
                }
            }

            double minDouble = Convert.ToDouble(_numOps.ToDouble(minVal));
            double maxDouble = Convert.ToDouble(_numOps.ToDouble(maxVal));

            // Avoid division by zero
            if (Math.Abs(maxDouble - minDouble) < 1e-10)
            {
                _scale = _numOps.One;
                _zeroPoint = 0;
                return;
            }

            // scale = (max - min) / (q_max - q_min)
            double scaleDouble = (maxDouble - minDouble) / (_qmax - _qmin);
            _scale = _numOps.FromDouble(scaleDouble);

            // zero_point = round(q_min - min / scale)
            _zeroPoint = (int)Math.Round(_qmin - minDouble / scaleDouble);

            // Clamp zero-point to valid range
            _zeroPoint = Math.Max(_qmin, Math.Min(_qmax, _zeroPoint));
        }

        public int Quantize(T value)
        {
            // q = round(x / scale) + zero_point
            double valueDouble = Convert.ToDouble(_numOps.ToDouble(value));
            double scaleDouble = Convert.ToDouble(_numOps.ToDouble(_scale));

            int quantized = (int)Math.Round(valueDouble / scaleDouble) + _zeroPoint;

            return Math.Max(_qmin, Math.Min(_qmax, quantized));
        }

        public T Dequantize(int quantizedValue)
        {
            // x_approx = scale * (q - zero_point)
            double scaleDouble = Convert.ToDouble(_numOps.ToDouble(_scale));
            double dequantized = scaleDouble * (quantizedValue - _zeroPoint);

            return _numOps.FromDouble(dequantized);
        }

        public T FakeQuantize(T value)
        {
            int q = Quantize(value);
            return Dequantize(q);
        }

        public Matrix<T> FakeQuantize(Matrix<T> matrix)
        {
            var result = new Matrix<T>(matrix.Rows, matrix.Columns);

            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Columns; j++)
                    result[i, j] = FakeQuantize(matrix[i, j]);

            return result;
        }

        public Tensor<T> FakeQuantize(Tensor<T> tensor)
        {
            var result = tensor.Clone();

            var flatSize = 1;
            for (int i = 0; i < tensor.Rank; i++)
                flatSize *= tensor.Dimensions[i];

            for (int i = 0; i < flatSize; i++)
            {
                var indices = FlatIndexToMultiDim(i, tensor.Dimensions);
                result[indices] = FakeQuantize(tensor[indices]);
            }

            return result;
        }

        private int[] FlatIndexToMultiDim(int flatIndex, int[] dimensions)
        {
            var indices = new int[dimensions.Length];
            int remaining = flatIndex;

            for (int dim = dimensions.Length - 1; dim >= 0; dim--)
            {
                indices[dim] = remaining % dimensions[dim];
                remaining /= dimensions[dim];
            }

            return indices;
        }
    }
}
```

### Phase 3: Fake Quantization Layer

#### Step 3.1: Implement FakeQuantizationLayer

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Quantization\FakeQuantizationLayer.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;

namespace AiDotNet.Quantization
{
    /// <summary>
    /// Fake quantization layer inserted into neural network during QAT.
    /// Quantizes then dequantizes to simulate quantization noise.
    /// Uses Straight-Through Estimator for gradients.
    /// </summary>
    public class FakeQuantizationLayer<T>
    {
        private readonly IQuantizer<T> _quantizer;
        private readonly INumericOperations<T> _numOps;
        private readonly bool _enabled;
        private int _batchesSeen;
        private readonly int _warmupBatches;

        public IQuantizer<T> Quantizer => _quantizer;

        /// <summary>
        /// Creates a fake quantization layer.
        /// </summary>
        /// <param name="quantizer">Quantizer to use</param>
        /// <param name="warmupBatches">Number of batches to observe before freezing params</param>
        /// <param name="enabled">Whether quantization is enabled (can disable during eval)</param>
        public FakeQuantizationLayer(
            IQuantizer<T> quantizer,
            int warmupBatches = 100,
            bool enabled = true)
        {
            _quantizer = quantizer ?? throw new ArgumentNullException(nameof(quantizer));
            _numOps = NumericOperations<T>.Instance;
            _warmupBatches = warmupBatches;
            _enabled = enabled;
            _batchesSeen = 0;
        }

        /// <summary>
        /// Forward pass: apply fake quantization.
        /// </summary>
        public Matrix<T> Forward(Matrix<T> input, bool isTraining = true)
        {
            if (!_enabled)
                return input;

            // During warmup, update quantization parameters
            if (isTraining && _batchesSeen < _warmupBatches)
            {
                _quantizer.Calibrate(input);
                _batchesSeen++;
            }

            // Apply fake quantization
            return _quantizer.FakeQuantize(input);
        }

        /// <summary>
        /// Forward pass for tensors (e.g., conv layer activations).
        /// </summary>
        public Tensor<T> Forward(Tensor<T> input, bool isTraining = true)
        {
            if (!_enabled)
                return input;

            if (isTraining && _batchesSeen < _warmupBatches)
            {
                // Calibrate using flattened tensor data
                var flatMatrix = TensorToMatrix(input);
                _quantizer.Calibrate(flatMatrix);
                _batchesSeen++;
            }

            return _quantizer.FakeQuantize(input);
        }

        /// <summary>
        /// Backward pass: Straight-Through Estimator.
        /// Gradient flows through as if quantization was identity function.
        /// </summary>
        public Matrix<T> Backward(Matrix<T> gradOutput)
        {
            // STE: ∂output/∂input ≈ 1
            // Gradient passes through unchanged
            return gradOutput.Clone();
        }

        public Tensor<T> Backward(Tensor<T> gradOutput)
        {
            // STE for tensors
            return gradOutput.Clone();
        }

        /// <summary>
        /// Resets calibration (for restarting warmup).
        /// </summary>
        public void ResetCalibration()
        {
            _batchesSeen = 0;
        }

        /// <summary>
        /// Gets current quantization parameters (for debugging).
        /// </summary>
        public (T scale, int zeroPoint) GetQuantizationParams()
        {
            return (_quantizer.Scale, _quantizer.ZeroPoint);
        }

        private Matrix<T> TensorToMatrix(Tensor<T> tensor)
        {
            // Flatten tensor to 2D for calibration
            var flatSize = 1;
            for (int i = 0; i < tensor.Rank; i++)
                flatSize *= tensor.Dimensions[i];

            var matrix = new Matrix<T>(flatSize, 1);

            for (int i = 0; i < flatSize; i++)
            {
                var indices = FlatIndexToMultiDim(i, tensor.Dimensions);
                matrix[i, 0] = tensor[indices];
            }

            return matrix;
        }

        private int[] FlatIndexToMultiDim(int flatIndex, int[] dimensions)
        {
            var indices = new int[dimensions.Length];
            int remaining = flatIndex;

            for (int dim = dimensions.Length - 1; dim >= 0; dim--)
            {
                indices[dim] = remaining % dimensions[dim];
                remaining /= dimensions[dim];
            }

            return indices;
        }
    }
}
```

### Phase 4: Quantization-Aware Trainer

#### Step 4.1: Implement QuantizationAwareTrainer

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Quantization\QuantizationAwareTrainer.cs
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Quantization
{
    /// <summary>
    /// Trainer for quantization-aware training.
    /// Inserts fake quantization layers into network and trains with simulated quantization.
    /// </summary>
    public class QuantizationAwareTrainer<TInput, TOutput, T>
    {
        private readonly IQuantizationConfig _config;
        private readonly Dictionary<string, FakeQuantizationLayer<T>> _quantLayers;

        public QuantizationAwareTrainer(IQuantizationConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _quantLayers = new Dictionary<string, FakeQuantizationLayer<T>>();
        }

        /// <summary>
        /// Prepares a model for QAT by inserting fake quantization layers.
        /// </summary>
        public void PrepareModel(object model)
        {
            // Insert fake quantization layers after each layer's weights and activations
            if (model is NeuralNetwork<TInput, TOutput, T> neuralNet)
            {
                foreach (var layer in neuralNet.Layers)
                {
                    // Create quantizer for weights
                    if (_config.QuantizeWeights)
                    {
                        var weightQuantizer = CreateQuantizer();
                        var weightQuantLayer = new FakeQuantizationLayer<T>(
                            weightQuantizer,
                            warmupBatches: _config.WarmupBatches
                        );

                        _quantLayers[$"{layer.Name}_weights"] = weightQuantLayer;
                    }

                    // Create quantizer for activations
                    if (_config.QuantizeActivations)
                    {
                        var activationQuantizer = CreateQuantizer();
                        var activationQuantLayer = new FakeQuantizationLayer<T>(
                            activationQuantizer,
                            warmupBatches: _config.WarmupBatches
                        );

                        _quantLayers[$"{layer.Name}_activations"] = activationQuantLayer;
                    }
                }
            }
        }

        /// <summary>
        /// Trains the model with quantization-aware training.
        /// </summary>
        public void Train(
            object model,
            TInput[] trainData,
            TOutput[] trainLabels,
            int epochs,
            int batchSize = 32,
            double learningRate = 0.001)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double epochLoss = 0;
                int numBatches = (trainData.Length + batchSize - 1) / batchSize;

                for (int b = 0; b < numBatches; b++)
                {
                    int start = b * batchSize;
                    int end = Math.Min(start + batchSize, trainData.Length);

                    var batchInputs = trainData[start..end];
                    var batchLabels = trainLabels[start..end];

                    // Forward pass with fake quantization
                    double batchLoss = TrainBatch(model, batchInputs, batchLabels, learningRate);
                    epochLoss += batchLoss;
                }

                Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / numBatches:F4}");

                // After warmup, print quantization stats
                if (epoch == _config.WarmupBatches / (trainData.Length / batchSize))
                {
                    PrintQuantizationStats();
                }
            }
        }

        /// <summary>
        /// Converts QAT model to actual quantized model (for deployment).
        /// </summary>
        public object ConvertToQuantizedModel(object model)
        {
            // Replace fake quantization layers with actual INT8 operations
            // This would integrate with ONNX export (Issue #410)

            Console.WriteLine("Converting model to fully quantized INT8...");

            foreach (var kvp in _quantLayers)
            {
                var (scale, zeroPoint) = kvp.Value.GetQuantizationParams();
                Console.WriteLine($"{kvp.Key}: scale={scale}, zero_point={zeroPoint}");
            }

            // Return quantized model (implementation depends on inference runtime)
            return model;
        }

        private double TrainBatch(object model, TInput[] inputs, TOutput[] labels, double learningRate)
        {
            double totalLoss = 0;

            foreach (var (input, label) in inputs.Zip(labels))
            {
                // Forward pass through model with fake quantization
                var output = ForwardWithQuantization(model, input, isTraining: true);

                // Compute loss
                var loss = ComputeLoss(output, label);
                totalLoss += loss;

                // Backward pass (gradients flow through STE)
                BackwardWithQuantization(model, output, label, learningRate);
            }

            return totalLoss / inputs.Length;
        }

        private TOutput ForwardWithQuantization(object model, TInput input, bool isTraining)
        {
            // This is model-specific implementation
            // Pseudo-code:
            // 1. For each layer:
            //    a. Apply fake quantization to weights
            //    b. Compute layer output
            //    c. Apply fake quantization to activations
            // 2. Return final output

            throw new NotImplementedException("Model-specific forward pass");
        }

        private void BackwardWithQuantization(object model, TOutput output, TOutput label, double learningRate)
        {
            // Backward pass with STE
            // Gradients pass through fake quantization layers unchanged
            throw new NotImplementedException("Model-specific backward pass");
        }

        private double ComputeLoss(TOutput predicted, TOutput actual)
        {
            // Loss computation (model-specific)
            throw new NotImplementedException("Model-specific loss");
        }

        private IQuantizer<T> CreateQuantizer()
        {
            return _config.Scheme switch
            {
                QuantizationScheme.Symmetric => new SymmetricQuantizer<T>(_config.Bits),
                QuantizationScheme.Asymmetric => new AsymmetricQuantizer<T>(_config.Bits),
                _ => throw new NotSupportedException($"Quantization scheme {_config.Scheme} not supported")
            };
        }

        private void PrintQuantizationStats()
        {
            Console.WriteLine("\n=== Quantization Parameters (after warmup) ===");

            foreach (var kvp in _quantLayers)
            {
                var (scale, zeroPoint) = kvp.Value.GetQuantizationParams();
                Console.WriteLine($"{kvp.Key}:");
                Console.WriteLine($"  Scale: {scale}");
                Console.WriteLine($"  Zero-point: {zeroPoint}");
            }

            Console.WriteLine("===============================================\n");
        }
    }
}
```

### Phase 5: Quantization Configuration

#### Step 5.1: Implement QuantizationConfig

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Quantization\QuantizationConfig.cs
using AiDotNet.Interfaces;

namespace AiDotNet.Quantization
{
    /// <summary>
    /// Configuration for quantization-aware training.
    /// </summary>
    public class QuantizationConfig : IQuantizationConfig
    {
        public int Bits { get; set; } = 8;
        public QuantizationScheme Scheme { get; set; } = QuantizationScheme.Symmetric;
        public QuantizationGranularity Granularity { get; set; } = QuantizationGranularity.PerTensor;
        public bool QuantizeWeights { get; set; } = true;
        public bool QuantizeActivations { get; set; } = true;
        public ObserverType Observer { get; set; } = ObserverType.MinMax;
        public int WarmupBatches { get; set; } = 100;

        /// <summary>
        /// Creates default INT8 quantization config.
        /// </summary>
        public static QuantizationConfig Default()
        {
            return new QuantizationConfig
            {
                Bits = 8,
                Scheme = QuantizationScheme.Symmetric,
                Granularity = QuantizationGranularity.PerTensor,
                QuantizeWeights = true,
                QuantizeActivations = true,
                Observer = ObserverType.MovingAverageMinMax,
                WarmupBatches = 100
            };
        }

        /// <summary>
        /// Creates config optimized for mobile deployment.
        /// </summary>
        public static QuantizationConfig MobileOptimized()
        {
            return new QuantizationConfig
            {
                Bits = 8,
                Scheme = QuantizationScheme.Asymmetric, // Better for ReLU activations
                Granularity = QuantizationGranularity.PerChannel, // More accurate
                QuantizeWeights = true,
                QuantizeActivations = true,
                Observer = ObserverType.MovingAverageMinMax,
                WarmupBatches = 200 // More calibration for stability
            };
        }

        /// <summary>
        /// Creates aggressive 4-bit quantization config.
        /// </summary>
        public static QuantizationConfig INT4Aggressive()
        {
            return new QuantizationConfig
            {
                Bits = 4,
                Scheme = QuantizationScheme.Symmetric,
                Granularity = QuantizationGranularity.PerChannel,
                QuantizeWeights = true,
                QuantizeActivations = false, // Keep activations FP32 for accuracy
                Observer = ObserverType.Percentile, // Use percentile to avoid outliers
                WarmupBatches = 500 // More warmup for 4-bit
            };
        }
    }
}
```

## Testing Strategy

### Phase 6: Comprehensive Tests

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\tests\Quantization\QuantizationTests.cs
using Xunit;
using AiDotNet.Quantization;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.Quantization
{
    public class QuantizationTests
    {
        [Fact]
        public void SymmetricQuantizer_INT8_QuantizesCorrectly()
        {
            // Arrange
            var quantizer = new SymmetricQuantizer<double>(bits: 8);

            var weights = new Matrix<double>(2, 2);
            weights[0, 0] = 1.0;
            weights[0, 1] = -1.0;
            weights[1, 0] = 0.5;
            weights[1, 1] = -0.5;

            // Act
            quantizer.Calibrate(weights);

            // Assert
            // max_abs = 1.0, q_max = 127
            // scale = 1.0 / 127 ≈ 0.00787
            Assert.True(Math.Abs(Convert.ToDouble(quantizer.Scale) - 1.0 / 127.0) < 1e-4);
            Assert.Equal(0, quantizer.ZeroPoint); // Symmetric → zero-point = 0

            // Test quantization
            int q1 = quantizer.Quantize(1.0);
            int q2 = quantizer.Quantize(-1.0);

            Assert.Equal(127, q1);   // 1.0 → 127
            Assert.Equal(-127, q2);  // -1.0 → -127
        }

        [Fact]
        public void AsymmetricQuantizer_UINT8_UsesFullRange()
        {
            // Arrange
            var quantizer = new AsymmetricQuantizer<double>(bits: 8);

            var activations = new Matrix<double>(2, 2);
            activations[0, 0] = 0.0;   // ReLU output
            activations[0, 1] = 2.0;
            activations[1, 0] = 1.0;
            activations[1, 1] = 3.0;

            // Act
            quantizer.Calibrate(activations);

            // Assert
            // min = 0.0, max = 3.0, range = [0, 255]
            // scale = 3.0 / 255 ≈ 0.0118
            double expectedScale = 3.0 / 255.0;
            Assert.True(Math.Abs(Convert.ToDouble(quantizer.Scale) - expectedScale) < 1e-3);

            // zero_point should be 0 (since min is 0)
            Assert.Equal(0, quantizer.ZeroPoint);

            // Test quantization
            int q0 = quantizer.Quantize(0.0);
            int q3 = quantizer.Quantize(3.0);

            Assert.Equal(0, q0);     // Min → 0
            Assert.Equal(255, q3);   // Max → 255
        }

        [Fact]
        public void FakeQuantization_IntroducesQuantizationNoise()
        {
            // Arrange
            var quantizer = new SymmetricQuantizer<double>(bits: 8);

            var weights = new Matrix<double>(1, 1);
            weights[0, 0] = 0.123456; // Precise FP32 value

            quantizer.Calibrate(weights);

            // Act
            double original = weights[0, 0];
            double fakeQuantized = quantizer.FakeQuantize(original);

            // Assert
            // Should be slightly different due to quantization
            Assert.NotEqual(original, fakeQuantized);

            // But close (within scale resolution)
            double scale = Convert.ToDouble(quantizer.Scale);
            Assert.True(Math.Abs(original - fakeQuantized) <= scale);
        }

        [Fact]
        public void FakeQuantizationLayer_StraightThroughEstimator_PassesGradient()
        {
            // Arrange
            var quantizer = new SymmetricQuantizer<double>(bits: 8);
            var fakeQuantLayer = new FakeQuantizationLayer<double>(quantizer, warmupBatches: 10);

            var input = new Matrix<double>(2, 2);
            input[0, 0] = 1.0;
            input[0, 1] = 2.0;
            input[1, 0] = 3.0;
            input[1, 1] = 4.0;

            var gradOutput = new Matrix<double>(2, 2);
            gradOutput[0, 0] = 0.1;
            gradOutput[0, 1] = 0.2;
            gradOutput[1, 0] = 0.3;
            gradOutput[1, 1] = 0.4;

            // Act
            var output = fakeQuantLayer.Forward(input, isTraining: true);
            var gradInput = fakeQuantLayer.Backward(gradOutput);

            // Assert
            // STE: gradient passes through unchanged
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Assert.Equal(gradOutput[i, j], gradInput[i, j]);
                }
            }
        }

        [Fact]
        public void Quantization_8Bit_ReducesModelSize()
        {
            // Arrange
            var weights = new Matrix<double>(100, 100); // 10,000 weights

            // Fill with random values
            var random = new Random(42);
            for (int i = 0; i < 100; i++)
                for (int j = 0; j < 100; j++)
                    weights[i, j] = random.NextDouble() * 2.0 - 1.0;

            var quantizer = new SymmetricQuantizer<double>(bits: 8);
            quantizer.Calibrate(weights);

            // Act - Quantize all weights
            var quantizedWeights = new int[100 * 100];
            int idx = 0;
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    quantizedWeights[idx++] = quantizer.Quantize(weights[i, j]);
                }
            }

            // Assert
            // FP32: 10,000 weights × 4 bytes = 40KB
            // INT8: 10,000 weights × 1 byte + 1 scale (4 bytes) = 10KB + 4B ≈ 10KB
            // Size reduction: 4x

            int fp32Size = 100 * 100 * 4; // bytes
            int int8Size = 100 * 100 * 1 + 4; // bytes (weights + scale)

            double sizeReduction = (double)fp32Size / int8Size;
            Assert.True(sizeReduction >= 3.9 && sizeReduction <= 4.1); // ~4x reduction
        }

        [Fact]
        public void PerChannelQuantization_MoreAccurateThanPerTensor()
        {
            // Arrange - Weights with different scales per channel
            var weights = new Matrix<double>(3, 2);
            weights[0, 0] = 0.1;  weights[0, 1] = 10.0;  // Channel 1: small, Channel 2: large
            weights[1, 0] = 0.2;  weights[1, 1] = 20.0;
            weights[2, 0] = 0.3;  weights[2, 1] = 30.0;

            // Per-tensor quantization
            var perTensorQuant = new SymmetricQuantizer<double>(bits: 8);
            perTensorQuant.Calibrate(weights);

            // Per-channel quantization (simulate)
            var channel0 = new Matrix<double>(3, 1);
            var channel1 = new Matrix<double>(3, 1);
            for (int i = 0; i < 3; i++)
            {
                channel0[i, 0] = weights[i, 0];
                channel1[i, 0] = weights[i, 1];
            }

            var perChannelQuant0 = new SymmetricQuantizer<double>(bits: 8);
            var perChannelQuant1 = new SymmetricQuantizer<double>(bits: 8);

            perChannelQuant0.Calibrate(channel0);
            perChannelQuant1.Calibrate(channel1);

            // Act - Quantize small value (0.1)
            var perTensorResult = perTensorQuant.FakeQuantize(0.1);
            var perChannelResult = perChannelQuant0.FakeQuantize(0.1);

            // Assert
            // Per-channel should be more accurate for small values
            double perTensorError = Math.Abs(0.1 - perTensorResult);
            double perChannelError = Math.Abs(0.1 - perChannelResult);

            Assert.True(perChannelError < perTensorError);
        }
    }
}
```

## Usage Example: Complete QAT Workflow

```csharp
// Example: Quantization-Aware Training for MobileNet

// 1. Load pre-trained FP32 model
var model = NeuralNetwork.LoadFromFile("mobilenet_fp32.bin");

// 2. Configure quantization
var qatConfig = QuantizationConfig.MobileOptimized();

// 3. Create QAT trainer
var qatTrainer = new QuantizationAwareTrainer<Tensor<float>, Vector<float>, float>(qatConfig);

// 4. Prepare model (insert fake quantization layers)
qatTrainer.PrepareModel(model);

// 5. Train with QAT (fine-tune for quantization robustness)
qatTrainer.Train(
    model,
    trainImages,
    trainLabels,
    epochs: 10,     // Usually fewer epochs than from-scratch training
    batchSize: 64,
    learningRate: 0.0001  // Lower LR for fine-tuning
);

// 6. Evaluate accuracy
double fp32Accuracy = EvaluateFP32(model, testImages, testLabels);
double qatAccuracy = EvaluateWithQuantization(model, testImages, testLabels);

Console.WriteLine($"FP32 accuracy: {fp32Accuracy:P2}");
Console.WriteLine($"QAT accuracy: {qatAccuracy:P2}");
Console.WriteLine($"Accuracy drop: {(fp32Accuracy - qatAccuracy):P2}");

// 7. Convert to fully quantized INT8 model
var quantizedModel = qatTrainer.ConvertToQuantizedModel(model);

// 8. Export to ONNX for deployment (Issue #410)
OnnxExporter.Export(quantizedModel, "mobilenet_int8.onnx");

// 9. Verify final model size
var fp32Size = GetModelSize(model);
var int8Size = GetModelSize(quantizedModel);

Console.WriteLine($"Model size reduction: {fp32Size / int8Size:F1}x");
```

## Common Pitfalls to Avoid

1. **Not using warmup** - Quantization params stabilize over first 100-200 batches
2. **Quantizing first/last layers** - Often kept FP32 for accuracy
3. **Wrong quantization scheme** - Use symmetric for weights, asymmetric for ReLU activations
4. **Forgetting STE** - Without it, gradients can't flow through quantization
5. **Too aggressive quantization** - INT4 requires careful tuning
6. **Not calibrating properly** - Need representative data for min/max calculation

## Advanced Topics

### Mixed Precision Quantization

```csharp
// Keep first/last layers FP32, quantize middle layers to INT8
var config = new QuantizationConfig
{
    // Custom per-layer configuration
    LayerConfigs = new Dictionary<string, LayerQuantConfig>
    {
        ["layer_0"] = new LayerQuantConfig { QuantizeWeights = false }, // FP32
        ["layer_1_to_n"] = new LayerQuantConfig { Bits = 8 },           // INT8
        ["layer_last"] = new LayerQuantConfig { QuantizeWeights = false } // FP32
    }
};
```

### Dynamic vs. Static Quantization

**Static (done here)**: Quantization params fixed after calibration
**Dynamic**: Quantization params computed per-batch during inference

Static is faster but less accurate; dynamic is more accurate but slower.

## Validation Criteria

Your implementation is complete when:

1. Symmetric and asymmetric quantizers implemented correctly
2. Fake quantization layers with STE work
3. QAT trainer inserts quantization layers and trains model
4. Tests verify quantization accuracy and size reduction
5. Model achieves <1% accuracy drop with 4x size reduction
6. Integration with ONNX export works (Issue #410)
7. Per-channel quantization supported

## Learning Resources

- **Original QAT Paper**: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2017)
- **PyTorch QAT Guide**: https://pytorch.org/docs/stable/quantization.html
- **TensorFlow Lite Quantization**: https://www.tensorflow.org/lite/performance/post_training_quantization
- **Survey**: Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021)

## Next Steps

1. Implement dynamic quantization for inference
2. Add INT4 and mixed-precision support
3. Integrate with ONNX export (Issue #410)
4. Benchmark INT8 inference speed vs. FP32
5. Combine with pruning (Issue #407) and distillation (Issue #408)
6. Deploy quantized models to mobile (Issue #414)

---

**Good luck!** Quantization-aware training is critical for deploying neural networks on mobile and embedded devices. Mastering QAT will enable you to achieve FP32 accuracy with INT8 efficiency.
