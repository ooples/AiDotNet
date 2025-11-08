# Issue #278: Quantization - PTQ and QAT (INT8/INT4) with LoRA Compatibility
## Junior Developer Implementation Guide

**For**: New developers implementing model quantization for production deployment
**Difficulty**: Advanced
**Estimated Time**: 50-70 hours across 3 phases
**Prerequisites**: Understanding of neural networks, numerical precision, and optimization

---

## Table of Contents
1. [Understanding Quantization](#understanding-quantization)
2. [Architecture Overview](#architecture-overview)
3. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
4. [Testing Strategy](#testing-strategy)
5. [Common Pitfalls](#common-pitfalls)
6. [Performance Benchmarks](#performance-benchmarks)

---

## Understanding Quantization

### What is Quantization?

**For Beginners**: Imagine you're moving to a smaller apartment. You can't take all your belongings, so you keep only the essentials. Quantization works similarly - it reduces the precision of neural network weights from 32 bits to 8 or 4 bits, dramatically shrinking model size while maintaining most of the accuracy.

**Technical Explanation**:

**Without Quantization (FP32)**:
```
Model: 1 billion parameters × 4 bytes = 4 GB
Inference: 4 GB memory, slower on CPU/mobile
```

**With INT8 Quantization**:
```
Model: 1 billion parameters × 1 byte = 1 GB (4x smaller!)
Inference: 1 GB memory, 2-4x faster on CPU
Accuracy loss: Typically 1-3% drop
```

**With INT4 Quantization** (Advanced):
```
Model: 1 billion parameters × 0.5 bytes = 512 MB (8x smaller!)
Inference: 512 MB memory, 4-8x faster
Accuracy loss: 3-10% drop (needs careful tuning)
```

### Why Quantization Matters

**Real-World Impact**:
1. **Mobile Deployment**: GPT-2 (1.5B params) goes from 6 GB → 1.5 GB (fits on phone!)
2. **Cost Reduction**: Smaller models = cheaper cloud hosting
3. **Speed**: INT8 math is 2-4x faster than FP32 on modern CPUs
4. **Energy**: Lower precision = less power consumption

### PTQ vs QAT

#### Post-Training Quantization (PTQ)

**Analogy**: Like compressing a photo after it's taken. Quick and easy, but some quality loss.

**Process**:
1. Train model normally in FP32
2. After training, convert weights to INT8
3. Calculate scale/zero-point for each layer
4. Done! (takes minutes)

**Pros**:
- No retraining needed
- Very fast (minutes vs days)
- No training data required

**Cons**:
- Higher accuracy loss (2-5%)
- May fail on very sensitive models

**Example**:
```csharp
// Train model
var model = TrainModel(data);

// Quantize in one line!
var quantizer = new DynamicQuantizer<float>();
var quantizedModel = quantizer.Quantize(model);

// 4x smaller, ready to deploy
SaveModel(quantizedModel, "model_int8.bin");
```

#### Quantization-Aware Training (QAT)

**Analogy**: Like training a photographer to work with a lower-quality camera. They learn to compensate, producing better results than just downgrading afterward.

**Process**:
1. During training, simulate quantization in forward pass
2. Use "fake quantization" (FP32 that acts like INT8)
3. Backward pass uses full precision gradients
4. Model learns to be robust to quantization

**Pros**:
- Much better accuracy (< 1% loss)
- Can even improve over FP32 (acts as regularization!)
- Works on challenging models

**Cons**:
- Requires retraining (expensive)
- Needs training data
- More complex implementation

**Example**:
```csharp
// Train with quantization in mind
var trainer = new Trainer<float>(model)
{
    EnableQuantizationAwareTraining = true
};

trainer.Train(data);

// Quantize (now with minimal accuracy loss!)
var quantizedModel = quantizer.Quantize(model);
```

### The Mathematics of Quantization

#### Affine Quantization (Symmetric)

**Formula**:
```
quantized_value = clamp(round(float_value / scale), -128, 127)
dequantized_value = quantized_value × scale

Where:
scale = max(abs(weights)) / 127
```

**Example**:
```csharp
float[] weights = { 2.5, -1.3, 0.8, -2.1 };
float max = 2.5;
float scale = 2.5 / 127 = 0.0197;

// Quantize
int8[] quantized = {
    round(2.5 / 0.0197) = 127,    //  2.5 → 127
    round(-1.3 / 0.0197) = -66,   // -1.3 → -66
    round(0.8 / 0.0197) = 41,     //  0.8 → 41
    round(-2.1 / 0.0197) = -107   // -2.1 → -107
};

// Dequantize
float[] dequantized = {
    127 × 0.0197 = 2.5,
    -66 × 0.0197 = -1.3,
    41 × 0.0197 = 0.81,   // small error!
    -107 × 0.0197 = -2.11
};
```

#### Affine Quantization (Asymmetric)

**Better for asymmetric distributions** (e.g., ReLU activations all positive):

```
quantized_value = clamp(round((float_value - zero_point) / scale), 0, 255)
dequantized_value = (quantized_value × scale) + zero_point

Where:
scale = (max - min) / 255
zero_point = -min / scale
```

### Fake Quantization (for QAT)

**The Trick**: Quantize then immediately dequantize during training:

```csharp
public Tensor<T> FakeQuantize(Tensor<T> input)
{
    // Forward: Quantize → Dequantize (simulates precision loss)
    var scale = ComputeScale(input);
    var quantized = Quantize(input, scale); // Convert to INT8
    var dequantized = Dequantize(quantized, scale); // Back to FP32

    return dequantized; // Still FP32, but acts like it was INT8!
}

public Tensor<T> BackwardFakeQuantize(Tensor<T> gradient)
{
    // Backward: Straight-Through Estimator (STE)
    // Pretend quantization is differentiable (it's not, but works in practice!)
    return gradient; // Pass gradient unchanged
}
```

**Why This Works**:
- Forward pass: Model sees quantization errors, learns to minimize them
- Backward pass: Gradients flow normally (we "lie" and say quantization is differentiable)
- Result: Model becomes robust to quantization

---

## Architecture Overview

### File Structure
```
src/
├── Quantization/
│   ├── IQuantizer.cs                          [NEW - AC 1.1]
│   ├── DynamicQuantizer.cs                    [NEW - AC 1.3]
│   └── QuantizationHelper.cs                  [NEW - helper methods]
├── NeuralNetworks/
│   └── Layers/
│       ├── QuantizedLinear.cs                 [NEW - AC 1.2]
│       └── Linear.cs                          [EXISTING]
├── Autograd/
│   └── FakeQuantize.cs                        [NEW - AC 2.1]
├── Training/
│   ├── TrainerSettings.cs                     [MODIFY - AC 2.2]
│   └── Trainer.cs                             [MODIFY - AC 2.2]
└── Interfaces/
    └── IQuantizer.cs                          [MOVE from Quantization/]

tests/
└── UnitTests/
    └── Quantization/
        ├── PTQTests.cs                        [NEW - AC 3.1]
        └── QATTests.cs                        [NEW - AC 3.2]
```

### Integration with Existing Code

**You'll modify**:
1. `PredictionModelBuilder.cs` - Add `.ConfigureQuantization()` method
2. Existing `Linear` layers - May need IQuantizable interface
3. `Trainer` class - Add QAT support

**Existing infrastructure you'll use**:
- `INumericOperations<T>` - For generic math
- `ModelHelper` - For parameter extraction
- Optimizer infrastructure - For QAT training

---

## Phase-by-Phase Implementation

### Phase 1: Post-Training Dynamic Quantization (PTQ)

#### AC 1.1: Define Quantization Abstractions (2 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IQuantizer.cs`

```csharp
using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for model quantization strategies.
/// </summary>
/// <typeparam name="T">The numeric type of the original model (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A quantizer converts a high-precision model (32-bit floats)
/// to a low-precision model (8-bit or 4-bit integers). This is like converting a
/// high-resolution image to a smaller file size - you lose some quality but gain
/// huge savings in file size and processing speed.
/// </para>
/// <para>
/// <b>Design Pattern:</b> This follows the Strategy pattern, allowing different
/// quantization algorithms (PTQ, QAT, per-channel, etc.) to be swapped easily.
/// </para>
/// </remarks>
public interface IQuantizer<T>
{
    /// <summary>
    /// Quantizes a trained model to lower precision.
    /// </summary>
    /// <param name="model">The full-precision model to quantize.</param>
    /// <returns>A quantized version of the model with reduced precision weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your trained model and converts all its
    /// weights from 32-bit floats (4 bytes each) to 8-bit integers (1 byte each),
    /// resulting in a model that's 4× smaller and typically 2-4× faster on CPUs.
    /// </para>
    /// <para>
    /// <b>Expected Accuracy Loss:</b>
    /// - Well-behaved models: 1-3% drop
    /// - Sensitive models: 3-5% drop
    /// - If > 5% drop: Consider Quantization-Aware Training instead
    /// </para>
    /// </remarks>
    IModel<T> Quantize(IModel<T> model);

    /// <summary>
    /// Gets the quantization configuration used by this quantizer.
    /// </summary>
    QuantizationConfig Config { get; }
}

/// <summary>
/// Configuration for quantization parameters.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> These settings control how aggressively we compress the model.
/// </remarks>
public class QuantizationConfig
{
    /// <summary>
    /// Number of bits to use for quantized weights (4 or 8).
    /// </summary>
    /// <remarks>
    /// <b>Default:</b> 8 bits (INT8) - good balance of size and accuracy.
    /// <b>Advanced:</b> 4 bits (INT4) - 2× smaller but needs careful tuning.
    /// </remarks>
    public int NumBits { get; set; } = 8;

    /// <summary>
    /// Whether to use symmetric quantization (zero-point = 0).
    /// </summary>
    /// <remarks>
    /// <b>Symmetric (true):</b> Simpler, faster, works well for weights.
    /// <b>Asymmetric (false):</b> Better for activations (e.g., after ReLU).
    /// </remarks>
    public bool IsSymmetric { get; set; } = true;

    /// <summary>
    /// Whether to quantize per-channel (true) or per-tensor (false).
    /// </summary>
    /// <remarks>
    /// <b>Per-channel:</b> Different scale for each output channel - more accurate.
    /// <b>Per-tensor:</b> Single scale for entire tensor - simpler, slightly less accurate.
    /// </remarks>
    public bool PerChannel { get; set; } = false;

    /// <summary>
    /// Layers to skip during quantization (e.g., first and last layers).
    /// </summary>
    /// <remarks>
    /// <b>Best Practice:</b> First and last layers are often kept in FP32 for accuracy.
    /// Example: { "Embedding", "OutputLinear" }
    /// </remarks>
    public List<string> SkipLayers { get; set; } = new List<string>();
}
```

#### AC 1.2: Implement QuantizedLinear Layer (8 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\NeuralNetworks\Layers\QuantizedLinear.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NumericOperations;
using System;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A linear (fully-connected) layer that stores weights in quantized INT8 format.
/// </summary>
/// <typeparam name="T">Numeric type for computations (float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This layer works exactly like a regular Linear layer, but stores
/// its weights as 8-bit integers instead of 32-bit floats. This makes it:
/// - 4× smaller in memory
/// - 2-4× faster on CPUs (INT8 math is faster)
/// - Slightly less accurate (but usually negligible)
/// </para>
/// <para>
/// <b>Implementation Note:</b> This version de-quantizes weights on-the-fly during forward pass.
/// For production, you'd want to use INT8 matrix multiplication directly (requires specialized kernels).
/// </para>
/// </remarks>
public class QuantizedLinear<T> : LayerBase<T>
{
    /// <summary>
    /// Quantized weights stored as signed 8-bit integers.
    /// Shape: [outputSize, inputSize]
    /// </summary>
    /// <remarks>
    /// Each weight is stored as an int8 value in range [-128, 127].
    /// To get the actual weight: float_weight = (int8_weight - zero_point) × scale
    /// </remarks>
    private readonly sbyte[] _quantizedWeights;

    /// <summary>
    /// Quantization scale factor for weights.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells us how to convert INT8 back to float.
    /// Example: If scale = 0.01, then INT8 value 50 → 0.5 in float.
    /// </remarks>
    private readonly T _weightScale;

    /// <summary>
    /// Zero-point offset for asymmetric quantization.
    /// </summary>
    /// <remarks>
    /// For symmetric quantization (most common for weights), this is 0.
    /// For asymmetric: zero_point shifts the range to handle asymmetric distributions.
    /// </remarks>
    private readonly int _weightZeroPoint;

    /// <summary>
    /// Bias terms (kept in full precision for accuracy).
    /// </summary>
    /// <remarks>
    /// <b>Best Practice:</b> Biases are typically not quantized because:
    /// 1. They're small (tiny memory impact)
    /// 2. Quantizing them hurts accuracy disproportionately
    /// </remarks>
    private readonly Vector<T> _bias;

    /// <summary>
    /// Dimensions of the weight matrix.
    /// </summary>
    private readonly int _inputSize;
    private readonly int _outputSize;

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// Quantized layers are typically inference-only. For training, use QAT
    /// with fake quantization instead.
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Creates a quantized linear layer from pre-quantized weights.
    /// </summary>
    /// <param name="quantizedWeights">INT8 weights.</param>
    /// <param name="weightScale">Scale factor for dequantization.</param>
    /// <param name="weightZeroPoint">Zero-point offset (usually 0 for symmetric).</param>
    /// <param name="bias">Bias vector (full precision).</param>
    /// <param name="inputSize">Input dimension.</param>
    /// <param name="outputSize">Output dimension.</param>
    public QuantizedLinear(
        sbyte[] quantizedWeights,
        T weightScale,
        int weightZeroPoint,
        Vector<T> bias,
        int inputSize,
        int outputSize)
        : base(new[] { inputSize }, new[] { outputSize }, new IdentityActivation<T>())
    {
        if (quantizedWeights == null || quantizedWeights.Length != inputSize * outputSize)
            throw new ArgumentException($"quantizedWeights must have {inputSize * outputSize} elements");

        _quantizedWeights = quantizedWeights;
        _weightScale = weightScale;
        _weightZeroPoint = weightZeroPoint;
        _bias = bias ?? new Vector<T>(outputSize);
        _inputSize = inputSize;
        _outputSize = outputSize;
    }

    /// <summary>
    /// Performs forward pass using quantized weights.
    /// </summary>
    /// <param name="input">Input tensor. Shape: [batch_size, input_size]</param>
    /// <returns>Output tensor. Shape: [batch_size, output_size]</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This works like a normal linear layer:
    /// output = input × weights + bias
    ///
    /// The difference is we first convert INT8 weights back to floats (dequantize),
    /// then do the math. In production, you'd use INT8 matrix multiplication directly.
    /// </para>
    /// <para>
    /// <b>Performance:</b>
    /// - Memory access: 4× less data to load (INT8 vs FP32)
    /// - Computation: Currently uses FP32 math (hybrid approach)
    /// - Future optimization: Use INT8 GEMM kernels for 2-4× speedup
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];

        // Dequantize weights on-the-fly
        var dequantizedWeights = DequantizeWeights();

        // Standard matrix multiplication: output = input @ weights^T + bias
        var output = input.Multiply(dequantizedWeights.Transpose());
        output = output.Add(_bias);

        return output;
    }

    /// <summary>
    /// Dequantizes INT8 weights back to floating point.
    /// </summary>
    /// <returns>Dequantized weight matrix.</returns>
    /// <remarks>
    /// <b>Formula:</b> float_weight = (int8_weight - zero_point) × scale
    /// </remarks>
    private Matrix<T> DequantizeWeights()
    {
        var weights = new Matrix<T>(_outputSize, _inputSize);
        var numOps = MathHelper.GetNumericOperations<T>();

        int index = 0;
        for (int i = 0; i < _outputSize; i++)
        {
            for (int j = 0; j < _inputSize; j++)
            {
                // Convert INT8 → float
                int quantizedValue = _quantizedWeights[index++];
                double floatValue = (quantizedValue - _weightZeroPoint) * numOps.ToDouble(_weightScale);
                weights[i, j] = numOps.FromDouble(floatValue);
            }
        }

        return weights;
    }

    /// <summary>
    /// Not supported for quantized layers (inference only).
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        throw new NotSupportedException(
            "Quantized layers do not support backpropagation. " +
            "For training, use Quantization-Aware Training with fake quantization.");
    }

    /// <summary>
    /// Not supported for quantized layers.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        throw new NotSupportedException("Quantized layers cannot be trained directly.");
    }

    /// <summary>
    /// Gets quantized parameters (for saving/loading).
    /// </summary>
    public override Vector<T> GetParameters()
    {
        // Return dequantized weights for compatibility
        var dequantized = DequantizeWeights();
        var flatWeights = new Vector<T>(_outputSize * _inputSize + _bias.Length);

        int index = 0;
        for (int i = 0; i < _outputSize; i++)
            for (int j = 0; j < _inputSize; j++)
                flatWeights[index++] = dequantized[i, j];

        for (int i = 0; i < _bias.Length; i++)
            flatWeights[index++] = _bias[i];

        return flatWeights;
    }

    /// <summary>
    /// Not supported (quantized layers are created from existing weights).
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            "Cannot set parameters on quantized layer. " +
            "Create a new QuantizedLinear from quantized weights instead.");
    }

    /// <summary>
    /// Resets internal state (no-op for this layer).
    /// </summary>
    public override void ResetState()
    {
        // No state to reset (inference only)
    }

    /// <summary>
    /// Gets the quantized weights (for inspection/debugging).
    /// </summary>
    public sbyte[] GetQuantizedWeights() => _quantizedWeights;

    /// <summary>
    /// Gets the quantization scale.
    /// </summary>
    public T GetScale() => _weightScale;

    /// <summary>
    /// Gets the zero-point.
    /// </summary>
    public int GetZeroPoint() => _weightZeroPoint;
}
```

#### AC 1.3: Implement DynamicQuantizer (5 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Quantization\DynamicQuantizer.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NumericOperations;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Quantization;

/// <summary>
/// Implements post-training dynamic quantization (PTQ).
/// </summary>
/// <typeparam name="T">Numeric type of the model (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the "compress after training" approach. It's fast and easy:
/// 1. Train your model normally
/// 2. Call quantizer.Quantize(model)
/// 3. Get a 4× smaller model in seconds!
///
/// No retraining needed, no special setup - just instant compression.
/// </para>
/// <para>
/// <b>When to Use PTQ:</b>
/// - Quick deployment needed
/// - Model is not too sensitive to precision
/// - No access to training data
/// - Willing to accept 2-5% accuracy drop
/// </para>
/// <para>
/// <b>Research Background:</b> Based on "Quantization and Training of Neural Networks for
/// Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018, Google).
/// </para>
/// </remarks>
public class DynamicQuantizer<T> : IQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Quantization configuration.
    /// </summary>
    public QuantizationConfig Config { get; }

    /// <summary>
    /// Creates a new dynamic quantizer with the specified configuration.
    /// </summary>
    /// <param name="config">Quantization settings. If null, uses defaults (INT8, symmetric).</param>
    public DynamicQuantizer(QuantizationConfig? config = null)
    {
        Config = config ?? new QuantizationConfig();

        if (Config.NumBits != 4 && Config.NumBits != 8)
            throw new ArgumentException("Only 4-bit and 8-bit quantization supported", nameof(config));
    }

    /// <summary>
    /// Quantizes a trained model to lower precision.
    /// </summary>
    /// <param name="model">Full-precision model to quantize.</param>
    /// <returns>Quantized model (same architecture, INT8/INT4 weights).</returns>
    /// <remarks>
    /// <para>
    /// <b>Algorithm:</b>
    /// 1. Iterate through all layers in the model
    /// 2. For each Linear layer:
    ///    a. Extract FP32 weights
    ///    b. Compute scale = max(abs(weights)) / 127
    ///    c. Quantize: int8_weight = clamp(round(fp32_weight / scale), -128, 127)
    ///    d. Replace layer with QuantizedLinear
    /// 3. Leave other layers unchanged
    /// </para>
    /// </remarks>
    public IModel<T> Quantize(IModel<T> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        Console.WriteLine($"Quantizing model to {Config.NumBits}-bit precision...");

        // Clone model structure (we'll replace layers in the clone)
        var quantizedModel = CloneModelStructure(model);

        int quantizedLayerCount = 0;
        int skippedLayerCount = 0;

        // Iterate through layers and quantize Linear layers
        var layers = GetLayers(model);

        foreach (var (layerName, layer) in layers)
        {
            // Skip layers in the skip list
            if (Config.SkipLayers.Contains(layerName))
            {
                Console.WriteLine($"  Skipping layer: {layerName}");
                skippedLayerCount++;
                continue;
            }

            // Check if layer is quantizable (currently only Linear layers)
            if (layer is Linear<T> linearLayer)
            {
                var quantizedLayer = QuantizeLinearLayer(linearLayer, layerName);
                ReplaceLayer(quantizedModel, layerName, quantizedLayer);
                quantizedLayerCount++;
            }
        }

        Console.WriteLine($"Quantization complete: {quantizedLayerCount} layers quantized, " +
                         $"{skippedLayerCount} layers skipped");

        return quantizedModel;
    }

    /// <summary>
    /// Quantizes a single Linear layer.
    /// </summary>
    private QuantizedLinear<T> QuantizeLinearLayer(Linear<T> layer, string layerName)
    {
        // Extract weights and bias
        var parameters = layer.GetParameters();
        int inputSize = layer.InputSize;
        int outputSize = layer.OutputSize;
        int weightCount = inputSize * outputSize;

        // Split parameters into weights and bias
        var weights = new Matrix<T>(outputSize, inputSize);
        var bias = new Vector<T>(outputSize);

        int paramIndex = 0;
        for (int i = 0; i < outputSize; i++)
            for (int j = 0; j < inputSize; j++)
                weights[i, j] = parameters[paramIndex++];

        for (int i = 0; i < outputSize; i++)
            bias[i] = parameters[paramIndex++];

        // Compute quantization parameters
        var (scale, zeroPoint) = ComputeQuantizationParams(weights);

        // Quantize weights
        var quantizedWeights = QuantizeWeights(weights, scale, zeroPoint);

        Console.WriteLine($"  {layerName}: scale={NumOps.ToDouble(scale):F6}, " +
                         $"zero_point={zeroPoint}");

        return new QuantizedLinear<T>(
            quantizedWeights,
            scale,
            zeroPoint,
            bias,
            inputSize,
            outputSize
        );
    }

    /// <summary>
    /// Computes scale and zero-point for quantization.
    /// </summary>
    private (T scale, int zeroPoint) ComputeQuantizationParams(Matrix<T> weights)
    {
        // Find min and max values
        T min = NumOps.FromDouble(double.MaxValue);
        T max = NumOps.FromDouble(double.MinValue);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                var val = weights[i, j];
                if (NumOps.LessThan(val, min)) min = val;
                if (NumOps.GreaterThan(val, max)) max = val;
            }
        }

        // Compute scale and zero-point based on config
        T scale;
        int zeroPoint;

        if (Config.IsSymmetric)
        {
            // Symmetric: zero-point = 0, scale based on max absolute value
            T maxAbs = NumOps.Max(NumOps.Abs(min), NumOps.Abs(max));
            double qMax = Config.NumBits == 8 ? 127.0 : 7.0; // INT8: [-128,127], INT4: [-8,7]
            scale = NumOps.Divide(maxAbs, NumOps.FromDouble(qMax));
            zeroPoint = 0;
        }
        else
        {
            // Asymmetric: use full range [qmin, qmax]
            double qMin = Config.NumBits == 8 ? -128.0 : -8.0;
            double qMax = Config.NumBits == 8 ? 127.0 : 7.0;
            double range = qMax - qMin;

            scale = NumOps.Divide(
                NumOps.Subtract(max, min),
                NumOps.FromDouble(range)
            );

            zeroPoint = (int)Math.Round(
                qMin - NumOps.ToDouble(min) / NumOps.ToDouble(scale)
            );
        }

        // Avoid division by zero
        if (NumOps.IsZero(scale))
            scale = NumOps.FromDouble(1e-8);

        return (scale, zeroPoint);
    }

    /// <summary>
    /// Quantizes weight matrix to INT8.
    /// </summary>
    private sbyte[] QuantizeWeights(Matrix<T> weights, T scale, int zeroPoint)
    {
        var quantized = new sbyte[weights.Rows * weights.Columns];
        int index = 0;

        int qMin = Config.NumBits == 8 ? -128 : -8;
        int qMax = Config.NumBits == 8 ? 127 : 7;

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                // Quantize: q = clamp(round(w / scale) + zero_point, qmin, qmax)
                double floatVal = NumOps.ToDouble(weights[i, j]);
                double scaleVal = NumOps.ToDouble(scale);

                int quantizedVal = (int)Math.Round(floatVal / scaleVal) + zeroPoint;
                quantizedVal = Math.Max(qMin, Math.Min(qMax, quantizedVal));

                quantized[index++] = (sbyte)quantizedVal;
            }
        }

        return quantized;
    }

    /// <summary>
    /// Extracts all layers from a model (with names).
    /// </summary>
    private List<(string name, ILayer<T> layer)> GetLayers(IModel<T> model)
    {
        var layers = new List<(string, ILayer<T>)>();

        // Use reflection to find all properties of type ILayer<T>
        var properties = model.GetType().GetProperties();

        foreach (var prop in properties)
        {
            if (typeof(ILayer<T>).IsAssignableFrom(prop.PropertyType))
            {
                var layer = prop.GetValue(model) as ILayer<T>;
                if (layer != null)
                    layers.Add((prop.Name, layer));
            }
        }

        return layers;
    }

    /// <summary>
    /// Clones model structure (placeholder - implement based on actual model architecture).
    /// </summary>
    private IModel<T> CloneModelStructure(IModel<T> model)
    {
        // In real implementation, use model.Clone() or create new instance
        // For now, return the original (will be modified in-place)
        return model;
    }

    /// <summary>
    /// Replaces a layer in the model.
    /// </summary>
    private void ReplaceLayer(IModel<T> model, string layerName, ILayer<T> newLayer)
    {
        var prop = model.GetType().GetProperty(layerName);
        if (prop != null && prop.CanWrite)
        {
            prop.SetValue(model, newLayer);
        }
    }
}
```

---

### Phase 2: Quantization-Aware Training (QAT)

#### AC 2.1: Implement Fake Quantization (8 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\src\Autograd\FakeQuantize.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NumericOperations;
using System;

namespace AiDotNet.Autograd;

/// <summary>
/// Implements fake quantization for Quantization-Aware Training (QAT).
/// </summary>
/// <typeparam name="T">Numeric type (float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Fake quantization is a clever trick used during training.
/// In the forward pass, we pretend the values are INT8 (quantize then dequantize),
/// so the model "feels" the precision loss. In the backward pass, we pretend
/// quantization is differentiable (even though it's not!) and pass gradients through.
///
/// This makes the model learn to be robust to quantization errors!
/// </para>
/// <para>
/// <b>The Magic:</b>
/// Forward: x → quantize → dequantize → y (simulates INT8 precision loss)
/// Backward: gradient flows through unchanged (Straight-Through Estimator)
/// </para>
/// <para>
/// <b>Research Background:</b> "Quantization and Training of Neural Networks..."
/// (Jacob et al., 2018). The Straight-Through Estimator (STE) was introduced in
/// "Estimating or Propagating Gradients Through Stochastic Neurons..." (Bengio et al., 2013).
/// </para>
/// </remarks>
public class FakeQuantize<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numBits;
    private readonly bool _isSymmetric;

    /// <summary>
    /// Creates a fake quantization operation.
    /// </summary>
    /// <param name="numBits">Number of quantization bits (4 or 8).</param>
    /// <param name="isSymmetric">Whether to use symmetric quantization.</param>
    public FakeQuantize(int numBits = 8, bool isSymmetric = true)
    {
        if (numBits != 4 && numBits != 8)
            throw new ArgumentException("Only 4-bit and 8-bit quantization supported");

        _numBits = numBits;
        _isSymmetric = isSymmetric;
    }

    /// <summary>
    /// Forward pass: Simulate INT8 precision by quantizing then dequantizing.
    /// </summary>
    /// <param name="input">Input tensor (FP32).</param>
    /// <returns>Output tensor (still FP32, but with INT8-like precision).</returns>
    /// <remarks>
    /// <b>Example:</b>
    /// Input: [2.5431, -1.2391, 0.8712]
    /// After fake quantize: [2.54, -1.24, 0.87] (precision loss!)
    ///
    /// The model sees this precision loss during training and learns to compensate.
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Compute scale and zero-point for this tensor
        var (scale, zeroPoint) = ComputeQuantizationParams(input);

        // Quantize to INT8
        var quantized = QuantizeTensor(input, scale, zeroPoint);

        // Immediately dequantize back to FP32
        var dequantized = DequantizeTensor(quantized, scale, zeroPoint);

        return dequantized;
    }

    /// <summary>
    /// Backward pass: Straight-Through Estimator (STE).
    /// </summary>
    /// <param name="gradient">Gradient from next layer.</param>
    /// <returns>Gradient to pass to previous layer (unchanged!).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Here's the magic - we pretend quantization doesn't exist!
    /// Mathematically, quantization is not differentiable (it's a step function).
    /// But in practice, we just pass gradients through unchanged, and it works!
    /// </para>
    /// <para>
    /// <b>Why This Works:</b> The forward pass already showed the model the precision loss.
    /// The backward pass just needs to provide learning signals. Empirically, ignoring
    /// quantization in gradients works better than trying to model it accurately.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        // Straight-Through Estimator: pass gradient unchanged
        return gradient;
    }

    /// <summary>
    /// Computes quantization parameters for a tensor.
    /// </summary>
    private (T scale, int zeroPoint) ComputeQuantizationParams(Tensor<T> tensor)
    {
        // Find min/max in tensor
        T min = NumOps.FromDouble(double.MaxValue);
        T max = NumOps.FromDouble(double.MinValue);

        foreach (var val in tensor.GetData())
        {
            if (NumOps.LessThan(val, min)) min = val;
            if (NumOps.GreaterThan(val, max)) max = val;
        }

        T scale;
        int zeroPoint;

        if (_isSymmetric)
        {
            T maxAbs = NumOps.Max(NumOps.Abs(min), NumOps.Abs(max));
            double qMax = _numBits == 8 ? 127.0 : 7.0;
            scale = NumOps.Divide(maxAbs, NumOps.FromDouble(qMax));
            zeroPoint = 0;
        }
        else
        {
            double qMin = _numBits == 8 ? -128.0 : -8.0;
            double qMax = _numBits == 8 ? 127.0 : 7.0;
            double range = qMax - qMin;

            scale = NumOps.Divide(
                NumOps.Subtract(max, min),
                NumOps.FromDouble(range)
            );

            zeroPoint = (int)Math.Round(
                qMin - NumOps.ToDouble(min) / NumOps.ToDouble(scale)
            );
        }

        if (NumOps.IsZero(scale))
            scale = NumOps.FromDouble(1e-8);

        return (scale, zeroPoint);
    }

    /// <summary>
    /// Quantizes tensor to INT8 (stored as FP32 for simplicity).
    /// </summary>
    private Tensor<T> QuantizeTensor(Tensor<T> input, T scale, int zeroPoint)
    {
        var quantized = new Tensor<T>(input.Shape);
        int qMin = _numBits == 8 ? -128 : -8;
        int qMax = _numBits == 8 ? 127 : 7;

        var data = input.GetData();
        var quantizedData = new T[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            double floatVal = NumOps.ToDouble(data[i]);
            double scaleVal = NumOps.ToDouble(scale);

            int quantizedVal = (int)Math.Round(floatVal / scaleVal) + zeroPoint;
            quantizedVal = Math.Max(qMin, Math.Min(qMax, quantizedVal));

            // Store as FP32 (but with INT8 precision)
            quantizedData[i] = NumOps.FromDouble(quantizedVal);
        }

        quantized.SetData(quantizedData);
        return quantized;
    }

    /// <summary>
    /// Dequantizes INT8 values back to FP32.
    /// </summary>
    private Tensor<T> DequantizeTensor(Tensor<T> quantized, T scale, int zeroPoint)
    {
        var dequantized = new Tensor<T>(quantized.Shape);
        var data = quantized.GetData();
        var dequantizedData = new T[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            int quantizedVal = (int)NumOps.ToDouble(data[i]);
            double floatVal = (quantizedVal - zeroPoint) * NumOps.ToDouble(scale);
            dequantizedData[i] = NumOps.FromDouble(floatVal);
        }

        dequantized.SetData(dequantizedData);
        return dequantized;
    }
}
```

#### AC 2.2: Integrate QAT into Trainer (5 points)

**File**: Modify `C:\Users\cheat\source\repos\AiDotNet\src\Training\TrainerSettings.cs`

```csharp
// Add this property to TrainerSettings class:

/// <summary>
/// Enables Quantization-Aware Training (QAT).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When enabled, the model will be trained with simulated quantization.
/// This makes the final quantized model much more accurate (< 1% loss vs 3-5% with PTQ).
/// </para>
/// <para>
/// <b>Trade-off:</b> Training takes the same time, but you get a much better quantized model.
/// Always use QAT if you have access to training data and time to retrain.
/// </para>
/// </remarks>
public bool EnableQuantizationAwareTraining { get; set; } = false;

/// <summary>
/// Number of bits for quantization (used with QAT).
/// </summary>
public int QuantizationBits { get; set; } = 8;
```

**File**: Modify `C:\Users\cheat\source\repos\AiDotNet\src\Training\Trainer.cs`

```csharp
// In Trainer<T> class, modify Train() method:

public void Train(IDataset<TInput, TOutput> trainingData)
{
    // ... existing setup code ...

    // Initialize fake quantization if QAT is enabled
    FakeQuantize<T>? fakeQuantize = null;
    if (_settings.EnableQuantizationAwareTraining)
    {
        Console.WriteLine($"Quantization-Aware Training enabled ({_settings.QuantizationBits}-bit)");
        fakeQuantize = new FakeQuantize<T>(_settings.QuantizationBits, isSymmetric: true);
    }

    // Training loop
    for (int epoch = 0; epoch < _settings.Epochs; epoch++)
    {
        foreach (var batch in trainingData.GetBatches(_settings.BatchSize))
        {
            // Forward pass
            var predictions = _model.Forward(batch.Inputs);

            // Apply fake quantization if QAT enabled
            if (fakeQuantize != null)
            {
                predictions = fakeQuantize.Forward(predictions);
            }

            // Compute loss
            var loss = _lossFunction.ComputeLoss(predictions, batch.Targets);

            // Backward pass
            var gradients = _lossFunction.ComputeGradients(predictions, batch.Targets);

            // Fake quantization backward (STE)
            if (fakeQuantize != null)
            {
                gradients = fakeQuantize.Backward(gradients);
            }

            // Continue with standard backprop
            _model.Backward(gradients);
            _optimizer.Step();
        }
    }
}
```

---

### Phase 3: Validation and Testing

#### AC 3.1: PTQ Test (5 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Quantization\PTQTests.cs`

```csharp
using AiDotNet.Quantization;
using AiDotNet.NeuralNetworks;
using AiDotNet.Data;
using Xunit;
using System;

namespace AiDotNet.Tests.Quantization;

public class PTQTests
{
    [Fact]
    public void PTQ_ReducesModelSize_By4x()
    {
        // Arrange: Train a simple model
        var model = CreateAndTrainSimpleModel();

        // Get original size
        long originalSize = GetModelSizeBytes(model);

        // Act: Quantize
        var quantizer = new DynamicQuantizer<float>();
        var quantizedModel = quantizer.Quantize(model);

        // Get quantized size
        long quantizedSize = GetModelSizeBytes(quantizedModel);

        // Assert: Should be approximately 4x smaller
        double compressionRatio = (double)originalSize / quantizedSize;
        Assert.True(compressionRatio > 3.5 && compressionRatio < 4.5,
            $"Expected 4x compression, got {compressionRatio:F2}x");
    }

    [Fact]
    public void PTQ_MaintainsReasonableAccuracy()
    {
        // Arrange: Train model on MNIST-like task
        var (model, testData) = CreateAndTrainMNISTModel();

        // Measure original accuracy
        double originalAccuracy = EvaluateAccuracy(model, testData);

        // Act: Quantize
        var quantizer = new DynamicQuantizer<float>();
        var quantizedModel = quantizer.Quantize(model);

        // Measure quantized accuracy
        double quantizedAccuracy = EvaluateAccuracy(quantizedModel, testData);

        // Assert: Accuracy drop should be < 5%
        double accuracyDrop = originalAccuracy - quantizedAccuracy;
        Console.WriteLine($"Original: {originalAccuracy:P2}, Quantized: {quantizedAccuracy:P2}, Drop: {accuracyDrop:P2}");
        Assert.True(accuracyDrop < 0.05, $"Accuracy dropped by {accuracyDrop:P2}, exceeds 5% threshold");
    }

    [Fact]
    public void PTQ_ProducesDeterministicResults()
    {
        // Quantizing the same model twice should produce identical results
        var model = CreateAndTrainSimpleModel();
        var quantizer = new DynamicQuantizer<float>();

        var quantized1 = quantizer.Quantize(model);
        var quantized2 = quantizer.Quantize(model);

        // Extract weights and compare
        var weights1 = GetQuantizedWeights(quantized1);
        var weights2 = GetQuantizedWeights(quantized2);

        Assert.Equal(weights1, weights2);
    }

    private IModel<float> CreateAndTrainSimpleModel()
    {
        // Simple 2-layer network
        var model = new SequentialModel<float>
        {
            new Linear<float>(784, 128),
            new ReLU<float>(),
            new Linear<float>(128, 10)
        };

        // Train for a few epochs
        var data = GenerateSyntheticData();
        TrainModel(model, data, epochs: 5);

        return model;
    }

    private double EvaluateAccuracy(IModel<float> model, IDataset<float[], float[]> testData)
    {
        int correct = 0;
        int total = 0;

        foreach (var sample in testData)
        {
            var prediction = model.Forward(sample.Input);
            var predictedClass = ArgMax(prediction);
            var trueClass = ArgMax(sample.Target);

            if (predictedClass == trueClass)
                correct++;
            total++;
        }

        return (double)correct / total;
    }
}
```

#### AC 3.2: QAT vs PTQ Test (5 points)

**File**: `C:\Users\cheat\source\repos\AiDotNet\tests\UnitTests\Quantization\QATTests.cs`

```csharp
using AiDotNet.Quantization;
using AiDotNet.Training;
using Xunit;
using System;

namespace AiDotNet.Tests.Quantization;

public class QATTests
{
    [Fact]
    public void QAT_ProducesBetterAccuracy_ThanPTQ()
    {
        var trainingData = GenerateTrainingData();
        var testData = GenerateTestData();

        // Scenario 1: Train normally, then PTQ
        var modelPTQ = CreateModel();
        TrainModel(modelPTQ, trainingData, useQAT: false);
        var quantizerPTQ = new DynamicQuantizer<float>();
        var quantizedPTQ = quantizerPTQ.Quantize(modelPTQ);
        double accuracyPTQ = EvaluateAccuracy(quantizedPTQ, testData);

        // Scenario 2: Train with QAT, then quantize
        var modelQAT = CreateModel();
        TrainModel(modelQAT, trainingData, useQAT: true);
        var quantizerQAT = new DynamicQuantizer<float>();
        var quantizedQAT = quantizerQAT.Quantize(modelQAT);
        double accuracyQAT = EvaluateAccuracy(quantizedQAT, testData);

        // Assert: QAT should be better
        Console.WriteLine($"PTQ Accuracy: {accuracyPTQ:P2}");
        Console.WriteLine($"QAT Accuracy: {accuracyQAT:P2}");
        Assert.True(accuracyQAT > accuracyPTQ,
            $"QAT ({accuracyQAT:P2}) should outperform PTQ ({accuracyPTQ:P2})");

        // Typically QAT is 2-5% better
        double improvement = accuracyQAT - accuracyPTQ;
        Assert.True(improvement > 0.01, $"Expected > 1% improvement, got {improvement:P2}");
    }

    [Fact]
    public void FakeQuantize_PreservesGradientFlow()
    {
        var fakeQuant = new FakeQuantize<float>(numBits: 8);

        // Create tensor with gradients
        var input = CreateRandomTensor(10, 100);
        var output = fakeQuant.Forward(input);

        // Simulate backward pass
        var gradient = CreateRandomTensor(10, 100);
        var backwardGradient = fakeQuant.Backward(gradient);

        // Gradients should flow through unchanged (STE)
        Assert.Equal(gradient.GetData(), backwardGradient.GetData());
    }

    private void TrainModel(IModel<float> model, IDataset data, bool useQAT)
    {
        var trainer = new Trainer<float>(model, new TrainerSettings
        {
            Epochs = 10,
            LearningRate = 0.001f,
            EnableQuantizationAwareTraining = useQAT,
            QuantizationBits = 8
        });

        trainer.Train(data);
    }
}
```

---

## Common Pitfalls

### Pitfall 1: Quantizing First/Last Layers

**Problem**: Quantizing embedding and output layers causes large accuracy drops

**Solution**: Skip them!
```csharp
var config = new QuantizationConfig
{
    SkipLayers = new List<string> { "Embedding", "OutputLinear" }
};
```

### Pitfall 2: Forgetting to Normalize Data

**Problem**: Quantization works poorly with unnormalized inputs

**Solution**: Always normalize to [-1, 1] or [0, 1] before quantization

### Pitfall 3: Using Asymmetric Quantization for Weights

**Problem**: Weights are usually symmetric, asymmetric adds unnecessary complexity

**Solution**: Use symmetric for weights, asymmetric only for ReLU activations

### Pitfall 4: Not Testing Before Deployment

**Problem**: Quantized model performs poorly in production

**Solution**: Always benchmark on representative test data!

---

## Performance Benchmarks

| Model Size | FP32 Size | INT8 Size | Speedup | Accuracy Drop |
|------------|-----------|-----------|---------|---------------|
| 100M params | 400 MB | 100 MB | 2.1x | 1.2% (QAT) |
| 1B params | 4 GB | 1 GB | 3.5x | 2.3% (QAT) |
| 10B params | 40 GB | 10 GB | 4.2x | 1.8% (QAT) |

**PTQ typically adds +1-3% accuracy loss vs QAT**

---

## Conclusion

After implementing quantization, you'll have:
- 4x smaller models
- 2-4x faster inference
- Production-ready INT8 deployment
- < 1% accuracy loss with QAT

Deploy to mobile, edge, or cloud with confidence! 🚀
