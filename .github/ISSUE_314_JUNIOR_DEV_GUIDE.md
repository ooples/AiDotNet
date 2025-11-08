# Issue #314: Junior Developer Implementation Guide
## Implement 8-bit Adam Optimizer for Memory Efficiency

## Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [What is 8-bit Quantization?](#what-is-8-bit-quantization)
3. [Existing Adam Infrastructure](#existing-adam-infrastructure)
4. [8-bit Quantization Scheme](#8-bit-quantization-scheme)
5. [Implementation Guide](#implementation-guide)
6. [Testing Strategy](#testing-strategy)
7. [Numerical Stability Considerations](#numerical-stability-considerations)

---

## Understanding the Problem

### The Memory Problem with Optimizers

**Standard Adam Optimizer:**
- Stores momentum (`m`) in 32-bit floats
- Stores variance (`v`) in 32-bit floats
- For a model with 1 billion parameters:
  - Model weights: 4 GB (1B × 4 bytes)
  - Adam momentum: 4 GB
  - Adam variance: 4 GB
  - **Total: 12 GB** (optimizer states use 8 GB!)

**8-bit Adam Optimizer:**
- Stores momentum in 8-bit format
- Stores variance in 8-bit format
- For the same model:
  - Model weights: 4 GB
  - Adam momentum: 1 GB (8-bit)
  - Adam variance: 1 GB (8-bit)
  - **Total: 6 GB** (50% memory reduction!)

### Real-World Impact

| Model Size | Standard Adam | 8-bit Adam | Memory Saved |
|------------|---------------|------------|--------------|
| 100M params | 1.2 GB | 0.6 GB | 600 MB |
| 1B params | 12 GB | 6 GB | 6 GB |
| 10B params | 120 GB | 60 GB | 60 GB |

**Why This Matters:**
- Train larger models on the same hardware
- Train on consumer GPUs (8-16 GB VRAM)
- Reduce cloud training costs
- Enable edge device training

---

## What is 8-bit Quantization?

### Basic Concept

**Quantization** = Representing high-precision values with lower-precision values

**Example:**
```
32-bit float range: -3.4 × 10^38 to +3.4 × 10^38 (billions of unique values)
8-bit integer range: -128 to +127 (256 unique values)
```

### Block-wise Quantization (Best for Adam)

Instead of quantizing the entire vector at once, divide it into blocks:

```
Original vector (1000 elements):
[0.1, 0.2, ..., 0.15] → [5.2, 5.3, ..., 5.1] → [0.01, 0.02, ..., 0.015]
     Block 0                  Block 1                   Block 2

Each block has its own scale factor:
Block 0: scale = 0.002, offset = 0.0
Block 1: scale = 0.05, offset = 5.0
Block 2: scale = 0.0002, offset = 0.0
```

**Why Block-wise?**
- Handles varying magnitudes within the same vector
- Better numerical precision than global quantization
- Industry standard (used by bitsandbytes library)

### Quantization Formula

**Encode (float32 → int8):**
```csharp
// For each block of 256 elements
float min = block.Min();
float max = block.Max();
float scale = (max - min) / 255.0f;

for (int i = 0; i < block.Length; i++)
{
    byte quantized = (byte)((block[i] - min) / scale);
}
```

**Decode (int8 → float32):**
```csharp
float value = min + (quantized * scale);
```

---

## Existing Adam Infrastructure

### Standard AdamOptimizer Pattern

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/Optimizers/AdamOptimizer.cs`

Key components:
```csharp
public class AdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    // 32-bit storage (CURRENT)
    private Vector<T> _m;  // First moment (momentum)
    private Vector<T> _v;  // Second moment (variance)
    private int _t;        // Time step

    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        _t++;

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update momentum: m = beta1 * m + (1 - beta1) * gradient
            _m[i] = NumOps.Add(
                NumOps.Multiply(_m[i], NumOps.FromDouble(_options.Beta1)),
                NumOps.Multiply(gradient[i], NumOps.FromDouble(1 - _options.Beta1))
            );

            // Update variance: v = beta2 * v + (1 - beta2) * gradient^2
            _v[i] = NumOps.Add(
                NumOps.Multiply(_v[i], NumOps.FromDouble(_options.Beta2)),
                NumOps.Multiply(NumOps.Multiply(gradient[i], gradient[i]),
                    NumOps.FromDouble(1 - _options.Beta2))
            );

            // Bias correction
            T mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
            T vHat = NumOps.Divide(_v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            // Update parameters
            T update = NumOps.Divide(mHat, NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon)));
            parameters[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate)));
        }

        return parameters;
    }
}
```

---

## 8-bit Quantization Scheme

### Recommended Scheme: Block-wise Dynamic Quantization

**Parameters:**
- **Block size:** 256 elements (industry standard)
- **Quantization range:** [-127, 127] (signed 8-bit)
- **Dynamic scaling:** Each block has its own scale factor

### Data Structure Design

```csharp
/// <summary>
/// Represents a quantized vector with block-wise scaling.
/// </summary>
public class QuantizedVector<T>
{
    // Quantized data (8-bit integers)
    private byte[] _data;

    // Scale factors for each block
    private T[] _scales;

    // Offset values for each block
    private T[] _offsets;

    // Block size (256 is standard)
    private const int BlockSize = 256;

    // Number of blocks
    private int _numBlocks;

    // Original vector length
    private int _length;
}
```

### Core Operations

#### 1. Quantize (float32 → int8)

```csharp
public static QuantizedVector<T> Quantize(Vector<T> vector, INumericOperations<T> numOps)
{
    int numBlocks = (vector.Length + BlockSize - 1) / BlockSize;
    var quantized = new QuantizedVector<T>
    {
        _data = new byte[vector.Length],
        _scales = new T[numBlocks],
        _offsets = new T[numBlocks],
        _numBlocks = numBlocks,
        _length = vector.Length
    };

    for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
    {
        int start = blockIdx * BlockSize;
        int end = Math.Min(start + BlockSize, vector.Length);

        // Find min and max in this block
        T min = vector[start];
        T max = vector[start];
        for (int i = start + 1; i < end; i++)
        {
            if (numOps.LessThan(vector[i], min)) min = vector[i];
            if (numOps.GreaterThan(vector[i], max)) max = vector[i];
        }

        // Calculate scale and offset
        T range = numOps.Subtract(max, min);
        T scale = numOps.Divide(range, numOps.FromDouble(255.0));
        quantized._scales[blockIdx] = scale;
        quantized._offsets[blockIdx] = min;

        // Quantize elements in this block
        for (int i = start; i < end; i++)
        {
            T normalized = numOps.Divide(
                numOps.Subtract(vector[i], min),
                scale
            );
            quantized._data[i] = (byte)Math.Round(numOps.ToDouble(normalized));
        }
    }

    return quantized;
}
```

#### 2. Dequantize (int8 → float32)

```csharp
public Vector<T> Dequantize(INumericOperations<T> numOps)
{
    var result = new Vector<T>(_length);

    for (int blockIdx = 0; blockIdx < _numBlocks; blockIdx++)
    {
        int start = blockIdx * BlockSize;
        int end = Math.Min(start + BlockSize, _length);

        T scale = _scales[blockIdx];
        T offset = _offsets[blockIdx];

        for (int i = start; i < end; i++)
        {
            T value = numOps.Add(
                offset,
                numOps.Multiply(numOps.FromDouble(_data[i]), scale)
            );
            result[i] = value;
        }
    }

    return result;
}
```

---

## Implementation Guide

### Step 1: Create QuantizedVector Helper Class

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/Helpers/QuantizedVector.cs`

```csharp
namespace AiDotNet.Helpers;

/// <summary>
/// Represents a vector quantized to 8-bit precision using block-wise dynamic quantization.
/// </summary>
/// <typeparam name="T">The numeric type for dequantized values (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class stores floating-point numbers in 8-bit format to save memory.
/// Instead of storing each number with 32 bits (4 bytes), we store it with 8 bits (1 byte).
///
/// Think of it like this:
/// - Original: High-quality photo (32-bit per pixel) = 10 MB
/// - Quantized: Compressed JPEG (8-bit per pixel) = 2.5 MB
/// - You lose some quality, but save 75% space
///
/// Block-wise quantization:
/// - Divide the vector into blocks of 256 elements
/// - Each block has its own compression settings (scale and offset)
/// - This preserves more accuracy than compressing the whole vector at once
///
/// Used by: 8-bit Adam optimizer to reduce memory usage by 75%
/// </para>
/// </remarks>
public class QuantizedVector<T>
{
    /// <summary>
    /// The quantized data (8-bit integers).
    /// </summary>
    private byte[] _data;

    /// <summary>
    /// Scale factors for each block.
    /// </summary>
    private T[] _scales;

    /// <summary>
    /// Offset values for each block.
    /// </summary>
    private T[] _offsets;

    /// <summary>
    /// Block size for quantization (256 is industry standard).
    /// </summary>
    private const int BlockSize = 256;

    /// <summary>
    /// Number of blocks in the quantized vector.
    /// </summary>
    private int _numBlocks;

    /// <summary>
    /// Original vector length.
    /// </summary>
    private int _length;

    /// <summary>
    /// Gets the length of the original vector.
    /// </summary>
    public int Length => _length;

    /// <summary>
    /// Quantizes a vector to 8-bit precision using block-wise dynamic quantization.
    /// </summary>
    /// <param name="vector">The vector to quantize.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    /// <returns>A quantized vector with 8-bit precision.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts high-precision numbers to low-precision format.
    ///
    /// Steps:
    /// 1. Divide vector into blocks of 256 elements
    /// 2. For each block, find min and max values
    /// 3. Calculate scale = (max - min) / 255
    /// 4. Convert each value: quantized = (value - min) / scale
    /// 5. Store as 8-bit integer (0-255)
    ///
    /// Example block:
    /// - Values: [0.1, 0.5, 0.9]
    /// - Min: 0.1, Max: 0.9
    /// - Scale: (0.9 - 0.1) / 255 = 0.00314
    /// - Quantized: [(0.1-0.1)/0.00314, (0.5-0.1)/0.00314, (0.9-0.1)/0.00314]
    /// - Result: [0, 127, 255]
    /// </remarks>
    public static QuantizedVector<T> Quantize(Vector<T> vector, INumericOperations<T> numOps)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Vector cannot be empty", nameof(vector));

        int numBlocks = (vector.Length + BlockSize - 1) / BlockSize;
        var quantized = new QuantizedVector<T>
        {
            _data = new byte[vector.Length],
            _scales = new T[numBlocks],
            _offsets = new T[numBlocks],
            _numBlocks = numBlocks,
            _length = vector.Length
        };

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int start = blockIdx * BlockSize;
            int end = Math.Min(start + BlockSize, vector.Length);

            // Find min and max in this block
            T min = vector[start];
            T max = vector[start];
            for (int i = start + 1; i < end; i++)
            {
                if (numOps.LessThan(vector[i], min)) min = vector[i];
                if (numOps.GreaterThan(vector[i], max)) max = vector[i];
            }

            // Handle edge case: all values in block are identical
            T range = numOps.Subtract(max, min);
            T scale;
            if (numOps.Compare(range, numOps.Zero) == 0)
            {
                // All values are the same, use scale = 1 to avoid division by zero
                scale = numOps.One;
            }
            else
            {
                scale = numOps.Divide(range, numOps.FromDouble(255.0));
            }

            quantized._scales[blockIdx] = scale;
            quantized._offsets[blockIdx] = min;

            // Quantize elements in this block
            for (int i = start; i < end; i++)
            {
                T normalized = numOps.Divide(
                    numOps.Subtract(vector[i], min),
                    scale
                );
                // Clamp to [0, 255] range
                double normalizedValue = numOps.ToDouble(normalized);
                normalizedValue = Math.Max(0.0, Math.Min(255.0, normalizedValue));
                quantized._data[i] = (byte)Math.Round(normalizedValue);
            }
        }

        return quantized;
    }

    /// <summary>
    /// Dequantizes the vector back to full precision.
    /// </summary>
    /// <param name="numOps">Numeric operations for type T.</param>
    /// <returns>The dequantized vector with full precision.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method converts 8-bit values back to high-precision numbers.
    ///
    /// Steps:
    /// 1. For each block, retrieve scale and offset
    /// 2. For each 8-bit value: value = offset + (quantized * scale)
    ///
    /// Example:
    /// - Quantized: [0, 127, 255]
    /// - Offset: 0.1, Scale: 0.00314
    /// - Dequantized: [0.1 + 0*0.00314, 0.1 + 127*0.00314, 0.1 + 255*0.00314]
    /// - Result: [0.1, 0.499, 0.900] (close to original [0.1, 0.5, 0.9])
    /// </remarks>
    public Vector<T> Dequantize(INumericOperations<T> numOps)
    {
        var result = new Vector<T>(_length);

        for (int blockIdx = 0; blockIdx < _numBlocks; blockIdx++)
        {
            int start = blockIdx * BlockSize;
            int end = Math.Min(start + BlockSize, _length);

            T scale = _scales[blockIdx];
            T offset = _offsets[blockIdx];

            for (int i = start; i < end; i++)
            {
                T value = numOps.Add(
                    offset,
                    numOps.Multiply(numOps.FromDouble(_data[i]), scale)
                );
                result[i] = value;
            }
        }

        return result;
    }
}
```

### Step 2: Create Adam8BitOptimizer

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/Optimizers/Adam8BitOptimizer.cs`

```csharp
namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the 8-bit Adam optimizer for memory-efficient training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is a memory-efficient version of the Adam optimizer.
/// It stores optimizer states (momentum and variance) in 8-bit format instead of 32-bit,
/// reducing memory usage by approximately 75%.
///
/// Memory comparison for 1 billion parameters:
/// - Standard Adam: 12 GB (4 GB weights + 4 GB momentum + 4 GB variance)
/// - 8-bit Adam: 6 GB (4 GB weights + 1 GB momentum + 1 GB variance)
///
/// How it works:
/// 1. Calculate gradients normally (32-bit)
/// 2. Update momentum and variance normally (32-bit)
/// 3. Quantize momentum and variance to 8-bit for storage
/// 4. Dequantize when needed for next update
///
/// Trade-offs:
/// - Pro: 75% less memory for optimizer states
/// - Pro: Train larger models on same hardware
/// - Con: Slight numerical precision loss (usually negligible)
/// - Con: Small computational overhead for quantization
///
/// Use this when:
/// - GPU memory is limited
/// - Training very large models
/// - Memory is more critical than slight speed loss
/// </para>
/// </remarks>
public class Adam8BitOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Adam optimizer.
    /// </summary>
    private AdamOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (momentum) stored in 8-bit format.
    /// </summary>
    private QuantizedVector<T>? _mQuantized;

    /// <summary>
    /// The second moment vector (variance) stored in 8-bit format.
    /// </summary>
    private QuantizedVector<T>? _vQuantized;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the Adam8BitOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Adam optimizer.</param>
    public Adam8BitOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        AdamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new();
        _mQuantized = null;
        _vQuantized = null;
        _t = 0;
    }

    /// <summary>
    /// Updates a vector of parameters using the 8-bit Adam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method applies the Adam algorithm with 8-bit quantized states.
    ///
    /// Steps:
    /// 1. Dequantize momentum and variance from 8-bit to 32-bit
    /// 2. Update momentum: m = beta1 * m + (1 - beta1) * gradient
    /// 3. Update variance: v = beta2 * v + (1 - beta2) * gradient^2
    /// 4. Apply bias correction
    /// 5. Calculate parameter update
    /// 6. Quantize momentum and variance back to 8-bit for storage
    ///
    /// The key difference from standard Adam:
    /// - Standard Adam: Keep m and v in memory as 32-bit (uses more memory)
    /// - 8-bit Adam: Store m and v as 8-bit, convert to 32-bit only during updates
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (parameters.Length != gradient.Length)
            throw new ArgumentException("Parameters and gradient must have the same length");

        // Initialize quantized states on first call
        if (_mQuantized == null || _vQuantized == null)
        {
            var mInit = new Vector<T>(parameters.Length);
            var vInit = new Vector<T>(parameters.Length);
            _mQuantized = QuantizedVector<T>.Quantize(mInit, NumOps);
            _vQuantized = QuantizedVector<T>.Quantize(vInit, NumOps);
            _t = 0;
        }

        _t++;

        // Dequantize states to 32-bit for computation
        var m = _mQuantized.Dequantize(NumOps);
        var v = _vQuantized.Dequantize(NumOps);

        // Standard Adam update
        for (int i = 0; i < parameters.Length; i++)
        {
            // Update first moment (momentum)
            m[i] = NumOps.Add(
                NumOps.Multiply(m[i], NumOps.FromDouble(_options.Beta1)),
                NumOps.Multiply(gradient[i], NumOps.FromDouble(1 - _options.Beta1))
            );

            // Update second moment (variance)
            v[i] = NumOps.Add(
                NumOps.Multiply(v[i], NumOps.FromDouble(_options.Beta2)),
                NumOps.Multiply(NumOps.Multiply(gradient[i], gradient[i]),
                    NumOps.FromDouble(1 - _options.Beta2))
            );

            // Bias correction
            T mHat = NumOps.Divide(m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
            T vHat = NumOps.Divide(v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            // Calculate update
            T update = NumOps.Divide(
                mHat,
                NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
            );

            // Apply update
            parameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
            );
        }

        // Quantize states back to 8-bit for storage
        _mQuantized = QuantizedVector<T>.Quantize(m, NumOps);
        _vQuantized = QuantizedVector<T>.Quantize(v, NumOps);

        return parameters;
    }

    /// <summary>
    /// Updates a matrix of parameters using the 8-bit Adam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter matrix to be updated.</param>
    /// <param name="gradient">The gradient matrix corresponding to the parameters.</param>
    /// <returns>The updated parameter matrix.</returns>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        if (gradient == null) throw new ArgumentNullException(nameof(gradient));
        if (parameters.Rows != gradient.Rows || parameters.Columns != gradient.Columns)
            throw new ArgumentException("Parameters and gradient must have the same dimensions");

        // Flatten to vector, update, reshape back
        var paramVector = parameters.Flatten();
        var gradVector = gradient.Flatten();

        var updatedVector = UpdateParameters(paramVector, gradVector);

        return updatedVector.Reshape(parameters.Rows, parameters.Columns);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    public override void Reset()
    {
        _mQuantized = null;
        _vQuantized = null;
        _t = 0;
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdamOptimizerOptions<T, TInput, TOutput> adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdamOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the approximate memory usage of the optimizer states in bytes.
    /// </summary>
    /// <returns>The memory usage in bytes.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method calculates how much memory the optimizer is using.
    ///
    /// For comparison:
    /// - Standard Adam: parameterCount * 4 bytes * 2 (m and v) = 8 bytes per parameter
    /// - 8-bit Adam: parameterCount * 1 byte * 2 (m and v) + overhead = ~2 bytes per parameter
    ///
    /// The overhead comes from storing scale and offset for each block (256 elements).
    /// </remarks>
    public long GetMemoryUsageBytes()
    {
        if (_mQuantized == null || _vQuantized == null)
            return 0;

        // 8-bit data: 1 byte per element
        long dataSize = _mQuantized.Length * 2; // m and v

        // Scales and offsets: depends on block size
        int numBlocks = (_mQuantized.Length + 255) / 256;
        long metadataSize = numBlocks * 2 * sizeof(double) * 2; // scales and offsets for m and v

        return dataSize + metadataSize;
    }
}
```

### Step 3: Create Unit Tests

**File:** `C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/Optimizers/Adam8BitOptimizerTests.cs`

```csharp
namespace AiDotNet.Tests.Optimizers;

public class Adam8BitOptimizerTests
{
    [Fact]
    public void UpdateParameters_SimpleGradient_ConvergesCorrectly()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // Start with parameters far from optimum
        var parameters = new Vector<double>(new[] { 10.0, 10.0, 10.0 });

        // Gradient pointing towards zero (optimum)
        var gradient = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

        // Act - run multiple updates
        for (int i = 0; i < 100; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - parameters should move towards zero
        Assert.True(parameters[0] < 10.0); // Moved in right direction
        Assert.True(parameters[1] < 10.0);
        Assert.True(parameters[2] < 10.0);
    }

    [Fact]
    public void UpdateParameters_LargeParameterCount_UsesLessMemoryThanStandardAdam()
    {
        // Arrange
        int paramCount = 10000; // 10K parameters
        var optimizer8bit = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var optimizerStandard = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null);

        var parameters = new Vector<double>(paramCount);
        var gradient = new Vector<double>(paramCount);

        // Initialize both optimizers
        optimizer8bit.UpdateParameters(parameters, gradient);
        optimizerStandard.UpdateParameters(parameters, gradient);

        // Act
        long memory8bit = optimizer8bit.GetMemoryUsageBytes();
        long memoryStandard = paramCount * sizeof(double) * 2; // m and v in 32-bit

        // Assert - 8-bit should use significantly less memory
        Assert.True(memory8bit < memoryStandard);
        double reduction = 1.0 - ((double)memory8bit / memoryStandard);
        Assert.True(reduction > 0.5); // At least 50% reduction
    }

    [Fact]
    public void UpdateParameters_MultipleCalls_MaintainsNumericalStability()
    {
        // Arrange
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var gradient = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });

        // Act - run many updates
        for (int i = 0; i < 1000; i++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        // Assert - no NaN or Infinity
        Assert.All(parameters.ToArray(), p => Assert.False(double.IsNaN(p)));
        Assert.All(parameters.ToArray(), p => Assert.False(double.IsInfinity(p)));
    }

    [Fact]
    public void UpdateParameters_ComparedToStandardAdam_ProducesSimilarResults()
    {
        // Arrange
        var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };

        var optimizer8bit = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);
        var optimizerStandard = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var parameters8bit = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });
        var parametersStandard = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 });
        var gradient = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act - run same updates
        for (int i = 0; i < 50; i++)
        {
            parameters8bit = optimizer8bit.UpdateParameters(parameters8bit, gradient);
            parametersStandard = optimizerStandard.UpdateParameters(parametersStandard, gradient);
        }

        // Assert - results should be similar (within 5% due to quantization)
        for (int i = 0; i < parameters8bit.Length; i++)
        {
            double diff = Math.Abs(parameters8bit[i] - parametersStandard[i]);
            double relativeDiff = diff / Math.Abs(parametersStandard[i]);
            Assert.True(relativeDiff < 0.05); // Within 5%
        }
    }

    [Fact]
    public void Reset_AfterUpdates_ClearsState()
    {
        // Arrange
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null);
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var gradient = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

        // Initialize state
        optimizer.UpdateParameters(parameters, gradient);
        long memoryBefore = optimizer.GetMemoryUsageBytes();

        // Act
        optimizer.Reset();
        long memoryAfter = optimizer.GetMemoryUsageBytes();

        // Assert
        Assert.True(memoryBefore > 0);
        Assert.Equal(0, memoryAfter);
    }
}
```

---

## Testing Strategy

### 1. Correctness Tests
- Verify convergence on simple optimization problems
- Compare results with standard Adam (should be within 5%)
- Test gradient descent on quadratic functions

### 2. Memory Usage Tests
- Measure actual memory consumption
- Verify 50-75% memory reduction
- Test with varying parameter counts (1K, 10K, 100K, 1M)

### 3. Numerical Stability Tests
- Run 1000+ iterations without NaN/Infinity
- Test with extreme gradient values
- Test with very small learning rates

### 4. Edge Cases
- Empty vectors (should throw)
- Single-element vectors
- All-zero gradients
- All-identical values in a block

### 5. Integration Tests
- Train a simple neural network
- Verify loss decreases over epochs
- Compare training curves with standard Adam

---

## Numerical Stability Considerations

### 1. Quantization Error Accumulation

**Problem:** Small errors from quantization can accumulate over many iterations

**Solution:**
- Use bias correction (already in Adam algorithm)
- Periodic full-precision checkpoints
- Monitor training loss carefully

### 2. Block-wise Scaling Edge Cases

**Problem:** If all values in a block are identical, scale = 0 (division by zero)

**Solution:**
```csharp
if (numOps.Compare(range, numOps.Zero) == 0)
{
    scale = numOps.One; // Use scale = 1 to avoid division by zero
}
```

### 3. Gradient Clipping

**Recommendation:** Use gradient clipping with 8-bit Adam
```csharp
// Before UpdateParameters
gradient = ClipGradientNorm(gradient, maxNorm: 1.0);
```

### 4. Monitoring Training

**Watch for:**
- Training loss divergence (increase instead of decrease)
- NaN values in parameters
- Oscillating loss (too high learning rate)

**Mitigation:**
- Lower learning rate if training becomes unstable
- Compare with standard Adam on small scale first
- Use mixed precision training (FP16 for gradients, 8-bit for states)

---

## Performance Benchmarks (Expected)

| Parameter Count | Standard Adam Memory | 8-bit Adam Memory | Reduction | Performance Impact |
|----------------|---------------------|-------------------|-----------|-------------------|
| 1M | 8 MB | 2 MB | 75% | +5-10% compute time |
| 10M | 80 MB | 20 MB | 75% | +5-10% compute time |
| 100M | 800 MB | 200 MB | 75% | +5-10% compute time |
| 1B | 8 GB | 2 GB | 75% | +5-10% compute time |

**Note:** Small compute overhead comes from quantization/dequantization operations.

---

## Next Steps

1. **Implement QuantizedVector** helper class first
2. **Test QuantizedVector** independently (quantize → dequantize should be close to original)
3. **Implement Adam8BitOptimizer**
4. **Test memory reduction** (measure actual bytes used)
5. **Test numerical accuracy** (compare with standard Adam)
6. **Benchmark performance** (time per iteration)
7. **Integration test** (train a real model)

**Success Criteria:**
- 50-75% memory reduction for optimizer states
- Within 5% accuracy compared to standard Adam
- No NaN/Infinity during training
- Training converges on simple problems

**Good luck!** You're implementing a cutting-edge optimization technique used by state-of-the-art models!
