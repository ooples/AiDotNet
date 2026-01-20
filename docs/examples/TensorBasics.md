# Tensor Basics Guide

This guide demonstrates the fundamentals of working with tensors in AiDotNet.

## Overview

Tensors are the fundamental data structure in AiDotNet. They represent multi-dimensional arrays with support for GPU acceleration and automatic differentiation.

## Creating Tensors

### From Arrays

```csharp
using AiDotNet;
using AiDotNet.Tensors;

// 1D Tensor (Vector)
var vector = new Tensor<float>(new float[] { 1, 2, 3, 4, 5 });
Console.WriteLine($"Vector shape: {string.Join(", ", vector.Shape)}");  // [5]

// 2D Tensor (Matrix)
var matrix = new Tensor<float>(new float[,]
{
    { 1, 2, 3 },
    { 4, 5, 6 }
});
Console.WriteLine($"Matrix shape: {string.Join(", ", matrix.Shape)}");  // [2, 3]

// 3D Tensor
var tensor3d = new Tensor<float>(new[] { 2, 3, 4 });  // Shape: [2, 3, 4]
Console.WriteLine($"3D Tensor elements: {tensor3d.Size}");  // 24
```

### Factory Methods

```csharp
// Zeros
var zeros = Tensor<float>.Zeros(3, 4);  // 3x4 matrix of zeros

// Ones
var ones = Tensor<float>.Ones(2, 3);  // 2x3 matrix of ones

// Random uniform [0, 1)
var random = Tensor<float>.Random(100, 50);  // 100x50 random matrix

// Random normal (mean=0, std=1)
var normal = Tensor<float>.RandomNormal(100, 50);

// Identity matrix
var identity = Tensor<float>.Identity(4);  // 4x4 identity matrix

// Range
var range = Tensor<float>.Arange(0, 10, 1);  // [0, 1, 2, ..., 9]

// Linspace
var linspace = Tensor<float>.Linspace(0, 1, 11);  // 11 evenly spaced values from 0 to 1
```

## Basic Operations

### Element-wise Operations

```csharp
var a = new Tensor<float>(new float[] { 1, 2, 3, 4 });
var b = new Tensor<float>(new float[] { 5, 6, 7, 8 });

// Addition
var sum = a + b;  // or a.Add(b)
Console.WriteLine($"Sum: {string.Join(", ", sum.ToArray())}");  // [6, 8, 10, 12]

// Subtraction
var diff = a - b;  // or a.Subtract(b)

// Multiplication (element-wise)
var product = a * b;  // or a.Multiply(b)

// Division
var quotient = a / b;  // or a.Divide(b)

// Scalar operations
var scaled = a * 2.0f;  // [2, 4, 6, 8]
var offset = a + 10.0f;  // [11, 12, 13, 14]
```

### Matrix Operations

```csharp
var m1 = new Tensor<float>(new float[,]
{
    { 1, 2 },
    { 3, 4 }
});

var m2 = new Tensor<float>(new float[,]
{
    { 5, 6 },
    { 7, 8 }
});

// Matrix multiplication
var matmul = m1.MatMul(m2);
Console.WriteLine($"MatMul result shape: {string.Join(", ", matmul.Shape)}");

// Transpose
var transposed = m1.Transpose();

// Inverse (for square matrices)
var inverse = m1.Inverse();

// Determinant
var det = m1.Determinant();
```

### Reduction Operations

```csharp
var tensor = Tensor<float>.Random(10, 5);

// Sum
var totalSum = tensor.Sum();  // Sum of all elements
var rowSums = tensor.Sum(axis: 1);  // Sum along rows
var colSums = tensor.Sum(axis: 0);  // Sum along columns

// Mean
var mean = tensor.Mean();
var rowMeans = tensor.Mean(axis: 1);

// Min/Max
var min = tensor.Min();
var max = tensor.Max();
var argmax = tensor.ArgMax(axis: 1);  // Indices of max values per row

// Standard deviation
var std = tensor.Std();
var variance = tensor.Var();
```

## Indexing and Slicing

### Single Element Access

```csharp
var matrix = new Tensor<float>(new float[,]
{
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
});

// Get single element
float value = matrix[1, 2];  // 6

// Set single element
matrix[0, 0] = 100;
```

### Slicing

```csharp
// Get a row
var row = matrix[1, ..];  // [4, 5, 6]

// Get a column
var col = matrix[.., 0];  // [1, 4, 7]

// Get a submatrix
var sub = matrix[0..2, 1..3];  // 2x2 submatrix

// Negative indexing (from end)
var lastRow = matrix[^1, ..];  // Last row
var lastCol = matrix[.., ^1];  // Last column
```

## Reshaping

```csharp
var original = Tensor<float>.Arange(0, 12, 1);  // Shape: [12]

// Reshape to 3x4 matrix
var reshaped = original.Reshape(3, 4);

// Reshape to 2x2x3 tensor
var tensor3d = original.Reshape(2, 2, 3);

// Flatten to 1D
var flattened = tensor3d.Flatten();

// Squeeze removes dimensions of size 1
var squeezed = new Tensor<float>(new[] { 1, 3, 1, 4 }).Squeeze();  // Shape: [3, 4]

// Unsqueeze adds a dimension of size 1
var unsqueezed = original.Unsqueeze(0);  // Shape: [1, 12]
```

## Broadcasting

Broadcasting allows operations between tensors of different shapes:

```csharp
var matrix = Tensor<float>.Ones(3, 4);
var rowVector = new Tensor<float>(new float[] { 1, 2, 3, 4 });
var colVector = new Tensor<float>(new float[] { 10, 20, 30 }).Reshape(3, 1);

// Row vector broadcasts across rows
var result1 = matrix + rowVector;  // Each row adds [1, 2, 3, 4]

// Column vector broadcasts across columns
var result2 = matrix + colVector;  // Each column adds [10, 20, 30]

// Scalar broadcasts to all elements
var result3 = matrix + 5.0f;  // Add 5 to all elements
```

## Mathematical Functions

```csharp
var x = new Tensor<float>(new float[] { -2, -1, 0, 1, 2 });

// Trigonometric functions
var sin = x.Sin();
var cos = x.Cos();
var tan = x.Tan();

// Exponential and logarithm
var exp = x.Exp();
var log = x.Abs().Log();  // Log requires positive values

// Power
var squared = x.Pow(2);
var sqrt = x.Abs().Sqrt();

// Absolute value
var abs = x.Abs();

// Clipping
var clipped = x.Clip(-1, 1);  // Values clamped to [-1, 1]
```

## GPU Acceleration

```csharp
// Check GPU availability
if (TensorDevice.IsGpuAvailable)
{
    // Create tensor on GPU
    var gpuTensor = Tensor<float>.Zeros(1000, 1000, device: TensorDevice.GPU);

    // Move existing tensor to GPU
    var cpuTensor = Tensor<float>.Random(1000, 1000);
    var onGpu = cpuTensor.ToDevice(TensorDevice.GPU);

    // Operations automatically use GPU
    var result = onGpu.MatMul(onGpu.Transpose());

    // Move back to CPU for inspection
    var onCpu = result.ToDevice(TensorDevice.CPU);
}
```

## Data Types

```csharp
// Float32 (default)
var floatTensor = new Tensor<float>(new float[] { 1, 2, 3 });

// Float64 (double precision)
var doubleTensor = new Tensor<double>(new double[] { 1, 2, 3 });

// Integer
var intTensor = new Tensor<int>(new int[] { 1, 2, 3 });

// Type conversion
var asDouble = floatTensor.Cast<double>();
```

## Complete Example: Linear Regression

```csharp
using AiDotNet;
using AiDotNet.Tensors;

// Generate synthetic data: y = 2x + 3 + noise
int numSamples = 100;
var x = Tensor<float>.Random(numSamples, 1) * 10;  // Random x values [0, 10)
var noise = Tensor<float>.RandomNormal(numSamples, 1) * 0.5f;
var y = x * 2.0f + 3.0f + noise;  // True relationship with noise

// Add bias column to x
var xWithBias = Tensor<float>.Concatenate(
    x,
    Tensor<float>.Ones(numSamples, 1),
    axis: 1
);

// Solve using normal equations: weights = (X^T X)^-1 X^T y
var xTx = xWithBias.Transpose().MatMul(xWithBias);
var xTy = xWithBias.Transpose().MatMul(y);
var weights = xTx.Inverse().MatMul(xTy);

Console.WriteLine($"Learned weights: [{weights[0, 0]:F4}, {weights[1, 0]:F4}]");
Console.WriteLine("(Expected approximately: [2.0, 3.0])");

// Predict
var yPred = xWithBias.MatMul(weights);

// Calculate R^2 score
var ssRes = (y - yPred).Pow(2).Sum();
var ssTot = (y - y.Mean()).Pow(2).Sum();
var r2 = 1 - ssRes / ssTot;
Console.WriteLine($"R^2 Score: {r2:F4}");
```

## Best Practices

1. **Use GPU for Large Tensors**: Operations on tensors larger than 1000x1000 benefit from GPU acceleration

2. **Preallocate When Possible**: Avoid creating many small temporary tensors in loops

3. **Use In-Place Operations**: When memory is a concern, use in-place variants:
   ```csharp
   tensor.AddInPlace(other);  // Modifies tensor in place
   ```

4. **Batch Operations**: Process data in batches rather than element by element

5. **Check Shapes**: Use shape assertions to catch dimension mismatches early:
   ```csharp
   Debug.Assert(tensor.Shape[0] == expectedBatchSize);
   ```

## Common Issues

### Shape Mismatch

```csharp
// This will throw an exception - shapes don't match
var a = Tensor<float>.Zeros(3, 4);
var b = Tensor<float>.Zeros(4, 3);
// var c = a + b;  // Error!

// Fix: Transpose one of them
var c = a + b.Transpose();  // Now shapes match
```

### Memory Management

```csharp
// For large computations, dispose tensors when done
using (var temp = Tensor<float>.Random(10000, 10000))
{
    var result = temp.Sum();
    // temp is disposed when leaving the block
}
```

## Summary

Tensors in AiDotNet provide:
- Efficient multi-dimensional array operations
- Automatic broadcasting
- GPU acceleration
- NumPy-like syntax and operations
- Support for automatic differentiation

Use tensors as the foundation for all numerical computations in AiDotNet.
