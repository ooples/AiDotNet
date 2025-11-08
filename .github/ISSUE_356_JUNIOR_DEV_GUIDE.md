# Issue #356: Junior Developer Implementation Guide - Core Linear Algebra Data Structures

## Overview
This guide helps you create **unit tests** for the fundamental linear algebra data structures: Matrix, Vector, Tensor, MatrixBase, VectorBase, TensorBase, and Sample. These classes currently have **0% test coverage** despite being critical building blocks of the entire library.

**Goal**: Write comprehensive unit tests to ensure these data structures work correctly.

---

## Understanding the Classes

### Matrix<T> (`src/LinearAlgebra/Matrix.cs`)
A 2D array of numbers arranged in rows and columns.

**Key Properties**:
- `Rows`: Number of rows
- `Columns`: Number of columns
- `this[row, col]`: Access/set elements

**Key Methods**:
- `Add(Matrix<T>)`: Add two matrices
- `Subtract(Matrix<T>)`: Subtract matrices
- `Multiply(Matrix<T>)`: Matrix multiplication
- `Multiply(Vector<T>)`: Matrix-vector multiplication
- `Multiply(T scalar)`: Scalar multiplication
- `Transpose()`: Flip rows and columns
- `Clone()`: Create a copy
- `GetRow(int)`: Extract a row as vector
- `GetColumn(int)`: Extract a column as vector
- `Diagonal()`: Get diagonal elements
- `SubMatrix(...)`: Extract rectangular portion
- `CreateIdentity(int)`: Create identity matrix
- `CreateRandom(rows, cols)`: Create random matrix
- `CreateZeros(rows, cols)`: Create zero matrix
- `CreateOnes(rows, cols)`: Create ones matrix
- `FromRowVectors(...)`: Build from row vectors
- `FromColumnVectors(...)`: Build from column vectors

**Example Usage**:
```csharp
var matrix = new Matrix<double>(3, 3); // 3x3 matrix
matrix[0, 0] = 1.0;
matrix[1, 1] = 2.0;
var transposed = matrix.Transpose();
```

---

### Vector<T> (`src/LinearAlgebra/Vector.cs`)
A 1D array of numbers.

**Key Properties**:
- `Length`: Number of elements
- `this[index]`: Access/set elements

**Key Methods**:
- `Add(Vector<T>)`: Add vectors element-wise
- `Subtract(Vector<T>)`: Subtract vectors
- `Multiply(T scalar)`: Multiply by scalar
- `Divide(T scalar)`: Divide by scalar
- `DotProduct(Vector<T>)`: Dot product
- `Norm()`: Calculate L2 norm (magnitude)
- `Normalize()`: Create unit vector
- `Mean()`: Calculate average
- `Sum()`: Sum all elements
- `Clone()`: Create copy
- `ElementwiseMultiply(Vector<T>)`: Element-wise multiplication
- `ElementwiseDivide(Vector<T>)`: Element-wise division
- `GetSubVector(start, length)`: Extract portion
- `Concatenate(...)`: Join vectors
- `IndexOfMax()`: Find index of maximum value
- `OuterProduct(Vector<T>)`: Create matrix from two vectors
- `CreateRandom(size)`: Create random vector
- `CreateDefault(size, value)`: Fill with value

**Example Usage**:
```csharp
var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
var sum = v1.Add(v2); // [5.0, 7.0, 9.0]
var dot = v1.DotProduct(v2); // 1*4 + 2*5 + 3*6 = 32
```

---

### Tensor<T> (`src/LinearAlgebra/Tensor.cs`)
A multi-dimensional array of numbers.

**Key Properties**:
- `Shape`: Array of dimension sizes
- `Rank`: Number of dimensions
- `Length`: Total number of elements
- `this[indices...]`: Access/set elements

**Key Methods**:
- `Add(Tensor<T>)`: Element-wise addition
- `Subtract(Tensor<T>)`: Element-wise subtraction
- `Multiply(Tensor<T>)`: Matrix multiplication (2D only)
- `Multiply(T scalar)`: Scalar multiplication
- `ElementwiseMultiply(Tensor<T>)`: Element-wise multiplication
- `Transpose()`: Transpose (2D only)
- `Reshape(...)`: Change shape without changing data
- `Slice(...)`: Extract portion
- `ToVector()`: Flatten to 1D vector
- `ToMatrix()`: Convert 2D tensor to Matrix
- `Sum(axes)`: Sum along specific axes
- `Mean()`: Calculate mean of all elements
- `Max()`: Find maximum value and index
- `Stack(tensors[], axis)`: Stack tensors along axis
- `Concatenate(tensors[], axis)`: Join tensors
- `FromVector(Vector<T>)`: Create tensor from vector
- `FromMatrix(Matrix<T>)`: Create tensor from matrix
- `CreateRandom(dimensions)`: Create random tensor

**Example Usage**:
```csharp
// 2x3x4 tensor
var tensor = new Tensor<double>(new[] { 2, 3, 4 });
tensor[0, 1, 2] = 5.0;
var flattened = tensor.ToVector(); // 24 elements
var reshaped = tensor.Reshape(3, 8); // Now 3x8
```

---

### Sample<T> (`src/LinearAlgebra/Sample.cs`)
Represents a training example with features and target.

**Key Properties**:
- `Features`: Vector of input features
- `Target`: Output/label value

**Example Usage**:
```csharp
var features = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
var target = 10.0;
var sample = new Sample<double>(features, target);
```

---

## Phase 1: Matrix Tests

### Test File: `tests/UnitTests/LinearAlgebra/MatrixTests.cs`

**Create the test file structure**:
```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class MatrixTests
{
    // Tests go here
}
```

### AC 1.1: Constructor Tests

Test creating matrices in different ways.

```csharp
[Fact]
public void Constructor_WithDimensions_CreatesMatrixWithCorrectSize()
{
    // Arrange & Act
    var matrix = new Matrix<double>(3, 4);

    // Assert
    Assert.Equal(3, matrix.Rows);
    Assert.Equal(4, matrix.Columns);
}

[Fact]
public void Constructor_With2DArray_CreatesMatrixWithValues()
{
    // Arrange
    var data = new double[,]
    {
        { 1.0, 2.0 },
        { 3.0, 4.0 }
    };

    // Act
    var matrix = new Matrix<double>(data);

    // Assert
    Assert.Equal(2, matrix.Rows);
    Assert.Equal(2, matrix.Columns);
    Assert.Equal(1.0, matrix[0, 0]);
    Assert.Equal(4.0, matrix[1, 1]);
}

[Fact]
public void Constructor_WithJaggedArray_CreatesMatrixWithValues()
{
    // Arrange
    var rows = new[]
    {
        new[] { 1.0, 2.0, 3.0 },
        new[] { 4.0, 5.0, 6.0 }
    };

    // Act
    var matrix = new Matrix<double>(rows);

    // Assert
    Assert.Equal(2, matrix.Rows);
    Assert.Equal(3, matrix.Columns);
    Assert.Equal(5.0, matrix[1, 1]);
}
```

### AC 1.2: Indexer Tests

Test accessing and setting elements.

```csharp
[Fact]
public void Indexer_SetAndGet_WorksCorrectly()
{
    // Arrange
    var matrix = new Matrix<double>(2, 2);

    // Act
    matrix[0, 0] = 1.5;
    matrix[1, 1] = 2.5;

    // Assert
    Assert.Equal(1.5, matrix[0, 0]);
    Assert.Equal(2.5, matrix[1, 1]);
}

[Fact]
public void Indexer_OutOfBounds_ThrowsException()
{
    // Arrange
    var matrix = new Matrix<double>(2, 2);

    // Act & Assert
    Assert.Throws<IndexOutOfRangeException>(() => matrix[3, 0]);
    Assert.Throws<IndexOutOfRangeException>(() => matrix[0, 5]);
}
```

### AC 1.3: Addition Tests

Test matrix addition.

```csharp
[Fact]
public void Add_TwoMatrices_ReturnsCorrectSum()
{
    // Arrange
    var m1 = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
    var m2 = new Matrix<double>(new double[,] { { 5, 6 }, { 7, 8 } });

    // Act
    var result = m1.Add(m2);

    // Assert
    Assert.Equal(6.0, result[0, 0]); // 1 + 5
    Assert.Equal(8.0, result[0, 1]); // 2 + 6
    Assert.Equal(10.0, result[1, 0]); // 3 + 7
    Assert.Equal(12.0, result[1, 1]); // 4 + 8
}

[Fact]
public void Add_DifferentSizes_ThrowsException()
{
    // Arrange
    var m1 = new Matrix<double>(2, 2);
    var m2 = new Matrix<double>(3, 3);

    // Act & Assert
    Assert.Throws<ArgumentException>(() => m1.Add(m2));
}

[Fact]
public void OperatorPlus_TwoMatrices_ReturnsCorrectSum()
{
    // Arrange
    var m1 = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
    var m2 = new Matrix<double>(new double[,] { { 5, 6 }, { 7, 8 } });

    // Act
    var result = m1 + m2;

    // Assert
    Assert.Equal(6.0, result[0, 0]);
    Assert.Equal(12.0, result[1, 1]);
}
```

### AC 1.4: Multiplication Tests

Test different types of multiplication.

```csharp
[Fact]
public void Multiply_TwoMatrices_ReturnsCorrectProduct()
{
    // Arrange
    var m1 = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
    var m2 = new Matrix<double>(new double[,] { { 5, 6 }, { 7, 8 } });

    // Act
    var result = m1.Multiply(m2);

    // Assert
    // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
    Assert.Equal(19.0, result[0, 0]);
    Assert.Equal(22.0, result[0, 1]);
    Assert.Equal(43.0, result[1, 0]);
    Assert.Equal(50.0, result[1, 1]);
}

[Fact]
public void Multiply_MatrixByVector_ReturnsCorrectVector()
{
    // Arrange
    var matrix = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
    var vector = new Vector<double>(new[] { 5.0, 6.0 });

    // Act
    var result = matrix.Multiply(vector);

    // Assert
    // [1,2] * [5] = [1*5+2*6] = [17]
    // [3,4]   [6]   [3*5+4*6]   [39]
    Assert.Equal(17.0, result[0]);
    Assert.Equal(39.0, result[1]);
}

[Fact]
public void Multiply_MatrixByScalar_ReturnsScaledMatrix()
{
    // Arrange
    var matrix = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });

    // Act
    var result = matrix.Multiply(2.0);

    // Assert
    Assert.Equal(2.0, result[0, 0]);
    Assert.Equal(4.0, result[0, 1]);
    Assert.Equal(6.0, result[1, 0]);
    Assert.Equal(8.0, result[1, 1]);
}

[Fact]
public void Multiply_IncompatibleDimensions_ThrowsException()
{
    // Arrange
    var m1 = new Matrix<double>(2, 3);
    var m2 = new Matrix<double>(2, 2); // Should be 3xN

    // Act & Assert
    Assert.Throws<ArgumentException>(() => m1.Multiply(m2));
}
```

### AC 1.5: Transpose Tests

```csharp
[Fact]
public void Transpose_Matrix_FlipsRowsAndColumns()
{
    // Arrange
    var matrix = new Matrix<double>(new double[,]
    {
        { 1, 2, 3 },
        { 4, 5, 6 }
    });

    // Act
    var transposed = matrix.Transpose();

    // Assert
    Assert.Equal(3, transposed.Rows);
    Assert.Equal(2, transposed.Columns);
    Assert.Equal(1.0, transposed[0, 0]);
    Assert.Equal(4.0, transposed[0, 1]);
    Assert.Equal(2.0, transposed[1, 0]);
    Assert.Equal(5.0, transposed[1, 1]);
    Assert.Equal(6.0, transposed[2, 1]);
}

[Fact]
public void Transpose_SquareMatrix_PreservesDiagonal()
{
    // Arrange
    var matrix = new Matrix<double>(new double[,]
    {
        { 1, 2, 3 },
        { 4, 5, 6 },
        { 7, 8, 9 }
    });

    // Act
    var transposed = matrix.Transpose();

    // Assert
    Assert.Equal(1.0, transposed[0, 0]);
    Assert.Equal(5.0, transposed[1, 1]);
    Assert.Equal(9.0, transposed[2, 2]);
}
```

### AC 1.6: Static Factory Methods Tests

```csharp
[Fact]
public void CreateIdentity_ReturnsIdentityMatrix()
{
    // Act
    var identity = Matrix<double>.CreateIdentity(3);

    // Assert
    Assert.Equal(1.0, identity[0, 0]);
    Assert.Equal(1.0, identity[1, 1]);
    Assert.Equal(1.0, identity[2, 2]);
    Assert.Equal(0.0, identity[0, 1]);
    Assert.Equal(0.0, identity[1, 0]);
}

[Fact]
public void CreateZeros_ReturnsZeroMatrix()
{
    // Act
    var zeros = Matrix<double>.CreateZeros(2, 3);

    // Assert
    Assert.Equal(2, zeros.Rows);
    Assert.Equal(3, zeros.Columns);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            Assert.Equal(0.0, zeros[i, j]);
}

[Fact]
public void CreateOnes_ReturnsOnesMatrix()
{
    // Act
    var ones = Matrix<double>.CreateOnes(2, 2);

    // Assert
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            Assert.Equal(1.0, ones[i, j]);
}

[Fact]
public void CreateRandom_ReturnsMatrixWithRandomValues()
{
    // Act
    var random = Matrix<double>.CreateRandom(3, 3);

    // Assert
    Assert.Equal(3, random.Rows);
    Assert.Equal(3, random.Columns);

    // Check that values are different (not all zeros)
    bool hasNonZero = false;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (random[i, j] != 0.0)
                hasNonZero = true;

    Assert.True(hasNonZero);
}
```

### AC 1.7: Row/Column Operations Tests

```csharp
[Fact]
public void GetRow_ReturnsCorrectVector()
{
    // Arrange
    var matrix = new Matrix<double>(new double[,]
    {
        { 1, 2, 3 },
        { 4, 5, 6 }
    });

    // Act
    var row = matrix.GetRow(1);

    // Assert
    Assert.Equal(3, row.Length);
    Assert.Equal(4.0, row[0]);
    Assert.Equal(5.0, row[1]);
    Assert.Equal(6.0, row[2]);
}

[Fact]
public void GetColumn_ReturnsCorrectVector()
{
    // Arrange
    var matrix = new Matrix<double>(new double[,]
    {
        { 1, 2, 3 },
        { 4, 5, 6 }
    });

    // Act
    var column = matrix.GetColumn(1);

    // Assert
    Assert.Equal(2, column.Length);
    Assert.Equal(2.0, column[0]);
    Assert.Equal(5.0, column[1]);
}

[Fact]
public void SetRow_UpdatesRowValues()
{
    // Arrange
    var matrix = new Matrix<double>(2, 3);
    var newRow = new Vector<double>(new[] { 7.0, 8.0, 9.0 });

    // Act
    matrix.SetRow(1, newRow);

    // Assert
    Assert.Equal(7.0, matrix[1, 0]);
    Assert.Equal(8.0, matrix[1, 1]);
    Assert.Equal(9.0, matrix[1, 2]);
}

[Fact]
public void SetColumn_UpdatesColumnValues()
{
    // Arrange
    var matrix = new Matrix<double>(3, 2);
    var newColumn = new Vector<double>(new[] { 7.0, 8.0, 9.0 });

    // Act
    matrix.SetColumn(1, newColumn);

    // Assert
    Assert.Equal(7.0, matrix[0, 1]);
    Assert.Equal(8.0, matrix[1, 1]);
    Assert.Equal(9.0, matrix[2, 1]);
}
```

---

## Phase 2: Vector Tests

### Test File: `tests/UnitTests/LinearAlgebra/VectorTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class VectorTests
{
    [Fact]
    public void Constructor_WithLength_CreatesVectorWithCorrectSize()
    {
        // Act
        var vector = new Vector<double>(5);

        // Assert
        Assert.Equal(5, vector.Length);
    }

    [Fact]
    public void Constructor_WithArray_CreatesVectorWithValues()
    {
        // Arrange
        var values = new[] { 1.0, 2.0, 3.0 };

        // Act
        var vector = new Vector<double>(values);

        // Assert
        Assert.Equal(3, vector.Length);
        Assert.Equal(1.0, vector[0]);
        Assert.Equal(2.0, vector[1]);
        Assert.Equal(3.0, vector[2]);
    }

    [Fact]
    public void Add_TwoVectors_ReturnsCorrectSum()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = v1.Add(v2);

        // Assert
        Assert.Equal(5.0, result[0]);
        Assert.Equal(7.0, result[1]);
        Assert.Equal(9.0, result[2]);
    }

    [Fact]
    public void DotProduct_TwoVectors_ReturnsCorrectValue()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = v1.DotProduct(v2);

        // Assert
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Assert.Equal(32.0, result);
    }

    [Fact]
    public void Norm_Vector_ReturnsCorrectMagnitude()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var norm = vector.Norm();

        // Assert
        // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
        Assert.Equal(5.0, norm, precision: 10);
    }

    [Fact]
    public void Normalize_Vector_ReturnsUnitVector()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var normalized = vector.Normalize();

        // Assert
        Assert.Equal(0.6, normalized[0], precision: 10); // 3/5
        Assert.Equal(0.8, normalized[1], precision: 10); // 4/5
        Assert.Equal(1.0, normalized.Norm(), precision: 10);
    }

    [Fact]
    public void Mean_Vector_ReturnsCorrectAverage()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        // Act
        var mean = vector.Mean();

        // Assert
        Assert.Equal(4.0, mean);
    }

    [Fact]
    public void ElementwiseMultiply_TwoVectors_ReturnsCorrectProduct()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
        var v2 = new Vector<double>(new[] { 5.0, 6.0, 7.0 });

        // Act
        var result = v1.ElementwiseMultiply(v2);

        // Assert
        Assert.Equal(10.0, result[0]); // 2*5
        Assert.Equal(18.0, result[1]); // 3*6
        Assert.Equal(28.0, result[2]); // 4*7
    }

    [Fact]
    public void Concatenate_TwoVectors_ReturnsJoinedVector()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var result = Vector<double>.Concatenate(v1, v2);

        // Assert
        Assert.Equal(4, result.Length);
        Assert.Equal(1.0, result[0]);
        Assert.Equal(2.0, result[1]);
        Assert.Equal(3.0, result[2]);
        Assert.Equal(4.0, result[3]);
    }
}
```

---

## Phase 3: Tensor Tests

### Test File: `tests/UnitTests/LinearAlgebra/TensorTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class TensorTests
{
    [Fact]
    public void Constructor_WithDimensions_CreatesTensorWithCorrectShape()
    {
        // Act
        var tensor = new Tensor<double>(new[] { 2, 3, 4 });

        // Assert
        Assert.Equal(3, tensor.Rank);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(4, tensor.Shape[2]);
        Assert.Equal(24, tensor.Length);
    }

    [Fact]
    public void Indexer_SetAndGet_WorksCorrectly()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2, 2, 2 });

        // Act
        tensor[0, 1, 1] = 5.0;

        // Assert
        Assert.Equal(5.0, tensor[0, 1, 1]);
    }

    [Fact]
    public void Add_TwoTensors_ReturnsCorrectSum()
    {
        // Arrange
        var t1 = new Tensor<double>(new[] { 2, 2 });
        var t2 = new Tensor<double>(new[] { 2, 2 });
        t1[0, 0] = 1.0; t1[0, 1] = 2.0;
        t2[0, 0] = 3.0; t2[0, 1] = 4.0;

        // Act
        var result = t1.Add(t2);

        // Assert
        Assert.Equal(4.0, result[0, 0]);
        Assert.Equal(6.0, result[0, 1]);
    }

    [Fact]
    public void Reshape_Tensor_ChangesShapeCorrectly()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                tensor[i, j] = i * 3 + j;

        // Act
        var reshaped = tensor.Reshape(3, 2);

        // Assert
        Assert.Equal(3, reshaped.Shape[0]);
        Assert.Equal(2, reshaped.Shape[1]);
        Assert.Equal(6, reshaped.Length);
    }

    [Fact]
    public void ToVector_Tensor_FlattensCorrectly()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2, 3 });
        int value = 0;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                tensor[i, j] = value++;

        // Act
        var vector = tensor.ToVector();

        // Assert
        Assert.Equal(6, vector.Length);
        for (int i = 0; i < 6; i++)
            Assert.Equal((double)i, vector[i]);
    }

    [Fact]
    public void Slice_Tensor_ExtractsCorrectPortion()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3, 4 });

        // Act
        var slice = tensor.Slice(1);

        // Assert
        Assert.Equal(1, slice.Rank);
        Assert.Equal(4, slice.Length);
    }

    [Fact]
    public void FromMatrix_ConvertsMatrixToTensor()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act
        var tensor = Tensor<double>.FromMatrix(matrix);

        // Assert
        Assert.Equal(2, tensor.Rank);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(2, tensor.Shape[1]);
        Assert.Equal(1.0, tensor[0, 0]);
        Assert.Equal(4.0, tensor[1, 1]);
    }
}
```

---

## Phase 4: Sample Tests

### Test File: `tests/UnitTests/LinearAlgebra/SampleTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class SampleTests
{
    [Fact]
    public void Constructor_WithFeaturesAndTarget_CreatesSampleCorrectly()
    {
        // Arrange
        var features = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var target = 10.0;

        // Act
        var sample = new Sample<double>(features, target);

        // Assert
        Assert.Equal(3, sample.Features.Length);
        Assert.Equal(1.0, sample.Features[0]);
        Assert.Equal(10.0, sample.Target);
    }

    [Fact]
    public void Properties_CanBeModified()
    {
        // Arrange
        var sample = new Sample<double>(
            new Vector<double>(new[] { 1.0, 2.0 }),
            5.0
        );

        // Act
        sample.Features = new Vector<double>(new[] { 3.0, 4.0 });
        sample.Target = 7.0;

        // Assert
        Assert.Equal(3.0, sample.Features[0]);
        Assert.Equal(7.0, sample.Target);
    }
}
```

---

## Common Testing Patterns

### Edge Cases to Test

1. **Empty/Zero-sized structures**
   ```csharp
   var emptyMatrix = Matrix<double>.Empty();
   var emptyVector = Vector<double>.Empty();
   ```

2. **Single element structures**
   ```csharp
   var singleElement = new Matrix<double>(1, 1);
   ```

3. **Large structures**
   ```csharp
   var large = new Matrix<double>(1000, 1000);
   ```

4. **Boundary conditions**
   ```csharp
   // First and last elements
   matrix[0, 0] = 1.0;
   matrix[rows-1, cols-1] = 1.0;
   ```

### Type Testing

Test with different numeric types:
```csharp
[Fact]
public void Matrix_WorksWithFloat()
{
    var matrix = new Matrix<float>(2, 2);
    matrix[0, 0] = 1.5f;
    Assert.Equal(1.5f, matrix[0, 0]);
}

[Fact]
public void Matrix_WorksWithInt()
{
    var matrix = new Matrix<int>(2, 2);
    matrix[0, 0] = 5;
    Assert.Equal(5, matrix[0, 0]);
}
```

---

## Running Tests

```bash
# Run all LinearAlgebra tests
dotnet test --filter "FullyQualifiedName~LinearAlgebra"

# Run specific test class
dotnet test --filter "FullyQualifiedName~MatrixTests"

# Run with coverage
dotnet test /p:CollectCoverage=true
```

---

## Success Criteria

- [ ] Matrix tests cover: constructors, operations, indexing, row/column ops
- [ ] Vector tests cover: constructors, operations, norms, element-wise ops
- [ ] Tensor tests cover: multi-dimensional ops, reshaping, slicing
- [ ] Sample tests cover: construction and property access
- [ ] All tests pass with green checkmarks
- [ ] Code coverage increases from 0% to >80%
- [ ] Edge cases tested (empty, single element, large)
- [ ] Multiple numeric types tested (double, float, int)

---

## Common Pitfalls

1. **Don't use `default(T)` or `default!`** - Use proper initialization
2. **Don't assume `double` type** - Tests should work with generic `T`
3. **Don't forget precision in floating-point comparisons** - Use `Assert.Equal(expected, actual, precision: 10)`
4. **Do test exception cases** - Invalid indices, dimension mismatches
5. **Do test operators** - `+`, `-`, `*`, `/` operators
6. **Do verify immutability where appropriate** - Operations create new instances

Start with Matrix tests, then Vector, then Tensor, then Sample. Build incrementally!
