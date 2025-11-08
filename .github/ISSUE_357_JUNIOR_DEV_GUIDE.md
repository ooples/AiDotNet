# Issue #357: Junior Developer Implementation Guide - Advanced Linear Algebra Structures

## Overview
This guide helps you create **unit tests** for advanced linear algebra structures: Complex, FFT (if implemented), DecisionTreeNode, ConditionalInferenceTreeNode, ExpressionTree, ConfusionMatrix, and NodeModification. These classes currently have **0% test coverage**.

**Goal**: Write comprehensive unit tests to ensure these data structures work correctly.

---

## Understanding the Classes

### Complex<T> (`src/LinearAlgebra/Complex.cs`)
Represents complex numbers with real and imaginary parts (a + bi).

**Key Properties**:
- `Real`: The real part
- `Imaginary`: The imaginary part (coefficient of i)
- `Magnitude`: Distance from origin (|z| = sqrt(real² + imag²))
- `Phase`: Angle in complex plane (arg(z))
- `Conjugate`: Complex conjugate (real - imag*i)

**Key Methods**:
- `Add(Complex<T>)`: Add two complex numbers
- `Subtract(Complex<T>)`: Subtract complex numbers
- `Multiply(Complex<T>)`: Multiply complex numbers
- `Divide(Complex<T>)`: Divide complex numbers
- `Pow(int)`: Raise to integer power
- `Sqrt()`: Square root of complex number
- `Exp()`: e^z for complex z
- `Sin()`, `Cos()`, `Tan()`: Trigonometric functions
- `ToString()`: Format as "a + bi"

**Mathematical Background**:
- Complex multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
- Complex division: `(a+bi)/(c+di) = ((ac+bd)/(c²+d²)) + ((bc-ad)/(c²+d²))i`
- Magnitude: `|a+bi| = sqrt(a² + b²)`

**Example Usage**:
```csharp
var z1 = new Complex<double>(3.0, 4.0); // 3 + 4i
var z2 = new Complex<double>(1.0, 2.0); // 1 + 2i
var sum = z1.Add(z2); // 4 + 6i
var magnitude = z1.Magnitude; // 5.0 (since 3² + 4² = 25)
```

---

### DecisionTreeNode<T> (`src/LinearAlgebra/DecisionTreeNode.cs`)
Represents a node in a decision tree structure.

**Key Properties**:
- `FeatureIndex`: Which feature to split on
- `Threshold`: Split value
- `SplitValue`: Actual value from data used for split
- `Prediction`: Predicted value (for leaf nodes)
- `Left`: Left child node (< threshold)
- `Right`: Right child node (>= threshold)
- `IsLeaf`: True if this is a terminal node
- `Samples`: Training samples at this node
- `LeftSampleCount`: Count of samples going left
- `RightSampleCount`: Count of samples going right
- `SampleValues`: Target values of samples
- `LinearModel`: Optional linear regression model
- `Predictions`: Vector of predictions
- `SumSquaredError`: SSE for predictions

**Example Usage**:
```csharp
var root = new DecisionTreeNode<double>
{
    FeatureIndex = 0,
    Threshold = 5.0,
    IsLeaf = false
};

var leftLeaf = new DecisionTreeNode<double>
{
    IsLeaf = true,
    Prediction = 10.0
};

root.Left = leftLeaf;
```

---

### ConfusionMatrix<T> (`src/LinearAlgebra/ConfusionMatrix.cs`)
Evaluates classification model performance.

**Key Properties (Binary Classification)**:
- `TruePositives`: Correctly predicted positive (TP)
- `TrueNegatives`: Correctly predicted negative (TN)
- `FalsePositives`: Incorrectly predicted positive (FP)
- `FalseNegatives`: Incorrectly predicted negative (FN)
- `ClassCount`: Number of classes

**Key Methods**:
- `Increment(predicted, actual)`: Record a prediction
- `Accuracy()`: (TP + TN) / Total
- `Precision()`: TP / (TP + FP)
- `Recall()`: TP / (TP + FN)
- `F1Score()`: Harmonic mean of precision and recall
- `GetClassAccuracy(classIndex)`: Accuracy for specific class
- `GetConfidence()`: Overall confidence measure

**Example Usage**:
```csharp
// Binary classification
var cm = new ConfusionMatrix<int>(
    truePositives: 50,
    trueNegatives: 40,
    falsePositives: 5,
    falseNegatives: 5
);

double accuracy = cm.Accuracy(); // (50+40)/100 = 0.90
double precision = cm.Precision(); // 50/(50+5) = 0.909
```

---

### ConditionalInferenceTreeNode<T> (`src/LinearAlgebra/ConditionalInferenceTreeNode.cs`)
Extended decision tree node with statistical testing.

**Key Properties** (extends DecisionTreeNode):
- `PValue`: Statistical significance of split
- `TestStatistic`: Test statistic value
- `VariableImportance`: Importance score of feature
- `Depth`: Depth in tree (distance from root)

**Example Usage**:
```csharp
var node = new ConditionalInferenceTreeNode<double>
{
    FeatureIndex = 2,
    Threshold = 3.5,
    PValue = 0.01, // Statistically significant
    TestStatistic = 15.3,
    Depth = 2
};
```

---

### ExpressionTree<T> (`src/LinearAlgebra/ExpressionTree.cs`)
Represents mathematical expressions as a tree structure.

**Key Properties**:
- `Value`: Node value (if leaf)
- `Operator`: Operation type (+, -, *, /, etc.)
- `Left`: Left operand
- `Right`: Right operand
- `IsLeaf`: True if terminal node
- `Variable`: Variable name (if variable node)

**Key Methods**:
- `Evaluate(variables)`: Calculate expression value
- `Derivative(variable)`: Symbolic differentiation
- `Simplify()`: Simplify expression
- `ToString()`: Convert to string representation

**Example Usage**:
```csharp
// Representing: x² + 2x + 1
var tree = new ExpressionTree<double>
{
    Operator = "+",
    Left = new ExpressionTree<double> // x²
    {
        Operator = "^",
        Left = new ExpressionTree<double> { Variable = "x" },
        Right = new ExpressionTree<double> { Value = 2.0 }
    },
    Right = new ExpressionTree<double> // 2x + 1
    {
        Operator = "+",
        Left = new ExpressionTree<double>
        {
            Operator = "*",
            Left = new ExpressionTree<double> { Value = 2.0 },
            Right = new ExpressionTree<double> { Variable = "x" }
        },
        Right = new ExpressionTree<double> { Value = 1.0 }
    }
};

var result = tree.Evaluate(new Dictionary<string, double> { ["x"] = 3.0 }); // 9 + 6 + 1 = 16
```

---

## Phase 1: Complex Number Tests

### Test File: `tests/UnitTests/LinearAlgebra/ComplexTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class ComplexTests
{
    [Fact]
    public void Constructor_WithRealAndImaginary_CreatesComplexNumber()
    {
        // Act
        var complex = new Complex<double>(3.0, 4.0);

        // Assert
        Assert.Equal(3.0, complex.Real);
        Assert.Equal(4.0, complex.Imaginary);
    }

    [Fact]
    public void Magnitude_ComplexNumber_ReturnsCorrectValue()
    {
        // Arrange
        var complex = new Complex<double>(3.0, 4.0);

        // Act
        var magnitude = complex.Magnitude;

        // Assert
        // |3+4i| = sqrt(9 + 16) = sqrt(25) = 5
        Assert.Equal(5.0, magnitude, precision: 10);
    }

    [Fact]
    public void Add_TwoComplexNumbers_ReturnsCorrectSum()
    {
        // Arrange
        var z1 = new Complex<double>(1.0, 2.0); // 1 + 2i
        var z2 = new Complex<double>(3.0, 4.0); // 3 + 4i

        // Act
        var result = z1.Add(z2);

        // Assert
        // (1+2i) + (3+4i) = 4 + 6i
        Assert.Equal(4.0, result.Real);
        Assert.Equal(6.0, result.Imaginary);
    }

    [Fact]
    public void Multiply_TwoComplexNumbers_ReturnsCorrectProduct()
    {
        // Arrange
        var z1 = new Complex<double>(1.0, 2.0); // 1 + 2i
        var z2 = new Complex<double>(3.0, 4.0); // 3 + 4i

        // Act
        var result = z1.Multiply(z2);

        // Assert
        // (1+2i)(3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        Assert.Equal(-5.0, result.Real);
        Assert.Equal(10.0, result.Imaginary);
    }

    [Fact]
    public void Conjugate_ComplexNumber_ReturnsConjugate()
    {
        // Arrange
        var complex = new Complex<double>(3.0, 4.0);

        // Act
        var conjugate = complex.Conjugate;

        // Assert
        // Conjugate of 3+4i is 3-4i
        Assert.Equal(3.0, conjugate.Real);
        Assert.Equal(-4.0, conjugate.Imaginary);
    }

    [Fact]
    public void Divide_TwoComplexNumbers_ReturnsCorrectQuotient()
    {
        // Arrange
        var z1 = new Complex<double>(1.0, 2.0); // 1 + 2i
        var z2 = new Complex<double>(3.0, 4.0); // 3 + 4i

        // Act
        var result = z1.Divide(z2);

        // Assert
        // (1+2i)/(3+4i) = ((1*3+2*4)/(9+16)) + ((2*3-1*4)/(9+16))i = 11/25 + 2/25i
        Assert.Equal(11.0 / 25.0, result.Real, precision: 10);
        Assert.Equal(2.0 / 25.0, result.Imaginary, precision: 10);
    }

    [Fact]
    public void Phase_ComplexNumber_ReturnsCorrectAngle()
    {
        // Arrange
        var complex = new Complex<double>(1.0, 1.0);

        // Act
        var phase = complex.Phase;

        // Assert
        // Phase of 1+i is 45 degrees = π/4 radians
        Assert.Equal(Math.PI / 4.0, phase, precision: 10);
    }

    [Fact]
    public void Sqrt_ComplexNumber_ReturnsCorrectSquareRoot()
    {
        // Arrange
        var complex = new Complex<double>(0.0, 4.0); // 4i

        // Act
        var sqrt = complex.Sqrt();

        // Assert
        // sqrt(4i) ≈ 1.414 + 1.414i
        Assert.Equal(Math.Sqrt(2), sqrt.Real, precision: 10);
        Assert.Equal(Math.Sqrt(2), sqrt.Imaginary, precision: 10);
    }

    [Fact]
    public void ToString_ComplexNumber_FormatsCorrectly()
    {
        // Arrange
        var complex = new Complex<double>(3.0, 4.0);

        // Act
        var str = complex.ToString();

        // Assert
        Assert.Contains("3", str);
        Assert.Contains("4", str);
        Assert.Contains("i", str);
    }
}
```

---

## Phase 2: ConfusionMatrix Tests

### Test File: `tests/UnitTests/LinearAlgebra/ConfusionMatrixTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class ConfusionMatrixTests
{
    [Fact]
    public void Constructor_BinaryClassification_CreatesMatrix()
    {
        // Act
        var cm = new ConfusionMatrix<int>(50, 40, 5, 5);

        // Assert
        Assert.Equal(50, cm.TruePositives);
        Assert.Equal(40, cm.TrueNegatives);
        Assert.Equal(5, cm.FalsePositives);
        Assert.Equal(5, cm.FalseNegatives);
    }

    [Fact]
    public void Accuracy_BinaryClassification_ReturnsCorrectValue()
    {
        // Arrange
        var cm = new ConfusionMatrix<int>(50, 40, 5, 5);

        // Act
        var accuracy = cm.Accuracy();

        // Assert
        // Accuracy = (TP + TN) / Total = (50 + 40) / 100 = 0.90
        Assert.Equal(0.90, accuracy, precision: 10);
    }

    [Fact]
    public void Precision_BinaryClassification_ReturnsCorrectValue()
    {
        // Arrange
        var cm = new ConfusionMatrix<int>(50, 40, 5, 5);

        // Act
        var precision = cm.Precision();

        // Assert
        // Precision = TP / (TP + FP) = 50 / (50 + 5) = 0.909
        Assert.Equal(50.0 / 55.0, precision, precision: 10);
    }

    [Fact]
    public void Recall_BinaryClassification_ReturnsCorrectValue()
    {
        // Arrange
        var cm = new ConfusionMatrix<int>(50, 40, 5, 5);

        // Act
        var recall = cm.Recall();

        // Assert
        // Recall = TP / (TP + FN) = 50 / (50 + 5) = 0.909
        Assert.Equal(50.0 / 55.0, recall, precision: 10);
    }

    [Fact]
    public void F1Score_BinaryClassification_ReturnsCorrectValue()
    {
        // Arrange
        var cm = new ConfusionMatrix<int>(50, 40, 5, 5);

        // Act
        var f1 = cm.F1Score();

        // Assert
        // F1 = 2 * (Precision * Recall) / (Precision + Recall)
        double precision = 50.0 / 55.0;
        double recall = 50.0 / 55.0;
        double expected = 2 * (precision * recall) / (precision + recall);
        Assert.Equal(expected, f1, precision: 10);
    }

    [Fact]
    public void Constructor_MultiClass_CreatesNxNMatrix()
    {
        // Act
        var cm = new ConfusionMatrix<int>(3); // 3 classes

        // Assert
        Assert.Equal(3, cm.Rows);
        Assert.Equal(3, cm.Columns);
        Assert.Equal(3, cm.ClassCount);
    }

    [Fact]
    public void Increment_MultiClass_UpdatesCorrectCell()
    {
        // Arrange
        var cm = new ConfusionMatrix<int>(3);

        // Act
        cm.Increment(0, 0); // Correctly predicted class 0
        cm.Increment(1, 1); // Correctly predicted class 1
        cm.Increment(0, 1); // Predicted 0, actually 1 (error)

        // Assert
        Assert.Equal(1, cm[0, 0]);
        Assert.Equal(1, cm[1, 1]);
        Assert.Equal(1, cm[0, 1]);
    }

    [Fact]
    public void GetClassAccuracy_MultiClass_ReturnsCorrectValue()
    {
        // Arrange
        var cm = new ConfusionMatrix<int>(2);
        cm[0, 0] = 50; // TP for class 0
        cm[1, 1] = 40; // TN for class 0
        cm[0, 1] = 5;  // FN for class 0
        cm[1, 0] = 5;  // FP for class 0

        // Act
        var accuracy = cm.GetClassAccuracy(0);

        // Assert
        // Accuracy for class 0 = (TP + TN) / Total
        Assert.InRange(accuracy, 0.0, 1.0);
    }

    [Fact]
    public void Constructor_DimensionLessThan2_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ConfusionMatrix<int>(1));
    }
}
```

---

## Phase 3: DecisionTreeNode Tests

### Test File: `tests/UnitTests/LinearAlgebra/DecisionTreeNodeTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class DecisionTreeNodeTests
{
    [Fact]
    public void Constructor_Default_CreatesEmptyNode()
    {
        // Act
        var node = new DecisionTreeNode<double>();

        // Assert
        Assert.NotNull(node);
        Assert.NotNull(node.Samples);
        Assert.NotNull(node.SampleValues);
        Assert.Empty(node.Samples);
        Assert.Empty(node.SampleValues);
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        // Act
        var node = new DecisionTreeNode<double>
        {
            FeatureIndex = 2,
            Threshold = 5.0,
            Prediction = 10.0,
            IsLeaf = true
        };

        // Assert
        Assert.Equal(2, node.FeatureIndex);
        Assert.Equal(5.0, node.Threshold);
        Assert.Equal(10.0, node.Prediction);
        Assert.True(node.IsLeaf);
    }

    [Fact]
    public void LeftAndRight_CanBeSet()
    {
        // Arrange
        var root = new DecisionTreeNode<double>();
        var left = new DecisionTreeNode<double> { Prediction = 5.0 };
        var right = new DecisionTreeNode<double> { Prediction = 15.0 };

        // Act
        root.Left = left;
        root.Right = right;

        // Assert
        Assert.NotNull(root.Left);
        Assert.NotNull(root.Right);
        Assert.Equal(5.0, root.Left.Prediction);
        Assert.Equal(15.0, root.Right.Prediction);
    }

    [Fact]
    public void Samples_CanBeAdded()
    {
        // Arrange
        var node = new DecisionTreeNode<double>();
        var sample = new Sample<double>(
            new Vector<double>(new[] { 1.0, 2.0, 3.0 }),
            10.0
        );

        // Act
        node.Samples.Add(sample);

        // Assert
        Assert.Single(node.Samples);
        Assert.Equal(10.0, node.Samples[0].Target);
    }

    [Fact]
    public void SampleCounts_CanBeSet()
    {
        // Act
        var node = new DecisionTreeNode<double>
        {
            LeftSampleCount = 50,
            RightSampleCount = 30
        };

        // Assert
        Assert.Equal(50, node.LeftSampleCount);
        Assert.Equal(30, node.RightSampleCount);
    }

    [Fact]
    public void TreeStructure_MultiLevel_CanBeBuilt()
    {
        // Arrange & Act
        var root = new DecisionTreeNode<double>
        {
            FeatureIndex = 0,
            Threshold = 10.0,
            IsLeaf = false
        };

        root.Left = new DecisionTreeNode<double>
        {
            FeatureIndex = 1,
            Threshold = 5.0,
            IsLeaf = false
        };

        root.Right = new DecisionTreeNode<double>
        {
            IsLeaf = true,
            Prediction = 20.0
        };

        root.Left.Left = new DecisionTreeNode<double>
        {
            IsLeaf = true,
            Prediction = 5.0
        };

        root.Left.Right = new DecisionTreeNode<double>
        {
            IsLeaf = true,
            Prediction = 10.0
        };

        // Assert
        Assert.False(root.IsLeaf);
        Assert.False(root.Left.IsLeaf);
        Assert.True(root.Right.IsLeaf);
        Assert.True(root.Left.Left.IsLeaf);
        Assert.Equal(20.0, root.Right.Prediction);
        Assert.Equal(5.0, root.Left.Left.Prediction);
    }
}
```

---

## Phase 4: ConditionalInferenceTreeNode Tests

### Test File: `tests/UnitTests/LinearAlgebra/ConditionalInferenceTreeNodeTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class ConditionalInferenceTreeNodeTests
{
    [Fact]
    public void Constructor_Default_CreatesNode()
    {
        // Act
        var node = new ConditionalInferenceTreeNode<double>();

        // Assert
        Assert.NotNull(node);
    }

    [Fact]
    public void StatisticalProperties_CanBeSet()
    {
        // Act
        var node = new ConditionalInferenceTreeNode<double>
        {
            PValue = 0.01,
            TestStatistic = 15.3,
            VariableImportance = 0.85,
            Depth = 2
        };

        // Assert
        Assert.Equal(0.01, node.PValue);
        Assert.Equal(15.3, node.TestStatistic);
        Assert.Equal(0.85, node.VariableImportance);
        Assert.Equal(2, node.Depth);
    }

    [Fact]
    public void InheritsFrom_DecisionTreeNode()
    {
        // Act
        var node = new ConditionalInferenceTreeNode<double>
        {
            FeatureIndex = 1,
            Threshold = 5.0,
            IsLeaf = true,
            Prediction = 10.0
        };

        // Assert - Should have DecisionTreeNode properties
        Assert.Equal(1, node.FeatureIndex);
        Assert.Equal(5.0, node.Threshold);
        Assert.True(node.IsLeaf);
        Assert.Equal(10.0, node.Prediction);
    }

    [Fact]
    public void PValue_Significance_CanBeEvaluated()
    {
        // Arrange
        var significantNode = new ConditionalInferenceTreeNode<double>
        {
            PValue = 0.001 // Highly significant
        };

        var notSignificantNode = new ConditionalInferenceTreeNode<double>
        {
            PValue = 0.5 // Not significant
        };

        // Assert
        Assert.True(significantNode.PValue < 0.05);
        Assert.False(notSignificantNode.PValue < 0.05);
    }
}
```

---

## Phase 5: ExpressionTree Tests

### Test File: `tests/UnitTests/LinearAlgebra/ExpressionTreeTests.cs`

```csharp
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LinearAlgebra;

public class ExpressionTreeTests
{
    [Fact]
    public void Constructor_LeafNode_CreatesConstant()
    {
        // Act
        var leaf = new ExpressionTree<double>
        {
            IsLeaf = true,
            Value = 5.0
        };

        // Assert
        Assert.True(leaf.IsLeaf);
        Assert.Equal(5.0, leaf.Value);
    }

    [Fact]
    public void Constructor_VariableNode_CreatesVariable()
    {
        // Act
        var variable = new ExpressionTree<double>
        {
            IsLeaf = true,
            Variable = "x"
        };

        // Assert
        Assert.True(variable.IsLeaf);
        Assert.Equal("x", variable.Variable);
    }

    [Fact]
    public void Constructor_OperatorNode_HasChildren()
    {
        // Act
        var expr = new ExpressionTree<double>
        {
            Operator = "+",
            Left = new ExpressionTree<double> { Value = 3.0, IsLeaf = true },
            Right = new ExpressionTree<double> { Value = 4.0, IsLeaf = true }
        };

        // Assert
        Assert.Equal("+", expr.Operator);
        Assert.NotNull(expr.Left);
        Assert.NotNull(expr.Right);
        Assert.Equal(3.0, expr.Left.Value);
        Assert.Equal(4.0, expr.Right.Value);
    }

    [Fact]
    public void Evaluate_SimpleAddition_ReturnsCorrectValue()
    {
        // Arrange: 3 + 4
        var expr = new ExpressionTree<double>
        {
            Operator = "+",
            Left = new ExpressionTree<double> { Value = 3.0, IsLeaf = true },
            Right = new ExpressionTree<double> { Value = 4.0, IsLeaf = true }
        };

        // Act
        var result = expr.Evaluate(new Dictionary<string, double>());

        // Assert
        Assert.Equal(7.0, result);
    }

    [Fact]
    public void Evaluate_WithVariable_ReturnsCorrectValue()
    {
        // Arrange: x + 5
        var expr = new ExpressionTree<double>
        {
            Operator = "+",
            Left = new ExpressionTree<double> { Variable = "x", IsLeaf = true },
            Right = new ExpressionTree<double> { Value = 5.0, IsLeaf = true }
        };

        // Act
        var result = expr.Evaluate(new Dictionary<string, double> { ["x"] = 3.0 });

        // Assert
        Assert.Equal(8.0, result);
    }

    [Fact]
    public void Evaluate_ComplexExpression_ReturnsCorrectValue()
    {
        // Arrange: (2 * x) + (3 * y)
        var expr = new ExpressionTree<double>
        {
            Operator = "+",
            Left = new ExpressionTree<double>
            {
                Operator = "*",
                Left = new ExpressionTree<double> { Value = 2.0, IsLeaf = true },
                Right = new ExpressionTree<double> { Variable = "x", IsLeaf = true }
            },
            Right = new ExpressionTree<double>
            {
                Operator = "*",
                Left = new ExpressionTree<double> { Value = 3.0, IsLeaf = true },
                Right = new ExpressionTree<double> { Variable = "y", IsLeaf = true }
            }
        };

        // Act
        var result = expr.Evaluate(new Dictionary<string, double>
        {
            ["x"] = 4.0,
            ["y"] = 5.0
        });

        // Assert
        // (2*4) + (3*5) = 8 + 15 = 23
        Assert.Equal(23.0, result);
    }

    [Fact]
    public void ToString_SimpleExpression_FormatsCorrectly()
    {
        // Arrange: x + 5
        var expr = new ExpressionTree<double>
        {
            Operator = "+",
            Left = new ExpressionTree<double> { Variable = "x", IsLeaf = true },
            Right = new ExpressionTree<double> { Value = 5.0, IsLeaf = true }
        };

        // Act
        var str = expr.ToString();

        // Assert
        Assert.Contains("x", str);
        Assert.Contains("+", str);
        Assert.Contains("5", str);
    }
}
```

---

## Common Testing Patterns

### Edge Cases to Test

1. **Null/Empty checks**
   ```csharp
   var emptyNode = new DecisionTreeNode<double>();
   Assert.Empty(emptyNode.Samples);
   ```

2. **Boundary conditions**
   ```csharp
   // Test p-value boundaries
   Assert.InRange(node.PValue, 0.0, 1.0);
   ```

3. **Invalid operations**
   ```csharp
   // Division by zero in complex numbers
   var zero = new Complex<double>(0, 0);
   Assert.Throws<DivideByZeroException>(() => z.Divide(zero));
   ```

### Type Testing

Test with different numeric types:
```csharp
[Fact]
public void Complex_WorksWithFloat()
{
    var complex = new Complex<float>(3.0f, 4.0f);
    Assert.Equal(5.0f, complex.Magnitude, precision: 5);
}

[Fact]
public void ConfusionMatrix_WorksWithDouble()
{
    var cm = new ConfusionMatrix<double>(50.0, 40.0, 5.0, 5.0);
    Assert.Equal(0.90, cm.Accuracy(), precision: 10);
}
```

---

## Running Tests

```bash
# Run all LinearAlgebra tests
dotnet test --filter "FullyQualifiedName~LinearAlgebra"

# Run specific test class
dotnet test --filter "FullyQualifiedName~ComplexTests"

# Run with coverage
dotnet test /p:CollectCoverage=true
```

---

## Success Criteria

- [ ] Complex tests cover: arithmetic operations, magnitude, phase, trigonometry
- [ ] ConfusionMatrix tests cover: metrics (accuracy, precision, recall, F1), multi-class
- [ ] DecisionTreeNode tests cover: tree structure, properties, samples
- [ ] ConditionalInferenceTreeNode tests cover: statistical properties, inheritance
- [ ] ExpressionTree tests cover: construction, evaluation, variables
- [ ] All tests pass with green checkmarks
- [ ] Code coverage increases from 0% to >80%
- [ ] Edge cases tested (null, zero, boundary conditions)

---

## Common Pitfalls

1. **Don't forget floating-point precision** - Use `Assert.Equal(expected, actual, precision: 10)`
2. **Don't test implementation details** - Test observable behavior
3. **Do test mathematical correctness** - Verify formulas manually
4. **Do test tree traversal** - Ensure parent-child relationships work
5. **Do test statistical properties** - Verify p-values, test statistics
6. **Do handle null references** - Test nullable properties properly

Start with Complex, then ConfusionMatrix, then DecisionTreeNode, then ConditionalInferenceTreeNode, then ExpressionTree. Build incrementally!
