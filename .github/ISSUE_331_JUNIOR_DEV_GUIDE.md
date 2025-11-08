# Issue #331: Junior Developer Implementation Guide

## Understanding LossFunctionBase<T>

**File**: `src/LossFunctions/LossFunctionBase.cs`

### What is a Loss Function?
A loss function measures how wrong your model's predictions are. Lower loss = better predictions.

### Key Components:
```csharp
// Base class structure
public abstract class LossFunctionBase<T> : ILossFunction<T>
{
    protected readonly INumericOperations<T> NumOps;  // Automatically initialized

    // Calculate loss value (scalar)
    public abstract T CalculateLoss(Vector<T> predicted, Vector<T> actual);

    // Calculate gradient for backpropagation (vector)
    public abstract Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual);

    // Helper method for validation
    protected void ValidateVectorLengths(Vector<T> predicted, Vector<T> actual);
}
```

### Example Pattern (MeanSquaredErrorLoss):
```csharp
public class MeanSquaredErrorLoss<T> : LossFunctionBase<T>
{
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        return StatisticsHelper<T>.CalculateMeanSquaredError(predicted, actual);
    }

    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        // Derivative of MSE: 2*(predicted-actual)/n
        return predicted.Subtract(actual)
            .Transform(x => NumOps.Multiply(NumOps.FromDouble(2), x))
            .Divide(NumOps.FromDouble(predicted.Length));
    }
}
```

---

## Phase 1: Step-by-Step Implementation

### AC 1.1: RootMeanSquaredErrorLoss - WRAPPER AROUND EXISTING CODE

**Existing Code to Use**: `src/Helpers/StatisticsHelper.cs` (line 1355)

**Step 1**: Understand existing RMSE infrastructure
```bash
# Verify RMSE calculation exists
grep -n "CalculateRootMeanSquaredError" src/Helpers/StatisticsHelper.cs

# Output shows:
# 1355: public static T CalculateRootMeanSquaredError(Vector<T> actualValues, Vector<T> predictedValues)
# 1357:     return _numOps.Sqrt(CalculateMeanSquaredError(actualValues, predictedValues));
```

**Step 2**: Create RootMeanSquaredErrorLoss.cs
```csharp
// File: src/LossFunctions/RootMeanSquaredErrorLoss.cs
namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Root Mean Squared Error (RMSE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Root Mean Squared Error is the square root of MSE.
/// It measures the standard deviation of prediction errors.
///
/// The formula is: RMSE = sqrt((1/n) * ∑(predicted - actual)²)
///
/// Key properties:
/// - Same units as the target variable (unlike MSE which is squared)
/// - Less sensitive to outliers than MSE (due to square root)
/// - Commonly used for regression problems where interpretability matters
/// - Always positive, with perfect predictions giving zero
///
/// RMSE is ideal when:
/// - You want errors in the same units as your predictions
/// - You're comparing models on the same dataset
/// - Interpretability is important (e.g., "average error of 5 units")
/// </para>
/// </remarks>
public class RootMeanSquaredErrorLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Root Mean Squared Error between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The root mean squared error value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Use existing infrastructure from StatisticsHelper
        return StatisticsHelper<T>.CalculateRootMeanSquaredError(actual, predicted);
    }

    /// <summary>
    /// Calculates the derivative of the Root Mean Squared Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of RMSE for each prediction.</returns>
    /// <remarks>
    /// The derivative is: dRMSE/dpredicted = (predicted - actual) / (n * RMSE)
    ///
    /// Derivation:
    /// - RMSE = sqrt(MSE) = sqrt((1/n) * ∑(predicted - actual)²)
    /// - Let MSE = m, then RMSE = sqrt(m)
    /// - dRMSE/dpredicted = d(sqrt(m))/dm * dm/dpredicted
    /// - d(sqrt(m))/dm = 1/(2*sqrt(m)) = 1/(2*RMSE)
    /// - dm/dpredicted = 2*(predicted - actual)/n
    /// - Therefore: dRMSE/dpredicted = (1/(2*RMSE)) * (2*(predicted - actual)/n)
    /// - Simplifies to: (predicted - actual) / (n * RMSE)
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Calculate current RMSE value for denominator
        T rmse = CalculateLoss(predicted, actual);

        // Calculate n (number of samples)
        T n = NumOps.FromDouble(predicted.Length);

        // Calculate residuals: (predicted - actual)
        Vector<T> residuals = predicted.Subtract(actual);

        // Divide by (n * RMSE)
        T denominator = NumOps.Multiply(n, rmse);

        // Handle edge case: if RMSE is zero (perfect predictions), gradient is zero
        if (NumOps.Compare(rmse, NumOps.Zero) == 0)
        {
            return new Vector<T>(predicted.Length); // All zeros
        }

        return residuals.Divide(denominator);
    }
}
```

**Step 3**: Create unit test
```csharp
// File: tests/UnitTests/LossFunctions/RootMeanSquaredErrorLossTests.cs
namespace AiDotNet.Tests.LossFunctions;

public class RootMeanSquaredErrorLossTests
{
    [Fact]
    public void CalculateLoss_PerfectPredictions_ReturnsZero()
    {
        // Arrange
        var loss = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, result, precision: 10);
    }

    [Fact]
    public void CalculateLoss_ValidInputs_ReturnsCorrectRMSE()
    {
        // Arrange
        var loss = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.5, 2.5, 3.5, 4.5 });

        // Expected: sqrt((0.5² + 0.5² + 0.5² + 0.5²)/4) = sqrt(1/4) = 0.5

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.5, result, precision: 10);
    }

    [Fact]
    public void CalculateDerivative_ValidInputs_ReturnsCorrectGradient()
    {
        // Arrange
        var loss = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 2.0, 3.0, 4.0, 5.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // RMSE = sqrt((1² + 1² + 1² + 1²)/4) = sqrt(1) = 1.0
        // Gradient = (predicted - actual) / (n * RMSE) = [1,1,1,1] / (4 * 1) = [0.25, 0.25, 0.25, 0.25]

        // Act
        var gradient = loss.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(4, gradient.Length);
        Assert.Equal(0.25, gradient[0], precision: 10);
        Assert.Equal(0.25, gradient[1], precision: 10);
        Assert.Equal(0.25, gradient[2], precision: 10);
        Assert.Equal(0.25, gradient[3], precision: 10);
    }

    [Fact]
    public void CalculateDerivative_PerfectPredictions_ReturnsZeroGradient()
    {
        // Arrange
        var loss = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var gradient = loss.CalculateDerivative(predicted, actual);

        // Assert
        Assert.All(gradient.ToArray(), g => Assert.Equal(0.0, g, precision: 10));
    }

    [Fact]
    public void CalculateLoss_DifferentLengths_ThrowsArgumentException()
    {
        // Arrange
        var loss = new RootMeanSquaredErrorLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }
}
```

---

## Phase 2: Step-by-Step Implementation

### AC 2.1: SparseCategoricalCrossEntropyLoss - NEW IMPLEMENTATION NEEDED

**No existing code - you must implement integer label handling**

**What's the difference from CategoricalCrossEntropy?**

| Feature | CategoricalCrossEntropy | SparseCategoricalCrossEntropy |
|---------|------------------------|-------------------------------|
| Input format | One-hot encoded: `[0, 0, 1, 0]` | Integer label: `2` |
| Memory usage | High (N × num_classes) | Low (N × 1) |
| Use case | Small datasets | Large datasets, many classes |

**Implementation**:
```csharp
// File: src/LossFunctions/SparseCategoricalCrossEntropyLoss.cs
namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Sparse Categorical Cross Entropy loss function for multi-class classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sparse Categorical Cross Entropy is identical to Categorical Cross Entropy,
/// but accepts integer class labels instead of one-hot encoded vectors.
///
/// Example difference:
/// - Categorical: actual = [0, 0, 1, 0, 0] (one-hot for class 2)
/// - Sparse: actual = [2] (integer label for class 2)
///
/// The formula is: SCCE = -(1/n) * ∑[log(predicted[actual_class_index])]
///
/// Where:
/// - actual_class_index is the integer class label
/// - predicted is a probability distribution over all classes (from softmax)
/// - We only look at the predicted probability for the true class
///
/// Key properties:
/// - More memory efficient than one-hot encoding
/// - Same mathematical result as Categorical Cross Entropy
/// - Ideal for problems with many classes (e.g., 1000+ classes)
/// - Requires integer labels in range [0, num_classes)
///
/// Use this when:
/// - You have integer class labels (not one-hot encoded)
/// - Memory efficiency matters (large datasets or many classes)
/// - You want simpler data preprocessing (no one-hot encoding step)
/// </para>
/// </remarks>
public class SparseCategoricalCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Small value to prevent numerical instability with log(0).
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// Number of classes in the classification problem.
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// Initializes a new instance of the SparseCategoricalCrossEntropyLoss class.
    /// </summary>
    /// <param name="numClasses">The number of classes in the classification problem.</param>
    public SparseCategoricalCrossEntropyLoss(int numClasses)
    {
        if (numClasses < 2)
            throw new ArgumentException("Number of classes must be at least 2.", nameof(numClasses));

        _numClasses = numClasses;
        _epsilon = NumOps.FromDouble(1e-15);
    }

    /// <summary>
    /// Calculates the Sparse Categorical Cross Entropy loss between predicted probabilities and actual class labels.
    /// </summary>
    /// <param name="predicted">The predicted probabilities (length = num_samples * num_classes, flattened).</param>
    /// <param name="actual">The actual class labels as integers (length = num_samples).</param>
    /// <returns>The sparse categorical cross entropy loss value.</returns>
    /// <remarks>
    /// Input format:
    /// - predicted: [sample0_class0_prob, sample0_class1_prob, ..., sample1_class0_prob, ...]
    /// - actual: [sample0_true_class, sample1_true_class, ...]
    ///
    /// Example for 2 samples, 3 classes:
    /// - predicted = [0.1, 0.7, 0.2, 0.8, 0.1, 0.1] (flattened 2x3 matrix)
    /// - actual = [1, 0] (sample 0 is class 1, sample 1 is class 0)
    /// - Loss = -(log(0.7) + log(0.8)) / 2
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        // Validate input dimensions
        int numSamples = actual.Length;
        if (predicted.Length != numSamples * _numClasses)
        {
            throw new ArgumentException(
                $"Predicted vector length ({predicted.Length}) must equal num_samples ({numSamples}) * num_classes ({_numClasses}).");
        }

        T sum = NumOps.Zero;

        for (int i = 0; i < numSamples; i++)
        {
            // Convert class label from T to int
            int classIndex = Convert.ToInt32(NumOps.ToDouble(actual[i]));

            // Validate class index is in valid range
            if (classIndex < 0 || classIndex >= _numClasses)
            {
                throw new ArgumentException(
                    $"Class label {classIndex} is out of range [0, {_numClasses}).");
            }

            // Get predicted probability for the true class
            int predictedIndex = i * _numClasses + classIndex;
            T predictedProb = predicted[predictedIndex];

            // Clamp to prevent log(0)
            predictedProb = MathHelper.Clamp(predictedProb, _epsilon, NumOps.Subtract(NumOps.One, _epsilon));

            // Accumulate: -log(predicted_prob_for_true_class)
            sum = NumOps.Add(sum, NumOps.Log(predictedProb));
        }

        // Return negative average
        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(numSamples)));
    }

    /// <summary>
    /// Calculates the derivative of the Sparse Categorical Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted probabilities (length = num_samples * num_classes, flattened).</param>
    /// <param name="actual">The actual class labels as integers (length = num_samples).</param>
    /// <returns>A vector containing the derivatives of SCCE for each prediction.</returns>
    /// <remarks>
    /// The gradient for each element is:
    /// - If element corresponds to true class: (predicted - 1) / num_samples
    /// - Otherwise: predicted / num_samples
    ///
    /// This is equivalent to: (predicted - one_hot(actual)) / num_samples
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        // Validate input dimensions
        int numSamples = actual.Length;
        if (predicted.Length != numSamples * _numClasses)
        {
            throw new ArgumentException(
                $"Predicted vector length ({predicted.Length}) must equal num_samples ({numSamples}) * num_classes ({_numClasses}).");
        }

        // Initialize gradient vector (same shape as predicted)
        var gradient = new Vector<T>(predicted.Length);

        for (int i = 0; i < numSamples; i++)
        {
            // Convert class label from T to int
            int classIndex = Convert.ToInt32(NumOps.ToDouble(actual[i]));

            // Validate class index
            if (classIndex < 0 || classIndex >= _numClasses)
            {
                throw new ArgumentException(
                    $"Class label {classIndex} is out of range [0, {_numClasses}).");
            }

            // For each class in this sample
            for (int c = 0; c < _numClasses; c++)
            {
                int index = i * _numClasses + c;

                if (c == classIndex)
                {
                    // True class: gradient = (predicted - 1) / num_samples
                    gradient[index] = NumOps.Divide(
                        NumOps.Subtract(predicted[index], NumOps.One),
                        NumOps.FromDouble(numSamples)
                    );
                }
                else
                {
                    // Other classes: gradient = predicted / num_samples
                    gradient[index] = NumOps.Divide(
                        predicted[index],
                        NumOps.FromDouble(numSamples)
                    );
                }
            }
        }

        return gradient;
    }
}
```

**Step 3**: Create unit test
```csharp
// File: tests/UnitTests/LossFunctions/SparseCategoricalCrossEntropyLossTests.cs
namespace AiDotNet.Tests.LossFunctions;

public class SparseCategoricalCrossEntropyLossTests
{
    [Fact]
    public void Constructor_InvalidNumClasses_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SparseCategoricalCrossEntropyLoss<double>(1));
        Assert.Throws<ArgumentException>(() => new SparseCategoricalCrossEntropyLoss<double>(0));
    }

    [Fact]
    public void CalculateLoss_PerfectPredictions_ReturnsZero()
    {
        // Arrange (3 classes, 2 samples)
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);

        // Sample 0: class 1 (probabilities: [0.0, 1.0, 0.0])
        // Sample 1: class 0 (probabilities: [1.0, 0.0, 0.0])
        var predicted = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 }); // Class labels

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert (log(1.0) = 0, so loss = 0)
        Assert.True(result < 0.0001); // Near zero (accounting for epsilon)
    }

    [Fact]
    public void CalculateLoss_ValidInputs_ReturnsCorrectLoss()
    {
        // Arrange (3 classes, 2 samples)
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);

        // Sample 0: true class = 1, predicted = [0.1, 0.7, 0.2]
        // Sample 1: true class = 0, predicted = [0.8, 0.1, 0.1]
        var predicted = new Vector<double>(new[] { 0.1, 0.7, 0.2, 0.8, 0.1, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        // Expected: -(log(0.7) + log(0.8)) / 2 ≈ 0.2027
        double expected = -(Math.Log(0.7) + Math.Log(0.8)) / 2.0;

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(expected, result, precision: 4);
    }

    [Fact]
    public void CalculateLoss_InvalidClassIndex_ThrowsArgumentException()
    {
        // Arrange
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);
        var predicted = new Vector<double>(new[] { 0.1, 0.7, 0.2 });
        var actual = new Vector<double>(new[] { 5.0 }); // Invalid: class 5 doesn't exist

        // Act & Assert
        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void CalculateLoss_WrongPredictedLength_ThrowsArgumentException()
    {
        // Arrange
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);
        var predicted = new Vector<double>(new[] { 0.1, 0.7 }); // Should be 3 elements
        var actual = new Vector<double>(new[] { 1.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void CalculateDerivative_ValidInputs_ReturnsCorrectGradient()
    {
        // Arrange (3 classes, 1 sample)
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);

        // True class = 1, predicted = [0.2, 0.5, 0.3]
        var predicted = new Vector<double>(new[] { 0.2, 0.5, 0.3 });
        var actual = new Vector<double>(new[] { 1.0 });

        // Expected gradient = (predicted - [0, 1, 0]) / 1 = [0.2, -0.5, 0.3]

        // Act
        var gradient = loss.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(3, gradient.Length);
        Assert.Equal(0.2, gradient[0], precision: 10);
        Assert.Equal(-0.5, gradient[1], precision: 10);
        Assert.Equal(0.3, gradient[2], precision: 10);
    }

    [Fact]
    public void CalculateDerivative_MultipleSamples_ReturnsCorrectGradient()
    {
        // Arrange (3 classes, 2 samples)
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);

        // Sample 0: true class = 1, predicted = [0.1, 0.7, 0.2]
        // Sample 1: true class = 0, predicted = [0.8, 0.1, 0.1]
        var predicted = new Vector<double>(new[] { 0.1, 0.7, 0.2, 0.8, 0.1, 0.1 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        // Expected:
        // Sample 0: (pred - [0,1,0]) / 2 = [0.1, -0.3, 0.2] / 2 = [0.05, -0.15, 0.1]
        // Sample 1: (pred - [1,0,0]) / 2 = [-0.2, 0.1, 0.1] / 2 = [-0.1, 0.05, 0.05]

        // Act
        var gradient = loss.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(6, gradient.Length);

        // Sample 0 gradient
        Assert.Equal(0.05, gradient[0], precision: 10);
        Assert.Equal(-0.15, gradient[1], precision: 10);
        Assert.Equal(0.1, gradient[2], precision: 10);

        // Sample 1 gradient
        Assert.Equal(-0.1, gradient[3], precision: 10);
        Assert.Equal(0.05, gradient[4], precision: 10);
        Assert.Equal(0.05, gradient[5], precision: 10);
    }
}
```

---

## Common Pitfalls to Avoid:

1. **DON'T use wrong method names** - Use `CalculateLoss()` and `CalculateDerivative()`, NOT `Forward()` and `Backward()`
2. **DON'T forget to validate inputs** - Always call `ValidateVectorLengths()` at the start of methods
3. **DON'T hardcode numeric values** - Use `NumOps.FromDouble()` instead of `0.5`
4. **DON'T forget epsilon for log operations** - Prevents log(0) which is undefined
5. **DON'T mix up parameter order** - StatisticsHelper uses `(actual, predicted)` order
6. **DO leverage existing infrastructure** - RMSE calculation already exists in StatisticsHelper
7. **DO handle edge cases** - Perfect predictions (zero loss), divide by zero scenarios
8. **DO write comprehensive tests** - Test perfect predictions, typical cases, edge cases, invalid inputs

---

## Testing Strategy:

1. **Unit Tests**: Test each loss function independently
2. **Edge Cases**:
   - Perfect predictions (zero loss)
   - Completely wrong predictions (high loss)
   - Invalid inputs (wrong dimensions, out-of-range class labels)
3. **Numerical Stability**: Test with very small/large values
4. **Type Tests**: Test with `double` and `float` generic types
5. **Integration**: Test with actual neural network training

**Next Steps**: Start with RootMeanSquaredErrorLoss (simple wrapper), then SparseCategoricalCrossEntropyLoss (new implementation). Build incrementally and test thoroughly!
