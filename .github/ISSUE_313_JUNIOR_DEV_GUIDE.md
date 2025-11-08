# Issue #313: Junior Developer Implementation Guide
## Implement Missing Standard Loss Functions (RMSE, SparseCrossEntropy, MBE)

## Table of Contents
1. [Understanding Loss Functions](#understanding-loss-functions)
2. [Existing Infrastructure](#existing-infrastructure)
3. [Phase 1: Root Mean Squared Error (RMSE)](#phase-1-root-mean-squared-error-rmse)
4. [Phase 2: Sparse Categorical Cross-Entropy](#phase-2-sparse-categorical-cross-entropy)
5. [Phase 3: Mean Bias Error (MBE)](#phase-3-mean-bias-error-mbe)
6. [Common Pitfalls](#common-pitfalls)
7. [Testing Strategy](#testing-strategy)

---

## Understanding Loss Functions

### What is a Loss Function?
A loss function measures how wrong your model's predictions are. Lower loss = better predictions.

**Real-World Analogy:**
Think of a loss function like a scoring system in golf:
- Perfect shot (hole-in-one) = 0 loss
- Close to the hole = small loss
- Far from the hole = large loss
- Different loss functions are like different scoring rules (some penalize distance more harshly)

### Why Do We Need Different Loss Functions?

| Loss Function | Use Case | Example Problem |
|--------------|----------|-----------------|
| **RMSE** | Regression with interpretable units | Predicting house prices in dollars |
| **Sparse Categorical Cross-Entropy** | Multi-class classification with integer labels | Image classification (cat=0, dog=1, bird=2) |
| **MBE** | Detecting systematic bias | Checking if temperature sensor always reads 2 degrees high |

---

## Existing Infrastructure

### LossFunctionBase Pattern

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/LossFunctions/LossFunctionBase.cs`

All loss functions inherit from this base class:

```csharp
public abstract class LossFunctionBase<T> : ILossFunction<T>
{
    protected readonly INumericOperations<T> NumOps;  // Auto-initialized

    // Calculate scalar loss value
    public abstract T CalculateLoss(Vector<T> predicted, Vector<T> actual);

    // Calculate gradient vector for backpropagation
    public abstract Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual);

    // Helper for validation
    protected void ValidateVectorLengths(Vector<T> predicted, Vector<T> actual);
}
```

### Key Points:
1. **NumOps**: Use this for ALL arithmetic operations (never hardcode `double` or `float`)
2. **CalculateLoss**: Returns a single scalar value (how wrong overall)
3. **CalculateDerivative**: Returns a vector (how to adjust each prediction)
4. **ValidateVectorLengths**: Always call first in both methods

### Example: Mean Squared Error

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
        // Derivative: 2*(predicted-actual)/n
        return predicted.Subtract(actual)
            .Transform(x => NumOps.Multiply(NumOps.FromDouble(2), x))
            .Divide(NumOps.FromDouble(predicted.Length));
    }
}
```

---

## Phase 1: Root Mean Squared Error (RMSE)

### What is RMSE?

**Formula:** RMSE = sqrt((1/n) * sum((predicted - actual)^2))

**Key Properties:**
- Same units as the target variable (unlike MSE which is squared)
- Less sensitive to outliers than MSE (due to square root)
- Always positive, perfect predictions = 0
- Commonly used when interpretability matters

**When to Use:**
- You want errors in the same units as predictions
- Comparing models on the same dataset
- Interpretability is important (e.g., "average error of 5 dollars")

**Example:**
Predicting house prices:
- Actual: $100,000
- Predicted: $105,000
- Error: $5,000 (easy to interpret!)
- MSE would be: $25,000,000 (hard to interpret!)

### Existing Code to Leverage

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/Helpers/StatisticsHelper.cs` (line 1355)

```csharp
public static T CalculateRootMeanSquaredError(Vector<T> actualValues, Vector<T> predictedValues)
{
    return _numOps.Sqrt(CalculateMeanSquaredError(actualValues, predictedValues));
}
```

**NOTE:** Existing infrastructure already implements the core calculation!

### Implementation Steps

#### Step 1: Create RootMeanSquaredErrorLoss.cs

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/LossFunctions/RootMeanSquaredErrorLoss.cs`

```csharp
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
/// The formula is: RMSE = sqrt((1/n) * sum((predicted - actual)^2))
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
///
/// Example: Predicting house prices
/// - Actual: $100,000, Predicted: $105,000
/// - RMSE: $5,000 (interpretable!)
/// - MSE: $25,000,000 (hard to understand!)
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
    /// - RMSE = sqrt(MSE) = sqrt((1/n) * sum((predicted - actual)^2))
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

#### Step 2: Create Unit Tests

**File:** `C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/LossFunctions/RootMeanSquaredErrorLossTests.cs`

```csharp
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

        // Expected: sqrt((0.5^2 + 0.5^2 + 0.5^2 + 0.5^2)/4) = sqrt(1/4) = 0.5

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

        // RMSE = sqrt((1^2 + 1^2 + 1^2 + 1^2)/4) = sqrt(1) = 1.0
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

## Phase 2: Sparse Categorical Cross-Entropy

### What is Sparse Categorical Cross-Entropy?

**Formula:** SCCE = -(1/n) * sum[log(predicted[actual_class_index])]

**Key Properties:**
- Identical to Categorical Cross-Entropy, but accepts integer labels
- More memory efficient (no one-hot encoding needed)
- Perfect for multi-class classification with many classes

**When to Use:**
- You have integer class labels (not one-hot encoded)
- Memory efficiency matters (large datasets or many classes)
- You want simpler data preprocessing

### Difference from Categorical Cross-Entropy

| Feature | Categorical Cross-Entropy | Sparse Categorical Cross-Entropy |
|---------|---------------------------|----------------------------------|
| **Input format** | One-hot: `[0, 0, 1, 0]` | Integer: `2` |
| **Memory usage** | High (N × num_classes) | Low (N × 1) |
| **Use case** | Small datasets | Large datasets, many classes |
| **Example** | `[[0,0,1], [1,0,0]]` | `[2, 0]` |

### Implementation Steps

#### Step 1: Create SparseCategoricalCrossEntropyLoss.cs

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/LossFunctions/SparseCategoricalCrossEntropyLoss.cs`

```csharp
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
/// The formula is: SCCE = -(1/n) * sum[log(predicted[actual_class_index])]
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
///
/// Example: Image classification
/// - Classes: cat=0, dog=1, bird=2
/// - Input: actual = [1, 0, 2] (dog, cat, bird)
/// - Predictions: [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
/// - Loss: -(log(0.8) + log(0.7) + log(0.7)) / 3
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
    /// <exception cref="ArgumentException">Thrown when numClasses is less than 2.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> You must specify how many classes your classification problem has.
    /// For example, if classifying images as cat/dog/bird, numClasses = 3.
    /// </remarks>
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

#### Step 2: Create Unit Tests

**File:** `C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/LossFunctions/SparseCategoricalCrossEntropyLossTests.cs`

```csharp
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
    public void CalculateLoss_PerfectPredictions_ReturnsNearZero()
    {
        // Arrange (3 classes, 2 samples)
        var loss = new SparseCategoricalCrossEntropyLoss<double>(numClasses: 3);

        // Sample 0: class 1 (probabilities: [0.0, 1.0, 0.0])
        // Sample 1: class 0 (probabilities: [1.0, 0.0, 0.0])
        var predicted = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0, 0.0, 0.0 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 }); // Class labels

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert (log(1.0) = 0, so loss near 0, accounting for epsilon)
        Assert.True(result < 0.0001);
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

        // Expected: -(log(0.7) + log(0.8)) / 2
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
}
```

---

## Phase 3: Mean Bias Error (MBE)

### What is Mean Bias Error?

**Formula:** MBE = (1/n) * sum(predicted - actual)

**Key Properties:**
- Measures systematic bias (consistent over/under-prediction)
- Can be positive (over-prediction) or negative (under-prediction)
- NOT suitable as a primary loss function (can be gamed by opposite errors)
- Excellent as a diagnostic metric

**When to Use:**
- Detecting systematic bias in predictions
- Quality control (e.g., sensor calibration)
- Model evaluation (not training)

**Example:**
Temperature sensor:
- Actual: [20, 25, 30, 35]
- Predicted: [22, 27, 32, 37]
- MBE: +2 degrees (consistently 2 degrees high)
- Action: Calibrate sensor to subtract 2 degrees

### Implementation Steps

#### Step 1: Create MBELoss.cs

**File:** `C:/Users/cheat/source/repos/AiDotNet/src/LossFunctions/MBELoss.cs`

```csharp
namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Mean Bias Error (MBE) loss function.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mean Bias Error measures systematic bias in predictions.
/// Unlike MSE or RMSE which measure overall error, MBE tells you if your model
/// consistently over-predicts (positive MBE) or under-predicts (negative MBE).
///
/// The formula is: MBE = (1/n) * sum(predicted - actual)
///
/// Key properties:
/// - Can be positive, negative, or zero
/// - Positive MBE = over-prediction (predicting too high)
/// - Negative MBE = under-prediction (predicting too low)
/// - Zero MBE = no systematic bias (but can still have large random errors)
/// - NOT suitable as a primary training loss (can be gamed by opposite errors)
///
/// MBE is ideal for:
/// - Detecting systematic bias (e.g., sensor calibration)
/// - Quality control and diagnostics
/// - Evaluating model fairness (different bias for different groups)
///
/// WARNING: Do NOT use MBE alone for training!
/// - A model that predicts +10 and -10 alternately has MBE = 0 (looks perfect)
/// - But it has terrible predictions (MSE = 100)
///
/// Example: Temperature sensor
/// - Actual: [20, 25, 30, 35] degrees
/// - Predicted: [22, 27, 32, 37] degrees
/// - MBE: +2 degrees (consistently 2 degrees high)
/// - Action: Calibrate sensor to subtract 2 degrees
/// </para>
/// </remarks>
public class MBELoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Mean Bias Error between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The mean bias error value (can be positive, negative, or zero).</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // MBE = (1/n) * sum(predicted - actual)
        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Subtract(predicted[i], actual[i]));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    /// <summary>
    /// Calculates the derivative of the Mean Bias Error loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of MBE for each prediction.</returns>
    /// <remarks>
    /// The derivative is: dMBE/dpredicted = 1/n for all elements
    ///
    /// Derivation:
    /// - MBE = (1/n) * sum(predicted - actual)
    /// - d/dpredicted_i [(1/n) * sum(predicted - actual)] = 1/n
    /// - The gradient is constant (1/n) for all elements
    ///
    /// This means every prediction contributes equally to the bias,
    /// regardless of how far off it is.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Derivative is constant: 1/n for all elements
        T derivative = NumOps.Divide(NumOps.One, NumOps.FromDouble(predicted.Length));

        var gradient = new Vector<T>(predicted.Length);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = derivative;
        }

        return gradient;
    }
}
```

#### Step 2: Create Unit Tests

**File:** `C:/Users/cheat/source/repos/AiDotNet/tests/UnitTests/LossFunctions/MBELossTests.cs`

```csharp
namespace AiDotNet.Tests.LossFunctions;

public class MBELossTests
{
    [Fact]
    public void CalculateLoss_PerfectPredictions_ReturnsZero()
    {
        // Arrange
        var loss = new MBELoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, result, precision: 10);
    }

    [Fact]
    public void CalculateLoss_PositiveBias_ReturnsPositiveValue()
    {
        // Arrange
        var loss = new MBELoss<double>();
        var predicted = new Vector<double>(new[] { 2.0, 3.0, 4.0, 5.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Expected: (1 + 1 + 1 + 1) / 4 = 1.0 (over-predicting by 1)

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(1.0, result, precision: 10);
    }

    [Fact]
    public void CalculateLoss_NegativeBias_ReturnsNegativeValue()
    {
        // Arrange
        var loss = new MBELoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Expected: (-1 - 1 - 1 - 1) / 4 = -1.0 (under-predicting by 1)

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(-1.0, result, precision: 10);
    }

    [Fact]
    public void CalculateLoss_OppositeErrors_ReturnsZero()
    {
        // Arrange
        var loss = new MBELoss<double>();
        var predicted = new Vector<double>(new[] { 11.0, 9.0, 11.0, 9.0 });
        var actual = new Vector<double>(new[] { 10.0, 10.0, 10.0, 10.0 });

        // Expected: (+1 - 1 + 1 - 1) / 4 = 0.0 (no systematic bias, but high random error)

        // Act
        double result = loss.CalculateLoss(predicted, actual);

        // Assert
        Assert.Equal(0.0, result, precision: 10);
    }

    [Fact]
    public void CalculateDerivative_ValidInputs_ReturnsConstantGradient()
    {
        // Arrange
        var loss = new MBELoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.5, 2.5, 3.5, 4.5 });

        // Expected: all elements = 1/4 = 0.25

        // Act
        var gradient = loss.CalculateDerivative(predicted, actual);

        // Assert
        Assert.Equal(4, gradient.Length);
        Assert.All(gradient.ToArray(), g => Assert.Equal(0.25, g, precision: 10));
    }

    [Fact]
    public void CalculateLoss_DifferentLengths_ThrowsArgumentException()
    {
        // Arrange
        var loss = new MBELoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new[] { 1.0, 2.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }
}
```

---

## Common Pitfalls

### 1. Using Wrong Method Names
- Use `CalculateLoss()` and `CalculateDerivative()`, NOT `Forward()` and `Backward()`

### 2. Forgetting Input Validation
- Always call `ValidateVectorLengths()` at the start of both methods

### 3. Hardcoding Numeric Values
- WRONG: `return predicted.Divide(2.0);`
- RIGHT: `return predicted.Divide(NumOps.FromDouble(2.0));`

### 4. Forgetting Epsilon for Log Operations
- WRONG: `NumOps.Log(predictedProb)` (can cause log(0) = undefined)
- RIGHT: `MathHelper.Clamp(predictedProb, _epsilon, NumOps.Subtract(NumOps.One, _epsilon))`

### 5. Mixing Up Parameter Order
- StatisticsHelper uses `(actual, predicted)` order
- Loss functions use `(predicted, actual)` order
- Be careful when calling StatisticsHelper!

### 6. Not Handling Edge Cases
- Perfect predictions (zero loss)
- Divide by zero scenarios (RMSE with zero error)
- Invalid class indices (SparseCategoricalCrossEntropy)

### 7. Using MBE as Primary Training Loss
- MBE can be gamed by opposite errors
- Use MBE for diagnostics only, NOT for training

---

## Testing Strategy

### 1. Unit Tests
Test each loss function independently with:
- Perfect predictions (zero loss)
- Typical cases (known expected values)
- Edge cases (empty vectors, single element)
- Invalid inputs (wrong dimensions, out-of-range values)

### 2. Numerical Stability Tests
- Very small values (near epsilon)
- Very large values
- Values near boundaries (0, 1 for probabilities)

### 3. Type Tests
Test with multiple numeric types:
- `double`
- `float`

### 4. Gradient Verification
Use finite difference approximation to verify gradients:
```csharp
// Numerical gradient: (f(x + h) - f(x - h)) / (2h)
var h = 0.0001;
var predicted_plus = predicted.Clone();
predicted_plus[i] = predicted[i] + h;
var predicted_minus = predicted.Clone();
predicted_minus[i] = predicted[i] - h;

var numerical_gradient = (loss.CalculateLoss(predicted_plus, actual) -
                          loss.CalculateLoss(predicted_minus, actual)) / (2 * h);

// Compare with analytical gradient
var analytical_gradient = loss.CalculateDerivative(predicted, actual)[i];
Assert.Equal(numerical_gradient, analytical_gradient, precision: 4);
```

### 5. Integration Tests
Test with actual neural network training to ensure:
- Loss decreases during training
- Gradients flow correctly through backpropagation
- Convergence on simple problems

---

## Next Steps

1. **Start with RMSE** (simplest - wraps existing code)
2. **Then SparseCategoricalCrossEntropy** (new implementation, more complex)
3. **Finally MBE** (simple formula, but important edge cases)
4. **Test thoroughly** - each loss function is critical for model training
5. **Document edge cases** - especially divide-by-zero and log(0)

**Good luck!** Remember: loss functions are the foundation of machine learning. Get them right, and everything else builds on solid ground.
