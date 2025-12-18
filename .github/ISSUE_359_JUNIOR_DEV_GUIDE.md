# Issue #359: Junior Developer Implementation Guide - Statistics Classes

## Overview
This guide helps you create **unit tests** for statistics classes: BasicStats, ErrorStats, GeneticStats, ModelStats, PredictionStats, and Quartile. These classes currently have **0% test coverage** despite being essential for model evaluation and analysis.

**Goal**: Write comprehensive unit tests to verify statistical calculations are accurate.

---

## Understanding the Classes

### BasicStats<T> (`src/Statistics/BasicStats.cs`)
Calculates fundamental descriptive statistics for a dataset.

**Key Properties**:
- `Mean`: Average value (sum / count)
- `Variance`: Spread around mean (average of squared differences from mean)
- `StandardDeviation`: Square root of variance
- `Skewness`: Measure of asymmetry (negative = left-skewed, positive = right-skewed)
- `Kurtosis`: Measure of tail heaviness (high = heavy tails/outliers)
- `Min`: Minimum value
- `Max`: Maximum value
- `Range`: Max - Min
- `Quartiles`: Q1 (25th percentile), Q2 (median, 50th), Q3 (75th percentile)
- `InterquartileRange`: Q3 - Q1 (middle 50% spread)
- `Count`: Number of values

**Mathematical Formulas**:
- Mean: μ = (Σ x) / n
- Variance: σ² = Σ(x - μ)² / n
- Standard Deviation: σ = sqrt(σ²)
- Skewness: Σ[(x - μ)³] / [n × σ³]
- Kurtosis: Σ[(x - μ)⁴] / [n × σ⁴] - 3

**Example Usage**:
```csharp
var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
var stats = new BasicStats<double>(data);

Console.WriteLine($"Mean: {stats.Mean}"); // 3.0
Console.WriteLine($"StdDev: {stats.StandardDeviation}"); // ~1.414
Console.WriteLine($"Range: {stats.Range}"); // 4.0
```

---

### ErrorStats<T> (`src/Statistics/ErrorStats.cs`)
Calculates prediction error metrics.

**Key Properties**:
- `MAE`: Mean Absolute Error = Σ|predicted - actual| / n
- `MSE`: Mean Squared Error = Σ(predicted - actual)² / n
- `RMSE`: Root Mean Squared Error = sqrt(MSE)
- `MAPE`: Mean Absolute Percentage Error = Σ|predicted - actual| / |actual| × 100 / n
- `SMAPE`: Symmetric Mean Absolute Percentage Error
- `MeanBiasError`: Average of (predicted - actual)
- `MedianAbsoluteError`: Median of |predicted - actual|
- `MaxError`: Max(|predicted - actual|)
- `R2Score`: Coefficient of determination (1 = perfect, 0 = baseline, <0 = worse than baseline)
- `AUCROC`: Area Under ROC Curve (classification)
- `AUCPR`: Area Under Precision-Recall Curve

**Example Usage**:
```csharp
var actual = new[] { 3.0, -0.5, 2.0, 7.0 };
var predicted = new[] { 2.5, 0.0, 2.0, 8.0 };
var errors = new ErrorStats<double>(actual, predicted);

Console.WriteLine($"MAE: {errors.MAE}"); // Average absolute error
Console.WriteLine($"RMSE: {errors.RMSE}"); // Root mean squared error
Console.WriteLine($"R2: {errors.R2Score}"); // Goodness of fit
```

---

### GeneticStats<T> (`src/Statistics/GeneticStats.cs`)
Tracks genetic algorithm statistics.

**Key Properties**:
- `Generation`: Current generation number
- `BestFitness`: Best fitness value found
- `AverageFitness`: Average fitness in current population
- `WorstFitness`: Worst fitness in current population
- `FitnessVariance`: Variance of fitness values
- `SelectionPressure`: Ratio of best to average fitness
- `DiversityScore`: Measure of population diversity
- `ConvergenceRate`: Rate of improvement
- `StagnationCount`: Generations without improvement

**Example Usage**:
```csharp
var fitnessValues = new[] { 0.9, 0.8, 0.7, 0.6, 0.5 };
var stats = new GeneticStats<double>(100, fitnessValues);

Console.WriteLine($"Gen: {stats.Generation}"); // 100
Console.WriteLine($"Best: {stats.BestFitness}"); // 0.9
Console.WriteLine($"Diversity: {stats.DiversityScore}");
```

---

### ModelStats<T> (`src/Statistics/ModelStats.cs`)
Comprehensive model performance statistics.

**Key Properties**:
- `TrainingError`: Error on training set
- `ValidationError`: Error on validation set
- `TestError`: Error on test set
- `Overfitting`: ValidationError - TrainingError
- `ParameterCount`: Number of model parameters
- `TrainingTime`: Time to train (milliseconds)
- `PredictionTime`: Average prediction time
- `MemoryUsage`: Memory footprint (bytes)
- `Convergence`: Whether training converged
- `Epochs`: Number of training epochs

**Example Usage**:
```csharp
var stats = new ModelStats<double>
{
    TrainingError = 0.05,
    ValidationError = 0.08,
    TestError = 0.10,
    ParameterCount = 1000,
    TrainingTime = 5000
};

Console.WriteLine($"Overfitting: {stats.Overfitting}"); // 0.03
```

---

### PredictionStats<T> (`src/Statistics/PredictionStats.cs`)
Statistics about model predictions.

**Key Properties**:
- `TotalPredictions`: Number of predictions made
- `CorrectPredictions`: Number correct (classification)
- `Accuracy`: Correct / Total
- `AveragePredictionTime`: Average time per prediction
- `ConfidenceScores`: Distribution of confidence scores
- `AverageConfidence`: Mean confidence
- `LowConfidenceCount`: Predictions with confidence < threshold
- `HighConfidenceCount`: Predictions with confidence > threshold

**Example Usage**:
```csharp
var stats = new PredictionStats<double>
{
    TotalPredictions = 100,
    CorrectPredictions = 85,
    AveragePredictionTime = 10.5
};

Console.WriteLine($"Accuracy: {stats.Accuracy}"); // 0.85 or 85%
```

---

### Quartile<T> (`src/Statistics/Quartile.cs`)
Calculates quartile values for a dataset.

**Key Properties**:
- `Q0`: Minimum value (0th percentile)
- `Q1`: First quartile (25th percentile)
- `Q2`: Median (50th percentile)
- `Q3`: Third quartile (75th percentile)
- `Q4`: Maximum value (100th percentile)
- `IQR`: Interquartile range (Q3 - Q1)

**Example Usage**:
```csharp
var data = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
var quartile = new Quartile<int>(data);

Console.WriteLine($"Q1: {quartile.Q1}"); // 3
Console.WriteLine($"Median: {quartile.Q2}"); // 5.5
Console.WriteLine($"Q3: {quartile.Q3}"); // 8
Console.WriteLine($"IQR: {quartile.IQR}"); // 5
```

---

## Phase 1: BasicStats Tests

### Test File: `tests/UnitTests/Statistics/BasicStatsTests.cs`

```csharp
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Statistics;

public class BasicStatsTests
{
    [Fact]
    public void Constructor_ValidData_CalculatesCorrectMean()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        // Mean = (1+2+3+4+5)/5 = 15/5 = 3.0
        Assert.Equal(3.0, stats.Mean, precision: 10);
    }

    [Fact]
    public void Variance_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 2.0, 4.0, 6.0, 8.0, 10.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        // Mean = 6.0
        // Variance = [(2-6)² + (4-6)² + (6-6)² + (8-6)² + (10-6)²] / 5
        //          = [16 + 4 + 0 + 4 + 16] / 5 = 40/5 = 8.0
        Assert.Equal(8.0, stats.Variance, precision: 10);
    }

    [Fact]
    public void StandardDeviation_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 2.0, 4.0, 6.0, 8.0, 10.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        // StdDev = sqrt(Variance) = sqrt(8.0) ≈ 2.828
        Assert.Equal(Math.Sqrt(8.0), stats.StandardDeviation, precision: 10);
    }

    [Fact]
    public void MinMax_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 5.0, 1.0, 9.0, 3.0, 7.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        Assert.Equal(1.0, stats.Min);
        Assert.Equal(9.0, stats.Max);
        Assert.Equal(8.0, stats.Range); // 9.0 - 1.0
    }

    [Fact]
    public void Count_ReturnsCorrectNumber()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        Assert.Equal(5, stats.Count);
    }

    [Fact]
    public void Skewness_SymmetricData_ReturnsNearZero()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        // Symmetric distribution should have skewness near 0
        Assert.InRange(stats.Skewness, -0.1, 0.1);
    }

    [Fact]
    public void Quartiles_CalculateCorrectly()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        Assert.InRange(stats.Quartiles.Q1, 2.5, 3.5); // 25th percentile
        Assert.Equal(5.5, stats.Quartiles.Q2, precision: 1); // Median
        Assert.InRange(stats.Quartiles.Q3, 7.5, 8.5); // 75th percentile
    }

    [Fact]
    public void InterquartileRange_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        // IQR = Q3 - Q1
        Assert.InRange(stats.InterquartileRange, 4.0, 6.0);
    }

    [Fact]
    public void Constructor_SingleValue_ReturnsCorrectStats()
    {
        // Arrange
        var data = new[] { 5.0 };

        // Act
        var stats = new BasicStats<double>(data);

        // Assert
        Assert.Equal(5.0, stats.Mean);
        Assert.Equal(0.0, stats.Variance);
        Assert.Equal(0.0, stats.StandardDeviation);
        Assert.Equal(5.0, stats.Min);
        Assert.Equal(5.0, stats.Max);
    }

    [Fact]
    public void Constructor_EmptyData_ThrowsException()
    {
        // Arrange
        var data = Array.Empty<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new BasicStats<double>(data));
    }
}
```

---

## Phase 2: ErrorStats Tests

### Test File: `tests/UnitTests/Statistics/ErrorStatsTests.cs`

```csharp
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Statistics;

public class ErrorStatsTests
{
    [Fact]
    public void MAE_PerfectPredictions_ReturnsZero()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        Assert.Equal(0.0, errors.MAE);
    }

    [Fact]
    public void MAE_CalculatesCorrectly()
    {
        // Arrange
        var actual = new[] { 3.0, -0.5, 2.0, 7.0 };
        var predicted = new[] { 2.5, 0.0, 2.0, 8.0 };

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        // MAE = (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4
        //     = (0.5 + 0.5 + 0 + 1) / 4 = 2/4 = 0.5
        Assert.Equal(0.5, errors.MAE, precision: 10);
    }

    [Fact]
    public void MSE_CalculatesCorrectly()
    {
        // Arrange
        var actual = new[] { 3.0, -0.5, 2.0, 7.0 };
        var predicted = new[] { 2.5, 0.0, 2.0, 8.0 };

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        // MSE = [(3-2.5)² + (-0.5-0)² + (2-2)² + (7-8)²] / 4
        //     = [0.25 + 0.25 + 0 + 1] / 4 = 1.5/4 = 0.375
        Assert.Equal(0.375, errors.MSE, precision: 10);
    }

    [Fact]
    public void RMSE_CalculatesCorrectly()
    {
        // Arrange
        var actual = new[] { 3.0, -0.5, 2.0, 7.0 };
        var predicted = new[] { 2.5, 0.0, 2.0, 8.0 };

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        // RMSE = sqrt(MSE) = sqrt(0.375) ≈ 0.612
        Assert.Equal(Math.Sqrt(0.375), errors.RMSE, precision: 10);
    }

    [Fact]
    public void MeanBiasError_OverestimatingPredictions_ReturnsPositive()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0 };
        var predicted = new[] { 2.0, 3.0, 4.0 }; // All predictions +1 too high

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        // MeanBiasError = [(2-1) + (3-2) + (4-3)] / 3 = 3/3 = 1.0
        Assert.Equal(1.0, errors.MeanBiasError, precision: 10);
    }

    [Fact]
    public void MeanBiasError_UnderestimatingPredictions_ReturnsNegative()
    {
        // Arrange
        var actual = new[] { 2.0, 3.0, 4.0 };
        var predicted = new[] { 1.0, 2.0, 3.0 }; // All predictions -1 too low

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        // MeanBiasError = [(1-2) + (2-3) + (3-4)] / 3 = -3/3 = -1.0
        Assert.Equal(-1.0, errors.MeanBiasError, precision: 10);
    }

    [Fact]
    public void MaxError_FindsLargestError()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0 };
        var predicted = new[] { 1.1, 2.1, 8.0, 4.1 }; // Error of 5 at index 2

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        Assert.Equal(5.0, errors.MaxError, precision: 10);
    }

    [Fact]
    public void R2Score_PerfectPredictions_ReturnsOne()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        Assert.Equal(1.0, errors.R2Score, precision: 10);
    }

    [Fact]
    public void Constructor_MismatchedArrays_ThrowsException()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0 };
        var predicted = new[] { 1.0, 2.0 }; // Wrong length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ErrorStats<double>(actual, predicted));
    }

    [Fact]
    public void MedianAbsoluteError_RobustToOutliers()
    {
        // Arrange
        var actual = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predicted = new[] { 1.1, 2.1, 3.1, 4.1, 100.0 }; // One huge error

        // Act
        var errors = new ErrorStats<double>(actual, predicted);

        // Assert
        // MAE would be very high (19.0)
        // MedianAbsoluteError should be around 0.1 (robust to outlier)
        Assert.True(errors.MedianAbsoluteError < errors.MAE);
    }
}
```

---

## Phase 3: Quartile Tests

### Test File: `tests/UnitTests/Statistics/QuartileTests.cs`

```csharp
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Statistics;

public class QuartileTests
{
    [Fact]
    public void Constructor_ValidData_CalculatesQuartiles()
    {
        // Arrange
        var data = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        // Act
        var quartile = new Quartile<int>(data);

        // Assert
        Assert.Equal(1, quartile.Q0); // Min
        Assert.Equal(10, quartile.Q4); // Max
        Assert.InRange(quartile.Q2, 5, 6); // Median
    }

    [Fact]
    public void IQR_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        // IQR = Q3 - Q1
        double iqr = quartile.IQR;
        Assert.InRange(iqr, 4.0, 6.0);
    }

    [Fact]
    public void Quartiles_OddNumberOfElements_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 1, 2, 3, 4, 5 };

        // Act
        var quartile = new Quartile<int>(data);

        // Assert
        Assert.Equal(3, quartile.Q2); // Median is middle element
    }

    [Fact]
    public void Quartiles_EvenNumberOfElements_CalculatesCorrectly()
    {
        // Arrange
        var data = new[] { 1, 2, 3, 4 };

        // Act
        var quartile = new Quartile<int>(data);

        // Assert
        // Median should be average of 2 and 3
        Assert.InRange(quartile.Q2, 2, 3);
    }

    [Fact]
    public void Constructor_SingleElement_AllQuartilesEqual()
    {
        // Arrange
        var data = new[] { 5.0 };

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        Assert.Equal(5.0, quartile.Q0);
        Assert.Equal(5.0, quartile.Q1);
        Assert.Equal(5.0, quartile.Q2);
        Assert.Equal(5.0, quartile.Q3);
        Assert.Equal(5.0, quartile.Q4);
        Assert.Equal(0.0, quartile.IQR);
    }
}
```

---

## Phase 4: GeneticStats Tests

### Test File: `tests/UnitTests/Statistics/GeneticStatsTests.cs`

```csharp
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Statistics;

public class GeneticStatsTests
{
    [Fact]
    public void Constructor_ValidData_InitializesCorrectly()
    {
        // Arrange
        int generation = 50;
        var fitnessValues = new[] { 0.9, 0.8, 0.7, 0.6, 0.5 };

        // Act
        var stats = new GeneticStats<double>(generation, fitnessValues);

        // Assert
        Assert.Equal(50, stats.Generation);
        Assert.Equal(0.9, stats.BestFitness);
        Assert.Equal(0.5, stats.WorstFitness);
    }

    [Fact]
    public void AverageFitness_CalculatesCorrectly()
    {
        // Arrange
        var fitnessValues = new[] { 0.8, 0.6, 0.4, 0.2 };

        // Act
        var stats = new GeneticStats<double>(1, fitnessValues);

        // Assert
        // Average = (0.8 + 0.6 + 0.4 + 0.2) / 4 = 2.0 / 4 = 0.5
        Assert.Equal(0.5, stats.AverageFitness, precision: 10);
    }

    [Fact]
    public void SelectionPressure_CalculatesCorrectly()
    {
        // Arrange
        var fitnessValues = new[] { 1.0, 0.8, 0.6, 0.4 };

        // Act
        var stats = new GeneticStats<double>(1, fitnessValues);

        // Assert
        // Average = 0.7
        // SelectionPressure = Best / Average = 1.0 / 0.7 ≈ 1.43
        Assert.InRange(stats.SelectionPressure, 1.0, 2.0);
    }

    [Fact]
    public void FitnessVariance_CalculatesCorrectly()
    {
        // Arrange
        var fitnessValues = new[] { 0.9, 0.8, 0.7, 0.6, 0.5 };

        // Act
        var stats = new GeneticStats<double>(1, fitnessValues);

        // Assert
        Assert.True(stats.FitnessVariance > 0);
    }

    [Fact]
    public void DiversityScore_HighDiversity_ReturnsHighValue()
    {
        // Arrange
        var diversePopulation = new[] { 0.1, 0.3, 0.5, 0.7, 0.9 };

        // Act
        var stats = new GeneticStats<double>(1, diversePopulation);

        // Assert
        Assert.True(stats.DiversityScore > 0);
    }
}
```

---

## Phase 5: ModelStats and PredictionStats Tests

### Test File: `tests/UnitTests/Statistics/ModelStatsTests.cs`

```csharp
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Statistics;

public class ModelStatsTests
{
    [Fact]
    public void Overfitting_CalculatesCorrectly()
    {
        // Arrange
        var stats = new ModelStats<double>
        {
            TrainingError = 0.05,
            ValidationError = 0.15
        };

        // Act
        var overfitting = stats.Overfitting;

        // Assert
        // Overfitting = ValidationError - TrainingError = 0.15 - 0.05 = 0.10
        Assert.Equal(0.10, overfitting, precision: 10);
    }

    [Fact]
    public void Properties_CanBeSetAndRetrieved()
    {
        // Act
        var stats = new ModelStats<double>
        {
            TrainingError = 0.05,
            ValidationError = 0.08,
            TestError = 0.10,
            ParameterCount = 1000,
            TrainingTime = 5000,
            Convergence = true,
            Epochs = 100
        };

        // Assert
        Assert.Equal(0.05, stats.TrainingError);
        Assert.Equal(0.08, stats.ValidationError);
        Assert.Equal(0.10, stats.TestError);
        Assert.Equal(1000, stats.ParameterCount);
        Assert.Equal(5000, stats.TrainingTime);
        Assert.True(stats.Convergence);
        Assert.Equal(100, stats.Epochs);
    }
}

public class PredictionStatsTests
{
    [Fact]
    public void Accuracy_CalculatesCorrectly()
    {
        // Arrange
        var stats = new PredictionStats<double>
        {
            TotalPredictions = 100,
            CorrectPredictions = 85
        };

        // Act
        var accuracy = stats.Accuracy;

        // Assert
        // Accuracy = Correct / Total = 85 / 100 = 0.85
        Assert.Equal(0.85, accuracy, precision: 10);
    }

    [Fact]
    public void Accuracy_NoPredictions_ReturnsZero()
    {
        // Arrange
        var stats = new PredictionStats<double>
        {
            TotalPredictions = 0,
            CorrectPredictions = 0
        };

        // Act
        var accuracy = stats.Accuracy;

        // Assert
        Assert.Equal(0.0, accuracy);
    }

    [Fact]
    public void Properties_CanBeSetAndRetrieved()
    {
        // Act
        var stats = new PredictionStats<double>
        {
            TotalPredictions = 100,
            CorrectPredictions = 85,
            AveragePredictionTime = 10.5,
            AverageConfidence = 0.92
        };

        // Assert
        Assert.Equal(100, stats.TotalPredictions);
        Assert.Equal(85, stats.CorrectPredictions);
        Assert.Equal(10.5, stats.AveragePredictionTime);
        Assert.Equal(0.92, stats.AverageConfidence);
    }
}
```

---

## Common Testing Patterns

### Statistical Accuracy Tests

1. **Known Results**
   ```csharp
   // Test with manually calculated expected values
   var data = new[] { 1, 2, 3, 4, 5 };
   Assert.Equal(3.0, stats.Mean); // Verified by hand
   ```

2. **Edge Cases**
   ```csharp
   // Single element
   var single = new[] { 5.0 };
   Assert.Equal(0.0, new BasicStats<double>(single).Variance);
   ```

3. **Perfect Predictions**
   ```csharp
   // Zero error when predictions match actual
   var actual = new[] { 1, 2, 3 };
   var predicted = new[] { 1, 2, 3 };
   Assert.Equal(0.0, new ErrorStats<int>(actual, predicted).MAE);
   ```

### Property-Based Tests

```csharp
[Theory]
[InlineData(new[] { 1.0, 2.0, 3.0 })]
[InlineData(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 })]
public void Mean_AlwaysWithinRange(double[] data)
{
    var stats = new BasicStats<double>(data);
    Assert.InRange(stats.Mean, data.Min(), data.Max());
}
```

---

## Running Tests

```bash
# Run all Statistics tests
dotnet test --filter "FullyQualifiedName~Statistics"

# Run specific test class
dotnet test --filter "FullyQualifiedName~BasicStatsTests"

# Run with coverage
dotnet test /p:CollectCoverage=true
```

---

## Success Criteria

- [ ] BasicStats tests cover: mean, variance, stddev, skewness, kurtosis, quartiles
- [ ] ErrorStats tests cover: MAE, MSE, RMSE, R2, bias, median error
- [ ] Quartile tests cover: Q1, Q2, Q3, IQR, edge cases
- [ ] GeneticStats tests cover: fitness metrics, diversity, selection pressure
- [ ] ModelStats tests cover: overfitting, convergence, timing
- [ ] PredictionStats tests cover: accuracy, confidence
- [ ] All tests pass with green checkmarks
- [ ] Code coverage increases from 0% to >80%
- [ ] Mathematical accuracy verified with known results

---

## Common Pitfalls

1. **Don't forget floating-point precision** - Use `Assert.Equal(expected, actual, precision: 10)`
2. **Don't test with impossible data** - Ensure test data is realistic
3. **Do verify mathematical formulas** - Calculate expected values by hand
4. **Do test edge cases** - Empty, single element, all identical values
5. **Do test error conditions** - Mismatched array lengths, null inputs
6. **Do compare against known results** - Use textbook examples or online calculators

Start with BasicStats and Quartile (fundamental), then ErrorStats (most important for ML), then GeneticStats, ModelStats, and PredictionStats. Build incrementally!
