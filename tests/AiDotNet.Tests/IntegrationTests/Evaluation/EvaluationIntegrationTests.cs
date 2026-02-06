using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Comprehensive integration tests for the Evaluation module.
/// Tests PredictionType enum, PredictionStatsOptions, and prediction type inference.
/// </summary>
public class EvaluationIntegrationTests
{
    #region PredictionType Enum Tests

    [Fact]
    public void PredictionType_HasExpectedValues()
    {
        // Assert
        var values = (PredictionType[])Enum.GetValues(typeof(PredictionType));
        Assert.Contains(PredictionType.BinaryClassification, values);
        Assert.Contains(PredictionType.Regression, values);
        Assert.Contains(PredictionType.MultiClass, values);
        Assert.Contains(PredictionType.MultiLabel, values);
    }

    [Fact]
    public void PredictionType_HasFourValues()
    {
        // Assert
        var values = (PredictionType[])Enum.GetValues(typeof(PredictionType));
        Assert.Equal(4, values.Length);
    }

    [Theory]
    [InlineData(PredictionType.BinaryClassification)]
    [InlineData(PredictionType.Regression)]
    [InlineData(PredictionType.MultiClass)]
    [InlineData(PredictionType.MultiLabel)]
    public void PredictionType_CanBeUsedInSwitch(PredictionType predictionType)
    {
        // Act
        var description = predictionType switch
        {
            PredictionType.BinaryClassification => "Two classes",
            PredictionType.Regression => "Continuous values",
            PredictionType.MultiClass => "Multiple classes",
            PredictionType.MultiLabel => "Multiple labels",
            _ => "Unknown"
        };

        // Assert
        Assert.NotEqual("Unknown", description);
    }

    #endregion

    #region PredictionStatsOptions Tests

    [Fact]
    public void PredictionStatsOptions_DefaultValues()
    {
        // Arrange & Act
        var options = new PredictionStatsOptions();

        // Assert - check default values exist
        Assert.NotNull(options);
        Assert.True(options.ConfidenceLevel > 0);
        Assert.True(options.LearningCurveSteps > 0);
    }

    [Fact]
    public void PredictionStatsOptions_SetConfidenceLevel()
    {
        // Arrange & Act
        var options = new PredictionStatsOptions
        {
            ConfidenceLevel = 0.99
        };

        // Assert
        Assert.Equal(0.99, options.ConfidenceLevel);
    }

    [Fact]
    public void PredictionStatsOptions_SetLearningCurveSteps()
    {
        // Arrange & Act
        var options = new PredictionStatsOptions
        {
            LearningCurveSteps = 50
        };

        // Assert
        Assert.Equal(50, options.LearningCurveSteps);
    }

    #endregion

    #region PredictionType Inference via Reflection Tests

    // Note: PredictionTypeInference is internal, so we test via reflection
    // This allows comprehensive testing of the inference logic

    [Fact]
    public void PredictionTypeInference_EmptyVector_ReturnsRegression()
    {
        // Arrange
        var emptyVector = new Vector<double>(Array.Empty<double>());
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { emptyVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_BinaryClassification_ZeroOne()
    {
        // Arrange - binary labels 0 and 1
        var binaryVector = new Vector<double>(new double[] { 0, 1, 0, 1, 0, 1, 1, 0, 1, 0 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { binaryVector });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_BinaryClassification_AllZeros()
    {
        // Arrange - all zeros (still binary, just one class present)
        var allZeros = new Vector<double>(new double[] { 0, 0, 0, 0, 0 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { allZeros });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_BinaryClassification_AllOnes()
    {
        // Arrange - all ones (still binary, just one class present)
        var allOnes = new Vector<double>(new double[] { 1, 1, 1, 1, 1 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { allOnes });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_MultiClass_IntegerLabels()
    {
        // Arrange - multi-class labels 0, 1, 2, 3
        var multiClassVector = new Vector<double>(new double[] { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { multiClassVector });

        // Assert
        Assert.Equal(PredictionType.MultiClass, result);
    }

    [Fact]
    public void PredictionTypeInference_Regression_ContinuousValues()
    {
        // Arrange - continuous values
        var regressionVector = new Vector<double>(new double[] { 1.5, 2.7, 3.14159, 4.2, 5.9 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { regressionVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_Regression_NaN()
    {
        // Arrange - contains NaN
        var nanVector = new Vector<double>(new double[] { 1.0, double.NaN, 3.0 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { nanVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_Regression_Infinity()
    {
        // Arrange - contains infinity
        var infVector = new Vector<double>(new double[] { 1.0, double.PositiveInfinity, 3.0 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { infVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_Regression_NegativeInfinity()
    {
        // Arrange - contains negative infinity
        var negInfVector = new Vector<double>(new double[] { 1.0, double.NegativeInfinity, 3.0 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { negInfVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_Regression_HighUniqueRatio()
    {
        // Arrange - many unique integer values (high unique ratio -> regression)
        var highUniqueVector = new Vector<double>(Enumerable.Range(1, 100).Select(x => (double)x).ToArray());
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { highUniqueVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_Float_BinaryClassification()
    {
        // Arrange - binary labels with float type
        var binaryVector = new Vector<float>(new float[] { 0f, 1f, 0f, 1f, 0f, 1f });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(float))
            .Invoke(null, new object[] { binaryVector });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_Float_Regression()
    {
        // Arrange - continuous values with float type
        var regressionVector = new Vector<float>(new float[] { 1.5f, 2.7f, 3.14f, 4.2f, 5.9f });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(float))
            .Invoke(null, new object[] { regressionVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    #endregion

    #region PredictionTypeInference InferFromTargets Tests

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Vector_BinaryClassification()
    {
        // Arrange - binary labels
        var binaryVector = new Vector<double>(new double[] { 0, 1, 0, 1, 0, 1 });
        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Vector<double>))
            .Invoke(null, new object[] { binaryVector });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Vector_Regression()
    {
        // Arrange - continuous values
        var regressionVector = new Vector<double>(new double[] { 1.5, 2.7, 3.14, 4.2 });
        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Vector<double>))
            .Invoke(null, new object[] { regressionVector });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Matrix_SingleColumn()
    {
        // Arrange - single column matrix with binary labels
        var matrix = new Matrix<double>(6, 1);
        matrix[0, 0] = 0;
        matrix[1, 0] = 1;
        matrix[2, 0] = 0;
        matrix[3, 0] = 1;
        matrix[4, 0] = 0;
        matrix[5, 0] = 1;
        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Matrix<double>))
            .Invoke(null, new object[] { matrix });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Matrix_MultiColumn_MultiClass()
    {
        // Arrange - multi-column matrix with one-hot encoding (row sums to 1)
        var matrix = new Matrix<double>(4, 3);
        // Row 0: class 0
        matrix[0, 0] = 1.0; matrix[0, 1] = 0.0; matrix[0, 2] = 0.0;
        // Row 1: class 1
        matrix[1, 0] = 0.0; matrix[1, 1] = 1.0; matrix[1, 2] = 0.0;
        // Row 2: class 2
        matrix[2, 0] = 0.0; matrix[2, 1] = 0.0; matrix[2, 2] = 1.0;
        // Row 3: class 0
        matrix[3, 0] = 1.0; matrix[3, 1] = 0.0; matrix[3, 2] = 0.0;

        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Matrix<double>))
            .Invoke(null, new object[] { matrix });

        // Assert - should be MultiClass (one-hot encoding)
        Assert.Equal(PredictionType.MultiClass, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Matrix_MultiColumn_MultiLabel()
    {
        // Arrange - multi-column matrix with multi-label encoding (row can have multiple 1s)
        var matrix = new Matrix<double>(4, 3);
        // Row 0: labels 0 and 1
        matrix[0, 0] = 1.0; matrix[0, 1] = 1.0; matrix[0, 2] = 0.0;
        // Row 1: label 2 only
        matrix[1, 0] = 0.0; matrix[1, 1] = 0.0; matrix[1, 2] = 1.0;
        // Row 2: labels 0, 1, and 2
        matrix[2, 0] = 1.0; matrix[2, 1] = 1.0; matrix[2, 2] = 1.0;
        // Row 3: label 0 only
        matrix[3, 0] = 1.0; matrix[3, 1] = 0.0; matrix[3, 2] = 0.0;

        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Matrix<double>))
            .Invoke(null, new object[] { matrix });

        // Assert - should be MultiLabel (multiple labels per row)
        Assert.Equal(PredictionType.MultiLabel, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Matrix_Regression()
    {
        // Arrange - matrix with values outside [0, 1] range
        var matrix = new Matrix<double>(3, 2);
        matrix[0, 0] = 1.5; matrix[0, 1] = 2.7;
        matrix[1, 0] = 3.2; matrix[1, 1] = 4.8;
        matrix[2, 0] = 5.1; matrix[2, 1] = 6.3;

        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Matrix<double>))
            .Invoke(null, new object[] { matrix });

        // Assert - should be Regression (values outside [0, 1])
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Tensor_Rank1_BinaryClassification()
    {
        // Arrange - 1D tensor with binary labels
        var tensor = new Tensor<double>(new[] { 6 });
        tensor[0] = 0; tensor[1] = 1; tensor[2] = 0;
        tensor[3] = 1; tensor[4] = 0; tensor[5] = 1;

        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Tensor<double>))
            .Invoke(null, new object[] { tensor });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_Tensor_Rank2_SingleColumn()
    {
        // Arrange - 2D tensor with single column (binary labels)
        var tensor = new Tensor<double>(new[] { 6, 1 });
        tensor[0] = 0; tensor[1] = 1; tensor[2] = 0;
        tensor[3] = 1; tensor[4] = 0; tensor[5] = 1;

        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Tensor<double>))
            .Invoke(null, new object[] { tensor });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_NullTensor_ReturnsRegression()
    {
        // Arrange
        Tensor<double>? nullTensor = null;
        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Tensor<double>))
            .Invoke(null, new object?[] { nullTensor });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_InferFromTargets_EmptyMatrix_ReturnsRegression()
    {
        // Arrange
        var emptyMatrix = new Matrix<double>(0, 0);
        var inferFromTargetsMethod = GetInferFromTargetsMethod();

        // Act
        var result = inferFromTargetsMethod.MakeGenericMethod(typeof(double), typeof(Matrix<double>))
            .Invoke(null, new object[] { emptyMatrix });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void PredictionTypeInference_SingleElement_Binary()
    {
        // Arrange - single element (treated as binary if 0 or 1)
        var singleOne = new Vector<double>(new double[] { 1 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { singleOne });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_SingleElement_Regression()
    {
        // Arrange - single element with non-integer value
        var singleNonInt = new Vector<double>(new double[] { 1.5 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { singleNonInt });

        // Assert
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_NearInteger_BinaryClassification()
    {
        // Arrange - values very close to integers (within epsilon)
        var nearIntegers = new Vector<double>(new double[] { 0.0000000001, 0.9999999999, 0.0000000001 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { nearIntegers });

        // Assert
        Assert.Equal(PredictionType.BinaryClassification, result);
    }

    [Fact]
    public void PredictionTypeInference_NonContiguousLabels_MayBeRegression()
    {
        // Arrange - non-contiguous labels with high unique ratio
        var nonContiguous = new Vector<double>(Enumerable.Range(0, 50).Select(x => (double)(x * 10)).ToArray());
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { nonContiguous });

        // Assert - high unique ratio -> regression
        Assert.Equal(PredictionType.Regression, result);
    }

    [Fact]
    public void PredictionTypeInference_LowUniqueRatio_MultiClass()
    {
        // Arrange - repeating labels with low unique ratio
        var lowUniqueRatio = new Vector<double>(new double[] { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 });
        var inferMethod = GetInferMethod();

        // Act
        var result = inferMethod.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { lowUniqueRatio });

        // Assert
        Assert.Equal(PredictionType.MultiClass, result);
    }

    #endregion

    #region Helper Methods

    private static System.Reflection.MethodInfo GetInferMethod()
    {
        // Use ErrorStats from the same assembly to get the PredictionTypeInference type
        var inferenceType = typeof(AiDotNet.Statistics.ErrorStats<double>).Assembly
            .GetType("AiDotNet.Evaluation.PredictionTypeInference");

        if (inferenceType == null)
        {
            throw new InvalidOperationException("Could not find PredictionTypeInference type");
        }

        var method = inferenceType.GetMethod("Infer",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);

        if (method == null)
        {
            throw new InvalidOperationException("Could not find Infer method");
        }

        return method;
    }

    private static System.Reflection.MethodInfo GetInferFromTargetsMethod()
    {
        // Use ErrorStats from the same assembly to get the PredictionTypeInference type
        var inferenceType = typeof(AiDotNet.Statistics.ErrorStats<double>).Assembly
            .GetType("AiDotNet.Evaluation.PredictionTypeInference");

        if (inferenceType == null)
        {
            throw new InvalidOperationException("Could not find PredictionTypeInference type");
        }

        var method = inferenceType.GetMethod("InferFromTargets",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);

        if (method == null)
        {
            throw new InvalidOperationException("Could not find InferFromTargets method");
        }

        return method;
    }

    #endregion
}
