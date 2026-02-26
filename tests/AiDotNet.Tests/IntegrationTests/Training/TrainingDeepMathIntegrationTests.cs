using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Training.Factories;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Deep integration tests for Training:
/// LossFunctionFactory loss type coverage and parameter variants,
/// TrainingResult data model, loss function mathematical properties.
/// </summary>
public class TrainingDeepMathIntegrationTests
{
    // ============================
    // LossType Enum: Coverage
    // ============================

    [Fact]
    public void LossType_HasExpectedCount()
    {
        var values = Enum.GetValues<LossType>();
        // At minimum the factory handles 33 types + a few that require specialized constructors
        Assert.True(values.Length >= 33, $"LossType should have at least 33 values, found {values.Length}");
    }

    // ============================
    // Loss Functions: MSE Mathematical Properties
    // ============================

    [Fact]
    public void MSE_LossFunction_PerfectPrediction_ZeroLoss()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.Equal(0.0, result, 1e-10);
    }

    [Fact]
    public void MSE_LossFunction_KnownValues()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        // MSE = mean((predicted - actual)^2)
        // (1-2)^2 + (3-4)^2 + (5-6)^2 = 1 + 1 + 1 = 3, mean = 1.0
        var predicted = new Vector<double>(new double[] { 1.0, 3.0, 5.0 });
        var actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.Equal(1.0, result, 1e-10);
    }

    [Fact]
    public void MSE_LossFunction_Symmetric()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        var a = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

        double lossAB = loss.CalculateLoss(a, b);
        double lossBA = loss.CalculateLoss(b, a);
        Assert.Equal(lossAB, lossBA, 1e-10);
    }

    // ============================
    // Loss Functions: MAE Mathematical Properties
    // ============================

    [Fact]
    public void MAE_LossFunction_PerfectPrediction_ZeroLoss()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanAbsoluteError);
        var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.Equal(0.0, result, 1e-10);
    }

    [Fact]
    public void MAE_LossFunction_KnownValues()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanAbsoluteError);
        // MAE = mean(|predicted - actual|)
        // |1-2| + |3-5| + |5-3| = 1 + 2 + 2 = 5, mean = 5/3
        var predicted = new Vector<double>(new double[] { 1.0, 3.0, 5.0 });
        var actual = new Vector<double>(new double[] { 2.0, 5.0, 3.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.Equal(5.0 / 3.0, result, 1e-10);
    }

    // ============================
    // Loss Functions: RMSE = sqrt(MSE)
    // ============================

    [Fact]
    public void RMSE_LossFunction_EqualsSquareRootOfMSE()
    {
        var mse = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        var rmse = LossFunctionFactory<double>.Create(LossType.RootMeanSquaredError);
        var predicted = new Vector<double>(new double[] { 1.0, 3.0, 5.0 });
        var actual = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });

        double mseValue = mse.CalculateLoss(predicted, actual);
        double rmseValue = rmse.CalculateLoss(predicted, actual);

        Assert.Equal(Math.Sqrt(mseValue), rmseValue, 1e-10);
    }

    // ============================
    // Loss Functions: Hinge Loss Properties
    // ============================

    [Fact]
    public void Hinge_PerfectClassification_ZeroLoss()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.Hinge);
        // Hinge: max(0, 1 - y*f(x))
        // y=1, f(x)=1 -> max(0, 1-1) = 0
        var predicted = new Vector<double>(new double[] { 1.0 });
        var actual = new Vector<double>(new double[] { 1.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.True(result <= 1e-10, $"Hinge loss for perfect prediction should be ~0, got {result}");
    }

    // ============================
    // Loss Functions: LogCosh Properties
    // ============================

    [Fact]
    public void LogCosh_PerfectPrediction_ZeroLoss()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.LogCosh);
        var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.Equal(0.0, result, 1e-10);
    }

    [Fact]
    public void LogCosh_AlwaysNonNegative()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.LogCosh);
        var predicted = new Vector<double>(new double[] { 10.0, -5.0, 3.0 });
        var actual = new Vector<double>(new double[] { -2.0, 8.0, 0.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.True(result >= 0, $"LogCosh loss should be non-negative, got {result}");
    }

    // ============================
    // Loss Functions: MSE Always Non-Negative
    // ============================

    [Theory]
    [InlineData(LossType.MeanSquaredError)]
    [InlineData(LossType.MeanAbsoluteError)]
    [InlineData(LossType.RootMeanSquaredError)]
    [InlineData(LossType.LogCosh)]
    public void RegressionLoss_AlwaysNonNegative(LossType lossType)
    {
        var loss = LossFunctionFactory<double>.Create(lossType);
        var predicted = new Vector<double>(new double[] { 5.5, -3.2, 0.0, 100.0 });
        var actual = new Vector<double>(new double[] { -1.0, 7.7, 0.5, -50.0 });

        double result = loss.CalculateLoss(predicted, actual);
        Assert.True(result >= 0, $"{lossType} loss should be non-negative, got {result}");
    }

    // ============================
    // Loss Functions: Float Type
    // ============================

    [Theory]
    [InlineData(LossType.MeanSquaredError)]
    [InlineData(LossType.MeanAbsoluteError)]
    [InlineData(LossType.LogCosh)]
    public void LossFactory_FloatType_Works(LossType lossType)
    {
        var loss = LossFunctionFactory<float>.Create(lossType);
        Assert.NotNull(loss);

        var predicted = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });
        var actual = new Vector<float>(new float[] { 1.5f, 2.5f, 3.5f });

        float result = loss.CalculateLoss(predicted, actual);
        Assert.True(result >= 0, $"Float {lossType} loss should be non-negative");
    }

    // ============================
    // Loss Functions: MSE Derivative
    // ============================

    [Fact]
    public void MSE_Derivative_PerfectPrediction_ZeroDerivative()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var derivative = loss.CalculateDerivative(predicted, actual);
        for (int i = 0; i < derivative.Length; i++)
        {
            Assert.Equal(0.0, derivative[i], 1e-10);
        }
    }

    [Fact]
    public void MSE_Derivative_PositiveError_PositiveDerivative()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        // When predicted > actual, gradient should be positive
        var predicted = new Vector<double>(new double[] { 5.0 });
        var actual = new Vector<double>(new double[] { 3.0 });

        var derivative = loss.CalculateDerivative(predicted, actual);
        Assert.True(derivative[0] > 0, $"Derivative should be positive when predicted > actual, got {derivative[0]}");
    }

    [Fact]
    public void MSE_Derivative_NegativeError_NegativeDerivative()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        // When predicted < actual, gradient should be negative
        var predicted = new Vector<double>(new double[] { 1.0 });
        var actual = new Vector<double>(new double[] { 5.0 });

        var derivative = loss.CalculateDerivative(predicted, actual);
        Assert.True(derivative[0] < 0, $"Derivative should be negative when predicted < actual, got {derivative[0]}");
    }

    // ============================
    // Loss Functions: MSE Derivative Length
    // ============================

    [Fact]
    public void MSE_DerivativeLength_MatchesPredictedLength()
    {
        var loss = LossFunctionFactory<double>.Create(LossType.MeanSquaredError);
        var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new double[] { 1.5, 2.5, 3.5, 4.5 });

        var derivative = loss.CalculateDerivative(predicted, actual);
        Assert.Equal(predicted.Length, derivative.Length);
    }
}
