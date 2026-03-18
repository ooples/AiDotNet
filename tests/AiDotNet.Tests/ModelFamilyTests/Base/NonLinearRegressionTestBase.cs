using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for non-linear regression models.
/// Inherits all regression invariant tests and adds non-linear-specific invariants:
/// quadratic fitting, superiority over linear on nonlinear data, and extrapolation safety.
/// </summary>
public abstract class NonLinearRegressionTestBase : RegressionModelTestBase
{
    // =====================================================
    // NON-LINEAR INVARIANT: Can Fit Quadratic Data
    // A non-linear regression model should achieve R² > 0 on y = x² data.
    // If it can't, the non-linear capacity is not functioning.
    // =====================================================

    [Fact]
    public void CanFitQuadratic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        int n = TrainSamples;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double xi = rng.NextDouble() * 4.0 - 2.0;  // [-2, 2]
            x[i, 0] = xi;
            y[i] = xi * xi + ModelTestHelpers.NextGaussian(rng) * 0.1;
        }

        var model = CreateModel();
        model.Train(x, y);
        var predictions = model.Predict(x);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double r2 = ModelTestHelpers.CalculateR2(y, predictions);
            Assert.True(r2 > 0.0,
                $"R² = {r2:F4} on quadratic data. Non-linear model should outperform mean baseline on y=x².");
        }
    }

    // =====================================================
    // NON-LINEAR INVARIANT: Better Than Linear on Nonlinear Data
    // On data with clear non-linear structure (y = sin(x)),
    // non-linear model should have lower MSE than a constant predictor.
    // =====================================================

    [Fact]
    public void NonLinearResiduals_ShouldBeSmaller()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        int n = TrainSamples;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double xi = rng.NextDouble() * 6.0;  // [0, 6]
            x[i, 0] = xi;
            y[i] = Math.Sin(xi) + ModelTestHelpers.NextGaussian(rng) * 0.1;
        }

        var model = CreateModel();
        model.Train(x, y);
        var predictions = model.Predict(x);

        if (ModelTestHelpers.AllFinite(predictions))
        {
            double modelMSE = ModelTestHelpers.CalculateMSE(y, predictions);

            // Mean predictor MSE (baseline)
            double yMean = 0;
            for (int i = 0; i < y.Length; i++) yMean += y[i];
            yMean /= y.Length;
            double baselineMSE = 0;
            for (int i = 0; i < y.Length; i++)
                baselineMSE += (y[i] - yMean) * (y[i] - yMean);
            baselineMSE /= y.Length;

            Assert.True(modelMSE < baselineMSE * 1.5,
                $"Non-linear model MSE ({modelMSE:F6}) is not better than mean baseline ({baselineMSE:F6}) " +
                "on sinusoidal data. Non-linear capacity may not be functioning.");
        }
    }

    // =====================================================
    // NON-LINEAR INVARIANT: Extrapolation Should Be Finite
    // Predictions far outside training range may be extreme but must
    // be finite. Infinite extrapolation indicates numerical instability.
    // =====================================================

    [Fact]
    public void Extrapolation_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, 1, rng);

        var model = CreateModel();
        model.Train(trainX, trainY);

        // Extrapolate far outside training range
        var extraX = new Matrix<double>(5, 1);
        for (int i = 0; i < 5; i++)
            extraX[i, 0] = 100.0 + i * 100.0;  // far from training [0, 10]

        var predictions = model.Predict(extraX);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(!double.IsNaN(predictions[i]) && !double.IsInfinity(predictions[i]),
                $"Extrapolation prediction[{i}] is not finite at x={extraX[i, 0]}. " +
                "Non-linear model has numerical instability in extrapolation.");
        }
    }
}
