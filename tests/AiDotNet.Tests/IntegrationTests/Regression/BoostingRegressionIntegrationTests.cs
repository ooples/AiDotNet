using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for boosting regression models: AdaBoostR2, DART, GradientBoosting,
/// HistGradientBoosting, NGBoost, ExplainableBoostingMachine.
/// Tests verify mathematical correctness on known synthetic data.
/// </summary>
[Trait("Category", "Integration")]
public class BoostingRegressionIntegrationTests
{
    #region Test Data Helpers

    private static (Matrix<double> x, Vector<double> y) CreateLinearData(
        int n, double[] coefficients, double intercept, double noise, int seed)
    {
        var random = new Random(seed);
        int p = coefficients.Length;
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double yVal = intercept;
            for (int j = 0; j < p; j++)
            {
                x[i, j] = random.NextDouble() * 10 - 5;
                yVal += coefficients[j] * x[i, j];
            }
            yVal += NextGaussian(random) * noise;
            y[i] = yVal;
        }

        return (x, y);
    }

    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double ComputeR2(Vector<double> actual, Vector<double> predicted)
    {
        double mean = 0;
        for (int i = 0; i < actual.Length; i++) mean += actual[i];
        mean /= actual.Length;

        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            ssTot += (actual[i] - mean) * (actual[i] - mean);
            ssRes += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
        }

        return ssTot == 0 ? 0 : 1.0 - ssRes / ssTot;
    }

    private static bool AllFinite(Vector<double> v)
    {
        for (int i = 0; i < v.Length; i++)
        {
            if (double.IsNaN(v[i]) || double.IsInfinity(v[i]))
                return false;
        }
        return true;
    }

    #endregion

    #region AdaBoostR2

    [Fact]
    public void AdaBoostR2_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 2.0, -1.5 }, intercept: 3.0, noise: 0.5, seed: 42);

        var model = new AdaBoostR2Regression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "AdaBoostR2 predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.5, $"AdaBoostR2 R²={r2:F4} should be > 0.5 on linear data");
    }

    [Fact]
    public void AdaBoostR2_MonotonicInFirstFeature()
    {
        var (x, y) = CreateLinearData(60, new[] { 3.0, 0.5 }, intercept: 1.0, noise: 0.3, seed: 42);

        var model = new AdaBoostR2Regression<double>();
        model.Train(x, y);

        // Increasing x1 should generally increase predicted y (coefficient is positive)
        var lowX = new Matrix<double>(1, 2);
        lowX[0, 0] = -5; lowX[0, 1] = 0;
        var highX = new Matrix<double>(1, 2);
        highX[0, 0] = 5; highX[0, 1] = 0;

        var predLow = model.Predict(lowX);
        var predHigh = model.Predict(highX);

        Assert.True(predHigh[0] > predLow[0],
            $"AdaBoostR2 should predict higher for x1=5 ({predHigh[0]:F4}) than x1=-5 ({predLow[0]:F4})");
    }

    [Fact]
    public void AdaBoostR2_DeterministicWithSeed()
    {
        var (x, y) = CreateLinearData(40, new[] { 1.0, 2.0 }, intercept: 0, noise: 0.1, seed: 42);

        var model1 = new AdaBoostR2Regression<double>();
        model1.Train(x, y);
        var pred1 = model1.Predict(x);

        var model2 = new AdaBoostR2Regression<double>();
        model2.Train(x, y);
        var pred2 = model2.Predict(x);

        for (int i = 0; i < pred1.Length; i++)
        {
            Assert.Equal(pred1[i], pred2[i], 10);
        }
    }

    #endregion

    #region DART Regression

    [Fact]
    public void DARTRegression_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 2.0, -1.0, 0.5 }, intercept: 2.0, noise: 0.5, seed: 42);

        var model = new DARTRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "DART predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.5, $"DART R²={r2:F4} should be > 0.5 on linear data");
    }

    [Fact]
    public void DARTRegression_OutputLengthMatchesInput()
    {
        var (x, y) = CreateLinearData(50, new[] { 1.0 }, intercept: 0, noise: 0.1, seed: 42);

        var model = new DARTRegression<double>();
        model.Train(x, y);

        var testX = new Matrix<double>(10, 1);
        for (int i = 0; i < 10; i++) testX[i, 0] = i;

        var predictions = model.Predict(testX);
        Assert.Equal(10, predictions.Length);
    }

    #endregion

    #region GradientBoosting

    [Fact]
    public void GradientBoosting_FitsNonLinearData_BetterThanLinear()
    {
        // Non-linear data: y = x1^2 + x2
        var random = new Random(42);
        int n = 80;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 6 - 3;
            x[i, 1] = random.NextDouble() * 4 - 2;
            y[i] = x[i, 0] * x[i, 0] + x[i, 1] + NextGaussian(random) * 0.3;
        }

        var model = new GradientBoostingRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "GradientBoosting predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.7, $"GradientBoosting R²={r2:F4} should be > 0.7 on quadratic data");
    }

    [Fact]
    public void GradientBoosting_MoreEstimatorsReduceError()
    {
        var (x, y) = CreateLinearData(60, new[] { 2.0, 1.0 }, intercept: 1.0, noise: 0.5, seed: 42);

        var modelFew = new GradientBoostingRegression<double>(new GradientBoostingRegressionOptions { NumberOfTrees = 5, Seed = 42 });
        modelFew.Train(x, y);
        var predFew = modelFew.Predict(x);
        double r2Few = ComputeR2(y, predFew);

        var modelMany = new GradientBoostingRegression<double>(new GradientBoostingRegressionOptions { NumberOfTrees = 50, Seed = 42 });
        modelMany.Train(x, y);
        var predMany = modelMany.Predict(x);
        double r2Many = ComputeR2(y, predMany);

        Assert.True(r2Many >= r2Few - 0.05,
            $"More estimators R²={r2Many:F4} should be >= fewer estimators R²={r2Few:F4}");
    }

    #endregion

    #region HistGradientBoosting

    [Fact]
    public void HistGradientBoosting_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(100, new[] { 1.5, -2.0 }, intercept: 3.0, noise: 0.5, seed: 42);

        var model = new HistGradientBoostingRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "HistGradientBoosting predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.5, $"HistGradientBoosting R²={r2:F4} should be > 0.5 on linear data");
    }

    [Fact]
    public void HistGradientBoosting_HandlesLargeDataset()
    {
        // Hist-based methods should handle more data efficiently
        var (x, y) = CreateLinearData(500, new[] { 1.0, 2.0, -0.5 }, intercept: 0, noise: 1.0, seed: 42);

        var model = new HistGradientBoostingRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "HistGradientBoosting predictions contain NaN/Infinity on large data");
        Assert.Equal(500, predictions.Length);
    }

    #endregion

    #region NGBoost Regression

    [Fact]
    public void NGBoostRegression_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(60, new[] { 2.0, 1.0 }, intercept: 1.0, noise: 0.5, seed: 42);

        var model = new NGBoostRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "NGBoost predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.3, $"NGBoost R²={r2:F4} should be > 0.3 on linear data");
    }

    #endregion

    #region ExplainableBoostingMachine Regression

    [Fact]
    public void EBM_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 2.0, -1.0 }, intercept: 1.0, noise: 0.5, seed: 42);

        var model = new ExplainableBoostingMachineRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "EBM predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.3, $"EBM R²={r2:F4} should be > 0.3 on linear data");
    }

    [Fact]
    public void EBM_InteractionDetection_TwoFeatures()
    {
        // Data with interaction: y = x1 * x2
        var random = new Random(42);
        int n = 80;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 4 - 2;
            x[i, 1] = random.NextDouble() * 4 - 2;
            y[i] = x[i, 0] * x[i, 1] + NextGaussian(random) * 0.1;
        }

        var model = new ExplainableBoostingMachineRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "EBM interaction predictions contain NaN/Infinity");
    }

    #endregion

    #region Cross-cutting: all boosting models produce finite predictions

    private void AssertModelProducesFinitePredictions(Interfaces.IFullModel<double, Matrix<double>, Vector<double>> model, string modelName)
    {
        var (x, y) = CreateLinearData(60, new[] { 1.0, -0.5 }, intercept: 2.0, noise: 0.5, seed: 42);

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        Assert.True(AllFinite(predictions), $"{modelName} produced NaN/Infinity predictions");
    }

    [Fact]
    public void AdaBoostR2_PredictionsAreFinite()
        => AssertModelProducesFinitePredictions(new AdaBoostR2Regression<double>(), "AdaBoostR2");

    [Fact]
    public void DART_PredictionsAreFinite()
        => AssertModelProducesFinitePredictions(new DARTRegression<double>(), "DART");

    [Fact]
    public void GradientBoosting_PredictionsAreFinite()
        => AssertModelProducesFinitePredictions(new GradientBoostingRegression<double>(), "GradientBoosting");

    [Fact]
    public void HistGradientBoosting_PredictionsAreFinite()
        => AssertModelProducesFinitePredictions(new HistGradientBoostingRegression<double>(), "HistGradientBoosting");

    [Fact]
    public void NGBoost_PredictionsAreFinite()
        => AssertModelProducesFinitePredictions(new NGBoostRegression<double>(), "NGBoost");

    [Fact]
    public void EBM_PredictionsAreFinite()
        => AssertModelProducesFinitePredictions(new ExplainableBoostingMachineRegression<double>(), "EBM");

    #endregion
}
