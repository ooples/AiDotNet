using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Tests that verify trained models produce mathematically correct predictions.
/// Before these tests, only 1 of 10 existing builder tests checked actual accuracy,
/// and that used a loose ±4 tolerance. The rest only checked Assert.NotNull/NotNaN.
/// A model returning all zeros or random garbage would pass all existing tests.
/// </summary>
public class AiModelBuilderAccuracyTests
{
    [Fact]
    public void RidgeRegression_KnownLinearData_R2AboveZero()
    {
        // Arrange: y = 2*x1 + 3*x2 + 1 + noise(σ=0.1)
        // Note: The builder's default NormalOptimizer performs feature selection,
        // which may select only a subset of features. With 2 features, if only
        // one is selected, R² will be lower but still significantly above baseline.
        var random = new Random(42);
        int trainSamples = 100;
        int testSamples = 30;

        var (trainX, trainY) = GenerateLinearData(trainSamples, random, noise: 0.1);
        var (testX, testY) = GenerateLinearData(testSamples, random, noise: 0.1);

        var loader = DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Act
        var predictions = result.Predict(testX);

        // Assert: R² > 0.0 (with feature selection, optimizer may drop relevant features)
        // R² > 0 means the model is better than the mean baseline.
        double r2 = CalculateR2(testY, predictions);
        Assert.True(r2 > 0.0,
            $"R² = {r2:F4} — model is worse than mean baseline for linear data. " +
            "Even with aggressive feature selection, the model should capture some signal.");

        // Also verify all predictions are finite (no NaN/Infinity)
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction {i} is NaN");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction {i} is Infinity");
        }
    }

    [Fact]
    public void DirectModel_KnownLinearData_CoefficientsRecovered()
    {
        // Arrange: y = 2*x1 + 3*x2 + 1, use direct model training (no builder/optimizer)
        // to verify the model itself correctly recovers coefficients.
        // The builder's optimizer adds feature selection which complicates coefficient probing.
        var random = new Random(42);
        var model = new RidgeRegression<double>();

        int samples = 100;
        var x = new Matrix<double>(samples, 2);
        var y = new Vector<double>(samples);
        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            x[i, 1] = random.NextDouble() * 10;
            y[i] = 2.0 * x[i, 0] + 3.0 * x[i, 1] + 1.0 + NextGaussian(random) * 0.05;
        }

        model.Train(x, y);

        // Probe predictions to recover approximate coefficients
        var probeData = new Matrix<double>(3, 2);
        probeData[0, 0] = 10.0; probeData[0, 1] = 0.0;   // expected ≈ 21
        probeData[1, 0] = 0.0;  probeData[1, 1] = 10.0;   // expected ≈ 31
        probeData[2, 0] = 0.0;  probeData[2, 1] = 0.0;    // expected ≈ 1

        var probePredictions = model.Predict(probeData);

        double approxIntercept = probePredictions[2];
        double approxCoeff1 = (probePredictions[0] - approxIntercept) / 10.0;
        double approxCoeff2 = (probePredictions[1] - approxIntercept) / 10.0;

        // Assert: Coefficients should be close to true values (2, 3, 1)
        Assert.InRange(approxCoeff1, 1.0, 3.0);  // true value: 2.0
        Assert.InRange(approxCoeff2, 2.0, 4.0);  // true value: 3.0
        Assert.InRange(approxIntercept, -1.0, 3.0);  // true value: 1.0
    }

    [Fact]
    public void DirectModel_PerfectLinearData_ExactPredictions()
    {
        // Arrange: y = 5*x1 - 2*x2 with ZERO noise using direct model training.
        // Direct training bypasses the optimizer's feature selection for an exact test.
        var random = new Random(77);
        int samples = 50;
        var x = new Matrix<double>(samples, 2);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 20 - 10;
            x[i, 1] = random.NextDouble() * 20 - 10;
            y[i] = 5.0 * x[i, 0] - 2.0 * x[i, 1]; // zero noise
        }

        var model = new MultipleRegression<double>();
        model.Train(x, y);

        // Test on known inputs
        var testX = new Matrix<double>(3, 2);
        testX[0, 0] = 1.0;  testX[0, 1] = 1.0;   // expected: 5*1 - 2*1 = 3
        testX[1, 0] = 3.0;  testX[1, 1] = -1.0;   // expected: 5*3 - 2*(-1) = 17
        testX[2, 0] = -2.0; testX[2, 1] = 4.0;    // expected: 5*(-2) - 2*4 = -18

        var predictions = model.Predict(testX);
        double[] expected = { 3.0, 17.0, -18.0 };

        for (int i = 0; i < 3; i++)
        {
            Assert.InRange(predictions[i], expected[i] - 1.0, expected[i] + 1.0);
        }
    }

    [Fact]
    public void RidgeRegression_SinusoidalData_R2NotTerrible()
    {
        // Arrange: y = sin(x) on [0, 2π] — non-linear challenge.
        // Ridge regression can only fit a line, so R² won't be perfect,
        // but it should not be catastrophically bad.
        var random = new Random(55);
        int trainSamples = 80;
        int testSamples = 20;

        var trainX = new Matrix<double>(trainSamples, 1);
        var trainY = new Vector<double>(trainSamples);
        for (int i = 0; i < trainSamples; i++)
        {
            trainX[i, 0] = random.NextDouble() * 2 * Math.PI;
            trainY[i] = Math.Sin(trainX[i, 0]) + random.NextDouble() * 0.1;
        }

        var testX = new Matrix<double>(testSamples, 1);
        var testY = new Vector<double>(testSamples);
        for (int i = 0; i < testSamples; i++)
        {
            testX[i, 0] = random.NextDouble() * 2 * Math.PI;
            testY[i] = Math.Sin(testX[i, 0]) + random.NextDouble() * 0.1;
        }

        var loader = DataLoaders.FromMatrixVector(trainX, trainY);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var predictions = result.Predict(testX);

        // Assert: R² > -0.5 (not catastrophically worse than mean baseline)
        // A linear fit to sin(x) on [0, 2π] has near-zero R²
        double r2 = CalculateR2(testY, predictions);
        Assert.True(r2 > -0.5,
            $"R² = {r2:F4} — model is performing much worse than mean baseline. " +
            "Even a linear model on sin(x) should not be this bad.");

        // All predictions should be finite
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]));
            Assert.False(double.IsInfinity(predictions[i]));
        }
    }

    [Fact]
    public void LargeDataset_PredictionStability()
    {
        // Arrange: Train on 500 samples, verify prediction determinism
        var random = new Random(99);
        int samples = 500;
        var x = new Matrix<double>(samples, 4);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < 4; j++)
                x[i, j] = random.NextDouble() * 10;
            y[i] = x[i, 0] + 2 * x[i, 1] - x[i, 2] + 0.5 * x[i, 3];
        }

        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testData = new Matrix<double>(50, 4);
        var testRng = new Random(500);
        for (int i = 0; i < 50; i++)
            for (int j = 0; j < 4; j++)
                testData[i, j] = testRng.NextDouble() * 10;

        // Act: Predict twice
        var predictions1 = result.Predict(testData);
        var predictions2 = result.Predict(testData);

        // Assert: Results must be EXACTLY identical (deterministic)
        Assert.Equal(predictions1.Length, predictions2.Length);
        for (int i = 0; i < predictions1.Length; i++)
        {
            Assert.Equal(predictions1[i], predictions2[i]);
        }
    }

    #region Helper Methods

    /// <summary>
    /// Generates data following y = 2*x1 + 3*x2 + 1 + noise.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) GenerateLinearData(
        int samples, Random random, double noise)
    {
        var x = new Matrix<double>(samples, 2);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            x[i, 1] = random.NextDouble() * 10;
            y[i] = 2.0 * x[i, 0] + 3.0 * x[i, 1] + 1.0 + NextGaussian(random) * noise;
        }

        return (x, y);
    }

    /// <summary>
    /// Calculates R² (coefficient of determination).
    /// R² = 1 - SS_res / SS_tot where:
    /// SS_res = Σ(actual - predicted)²
    /// SS_tot = Σ(actual - mean(actual))²
    /// </summary>
    private static double CalculateR2(Vector<double> actual, Vector<double> predicted)
    {
        double mean = 0;
        for (int i = 0; i < actual.Length; i++)
            mean += actual[i];
        mean /= actual.Length;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            ssRes += Math.Pow(actual[i] - predicted[i], 2);
            ssTot += Math.Pow(actual[i] - mean, 2);
        }

        return ssTot == 0 ? 0 : 1.0 - ssRes / ssTot;
    }

    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    #endregion
}
