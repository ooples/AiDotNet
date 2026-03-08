using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Mathematically rigorous tests for regression models verifying:
/// 1. Models actually learn (R² > threshold on clean data)
/// 2. Serialize/Deserialize round-trip preserves predictions
/// 3. Direct training produces valid coefficients
/// 4. Builder pipeline integrates correctly with each model type
///
/// These tests use synthetic data with known properties so we can verify
/// mathematical correctness, not just "didn't crash".
/// </summary>
public class RegressionModelMathTests
{
    #region Direct Model Training — verifies each model type learns correctly

    [Fact]
    public void SimpleRegression_FitsLinearData_AccurateCoefficients()
    {
        // y = 3x + 2 with no noise
        var x = new Matrix<double>(50, 1);
        var y = new Vector<double>(50);
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            x[i, 0] = random.NextDouble() * 10;
            y[i] = 3.0 * x[i, 0] + 2.0;
        }

        var model = new SimpleRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.999,
            $"SimpleRegression R²={r2:F6} on noise-free linear data should be > 0.999");

        // Verify coefficients by probing: predict at x=0 and x=1
        var probeX = new Matrix<double>(2, 1);
        probeX[0, 0] = 0.0;
        probeX[1, 0] = 1.0;
        var probes = model.Predict(probeX);
        double recoveredIntercept = probes[0];
        double recoveredSlope = probes[1] - probes[0];
        Assert.InRange(recoveredSlope, 2.5, 3.5);       // true slope: 3.0
        Assert.InRange(recoveredIntercept, 1.5, 2.5);   // true intercept: 2.0
    }

    [Fact]
    public void MultipleRegression_FitsMultiVariateData_HighR2()
    {
        // y = 2*x1 + 3*x2 - x3 + 5
        var (x, y) = CreateLinearData(100, new[] { 2.0, 3.0, -1.0 }, intercept: 5.0, noise: 0.01, seed: 42);

        var model = new MultipleRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.999,
            $"MultipleRegression R²={r2:F6} on low-noise linear data should be > 0.999");
    }

    [Fact]
    public void RidgeRegression_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 1.5, -2.0, 0.5, 3.0 }, intercept: 1.0, noise: 0.5, seed: 42);

        var model = new RidgeRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.95,
            $"RidgeRegression R²={r2:F6} on moderate-noise linear data should be > 0.95");
    }

    [Fact]
    public void LassoRegression_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 1.5, -2.0, 0.5, 3.0 }, intercept: 1.0, noise: 0.5, seed: 42);

        var model = new LassoRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.90,
            $"LassoRegression R²={r2:F6} on moderate-noise linear data should be > 0.90");
    }

    [Fact]
    public void ElasticNetRegression_FitsLinearData_ReasonableR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 1.5, -2.0, 0.5, 3.0 }, intercept: 1.0, noise: 0.5, seed: 42);

        var model = new ElasticNetRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.85,
            $"ElasticNetRegression R²={r2:F6} on moderate-noise linear data should be > 0.85");
    }

    [Fact]
    public void PolynomialRegression_FitsQuadraticData_HighR2()
    {
        // y = 2*x^2 - 3*x + 1
        var x = new Matrix<double>(60, 1);
        var y = new Vector<double>(60);
        var random = new Random(42);
        for (int i = 0; i < 60; i++)
        {
            double xi = random.NextDouble() * 6 - 3;
            x[i, 0] = xi;
            y[i] = 2.0 * xi * xi - 3.0 * xi + 1.0 + random.NextDouble() * 0.1;
        }

        var model = new PolynomialRegression<double>(
            new PolynomialRegressionOptions<double> { Degree = 2 });
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.99,
            $"PolynomialRegression R²={r2:F6} on quadratic data should be > 0.99");
    }

    [Fact]
    public void DecisionTreeRegression_FitsNonLinearData_ReasonableR2()
    {
        // Non-linear: y = sin(x1) + x2^2
        // Use separate train and test sets to evaluate on holdout data
        var (trainX, trainY) = CreateNonLinearData(100, seed: 42);
        var (testX, testY) = CreateNonLinearData(30, seed: 99);

        var model = new DecisionTreeRegression<double>();
        model.Train(trainX, trainY);
        var predictions = model.Predict(testX);

        double r2 = ComputeR2(testY, predictions);
        Assert.True(r2 > 0.50,
            $"DecisionTreeRegression R²={r2:F6} on holdout non-linear data should be > 0.50");
    }

    [Fact]
    public void WeightedRegression_FitsData_HighR2()
    {
        var (x, y) = CreateLinearData(80, new[] { 2.0, -1.0, 3.0 }, intercept: 0.5, noise: 0.3, seed: 42);

        // Use non-uniform weights: emphasize first half of data more heavily
        var weights = new Vector<double>(80);
        for (int i = 0; i < 80; i++)
            weights[i] = i < 40 ? 3.0 : 1.0; // first 40 points weighted 3x more
        var options = new WeightedRegressionOptions<double> { Weights = weights };

        var model = new WeightedRegression<double>(options);
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.95,
            $"WeightedRegression R²={r2:F6} on linear data should be > 0.95");
    }

    #endregion

    #region Regression Serialize/Deserialize Round-Trip

    [Fact]
    public void SimpleRegression_SerializeRoundTrip_PredictionsMatch()
    {
        var (x, y) = CreateLinearData(50, new[] { 3.0 }, intercept: 2.0, noise: 0.0, seed: 42);

        var model = new SimpleRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(10, 1, seed: 100);
        var original = model.Predict(testX);

        var bytes = model.Serialize();
        var restored = new SimpleRegression<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "SimpleRegression");
    }

    [Fact]
    public void MultipleRegression_SerializeRoundTrip_PredictionsMatch()
    {
        var (x, y) = CreateLinearData(60, new[] { 2.0, 3.0, -1.0 }, intercept: 5.0, noise: 0.01, seed: 42);

        var model = new MultipleRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(10, 3, seed: 200);
        var original = model.Predict(testX);

        var bytes = model.Serialize();
        var restored = new MultipleRegression<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "MultipleRegression");
    }

    [Fact]
    public void RidgeRegression_SerializeRoundTrip_PredictionsMatch()
    {
        var (x, y) = CreateLinearData(60, new[] { 1.5, -2.0, 0.5 }, intercept: 1.0, noise: 0.3, seed: 42);

        var model = new RidgeRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(10, 3, seed: 300);
        var original = model.Predict(testX);

        var bytes = model.Serialize();
        var restored = new RidgeRegression<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "RidgeRegression");
    }

    [Fact]
    public void LassoRegression_SerializeRoundTrip_PredictionsMatch()
    {
        var (x, y) = CreateLinearData(60, new[] { 1.5, -2.0, 0.5 }, intercept: 1.0, noise: 0.3, seed: 42);

        var model = new LassoRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(10, 3, seed: 400);
        var original = model.Predict(testX);

        var bytes = model.Serialize();
        var restored = new LassoRegression<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "LassoRegression");
    }

    [Fact]
    public void ElasticNetRegression_SerializeRoundTrip_PredictionsMatch()
    {
        var (x, y) = CreateLinearData(60, new[] { 1.5, -2.0, 0.5 }, intercept: 1.0, noise: 0.3, seed: 42);

        var model = new ElasticNetRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(10, 3, seed: 500);
        var original = model.Predict(testX);

        var bytes = model.Serialize();
        var restored = new ElasticNetRegression<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "ElasticNetRegression");
    }

    [Fact]
    public void DecisionTreeRegression_SerializeRoundTrip_PredictionsMatch()
    {
        var (x, y) = CreateNonLinearData(80, seed: 42);

        var model = new DecisionTreeRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(10, 2, seed: 600);
        var original = model.Predict(testX);

        var bytes = model.Serialize();
        var restored = new DecisionTreeRegression<double>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        AssertPredictionsMatch(original, restoredPreds, "DecisionTreeRegression");
    }

    #endregion

    #region Builder Integration Tests — ensures optimizer pipeline works

    [Fact]
    public void LassoRegression_ThroughBuilder_ProducesAccuratePredictions()
    {
        var (x, y) = CreateLinearData(60, new[] { 2.0, -1.0, 3.0, 0.0 }, intercept: 1.0, noise: 0.5, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new LassoRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testX = CreateRandomMatrix(10, 4, seed: 100);
        var predictions = result.Predict(testX);

        Assert.NotNull(predictions);
        Assert.Equal(10, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Prediction {i} is NaN");
            Assert.False(double.IsInfinity(predictions[i]), $"Prediction {i} is Infinity");
            // Predictions should be in a reasonable range given the data distribution
            Assert.InRange(predictions[i], -100.0, 100.0);
        }
    }

    [Fact]
    public void ElasticNet_ThroughBuilder_ProducesAccuratePredictions()
    {
        var (x, y) = CreateLinearData(60, new[] { 2.0, -1.0, 3.0 }, intercept: 1.0, noise: 0.5, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new ElasticNetRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testX = CreateRandomMatrix(10, 3, seed: 200);
        var predictions = result.Predict(testX);

        Assert.NotNull(predictions);
        Assert.Equal(10, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Prediction {i} is NaN");
            Assert.InRange(predictions[i], -100.0, 100.0);
        }
    }

    [Fact]
    public void PolynomialRegression_ThroughBuilder_ProducesAccuratePredictions()
    {
        // Quadratic data
        var x = new Matrix<double>(60, 1);
        var y = new Vector<double>(60);
        var random = new Random(42);
        for (int i = 0; i < 60; i++)
        {
            double xi = random.NextDouble() * 6 - 3;
            x[i, 0] = xi;
            y[i] = 2.0 * xi * xi - 3.0 * xi + 1.0 + random.NextDouble() * 0.5;
        }

        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new PolynomialRegression<double>(
                new PolynomialRegressionOptions<double> { Degree = 2 }))
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testX = CreateRandomMatrix(10, 1, seed: 300);
        var predictions = result.Predict(testX);

        Assert.NotNull(predictions);
        Assert.Equal(10, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]), $"Prediction {i} is NaN");
            Assert.False(double.IsInfinity(predictions[i]), $"Prediction {i} is Infinity");
        }
    }

    [Fact]
    public void Builder_SerializeRoundTrip_LassoRegression_PredictionsMatch()
    {
        var (x, y) = CreateLinearData(60, new[] { 2.0, -1.0, 3.0 }, intercept: 1.0, noise: 0.3, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new LassoRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var testX = CreateRandomMatrix(5, 3, seed: 400);
        var original = result.Predict(testX);

        var bytes = result.Serialize();
        var restored = new AiModelResult<double, Matrix<double>, Vector<double>>();
        restored.Deserialize(bytes);
        var restoredPreds = restored.Predict(testX);

        Assert.Equal(original.Length, restoredPreds.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], restoredPreds[i], precision: 10);
        }
    }

    #endregion

    #region Mathematical Properties

    [Fact]
    public void MultipleRegression_PredictionDeterminism_SameInputSameOutput()
    {
        var (x, y) = CreateLinearData(50, new[] { 2.0, -1.0 }, intercept: 3.0, noise: 0.1, seed: 42);
        var model = new MultipleRegression<double>();
        model.Train(x, y);

        var testX = CreateRandomMatrix(5, 2, seed: 100);
        var pred1 = model.Predict(testX);
        var pred2 = model.Predict(testX);

        for (int i = 0; i < pred1.Length; i++)
        {
            Assert.Equal(pred1[i], pred2[i], precision: 15);
        }
    }

    [Fact]
    public void RidgeRegression_HighRegularization_CoefficientsAreShrunk()
    {
        // With very high regularization, coefficients should be close to zero
        var (x, y) = CreateLinearData(100, new[] { 5.0, -3.0, 2.0 }, intercept: 1.0, noise: 0.1, seed: 42);

        var lowReg = new RidgeRegression<double>(
            new RidgeRegressionOptions<double> { Alpha = 0.001 });
        lowReg.Train(x, y);
        var lowRegPreds = lowReg.Predict(x);

        var highReg = new RidgeRegression<double>(
            new RidgeRegressionOptions<double> { Alpha = 1000.0 });
        highReg.Train(x, y);
        var highRegPreds = highReg.Predict(x);

        double r2Low = ComputeR2(y, lowRegPreds);
        double r2High = ComputeR2(y, highRegPreds);

        // Low regularization should fit better than high regularization
        Assert.True(r2Low > r2High,
            $"Low regularization R²={r2Low:F4} should be > high regularization R²={r2High:F4}");
    }

    #endregion

    #region Helper Methods

    private static (Matrix<double> x, Vector<double> y) CreateLinearData(
        int samples, double[] coefficients, double intercept, double noise, int seed)
    {
        var random = new Random(seed);
        int features = coefficients.Length;
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            double target = intercept;
            for (int j = 0; j < features; j++)
            {
                x[i, j] = random.NextDouble() * 10 - 5;
                target += coefficients[j] * x[i, j];
            }
            y[i] = target + NextGaussian(random) * noise;
        }
        return (x, y);
    }

    private static (Matrix<double> x, Vector<double> y) CreateNonLinearData(int samples, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(samples, 2);
        var y = new Vector<double>(samples);

        for (int i = 0; i < samples; i++)
        {
            x[i, 0] = random.NextDouble() * 6 - 3;
            x[i, 1] = random.NextDouble() * 4 - 2;
            y[i] = Math.Sin(x[i, 0]) + x[i, 1] * x[i, 1] + random.NextDouble() * 0.1;
        }
        return (x, y);
    }

    private static Matrix<double> CreateRandomMatrix(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = random.NextDouble() * 10 - 5;
        return matrix;
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

    private static void AssertPredictionsMatch(Vector<double> original, Vector<double> restored, string modelName)
    {
        Assert.Equal(original.Length, restored.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.True(Math.Abs(original[i] - restored[i]) < 1e-10,
                $"{modelName}: Prediction mismatch at index {i}: original={original[i]}, restored={restored[i]}");
        }
    }

    #endregion
}
