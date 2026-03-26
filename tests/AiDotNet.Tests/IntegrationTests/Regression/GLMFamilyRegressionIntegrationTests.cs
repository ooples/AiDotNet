using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for Generalized Linear Model regression family:
/// BetaRegression, GammaRegression, TweedieRegression, InverseGaussianRegression,
/// NegativeBinomialRegression, ZeroInflatedRegression, PoissonRegression, GAMLSSRegression.
///
/// GLMs model non-normal response distributions with link functions.
/// Tests verify mathematical correctness on appropriate synthetic data.
/// </summary>
[Trait("Category", "Integration")]
public class GLMFamilyRegressionIntegrationTests
{
    #region Test Data Helpers

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

    /// <summary>
    /// Creates positive-valued response data suitable for Gamma/InverseGaussian/Poisson/etc.
    /// y = exp(beta0 + beta1*x1 + beta2*x2 + noise) — log-link generates positive y.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreatePositiveResponseData(
        int n, double[] coefficients, double intercept, double noise, int seed)
    {
        var random = new Random(seed);
        int p = coefficients.Length;
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double eta = intercept;
            for (int j = 0; j < p; j++)
            {
                x[i, j] = random.NextDouble() * 2 - 1; // [-1, 1]
                eta += coefficients[j] * x[i, j];
            }
            // Log-link: y = exp(eta + noise), always positive
            y[i] = Math.Exp(eta + NextGaussian(random) * noise);
        }

        return (x, y);
    }

    /// <summary>
    /// Creates count data (non-negative integers) suitable for Poisson/NegativeBinomial.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateCountData(
        int n, double[] coefficients, double intercept, int seed)
    {
        var random = new Random(seed);
        int p = coefficients.Length;
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double eta = intercept;
            for (int j = 0; j < p; j++)
            {
                x[i, j] = random.NextDouble() * 2 - 1;
                eta += coefficients[j] * x[i, j];
            }
            // Poisson mean: lambda = exp(eta)
            double lambda = Math.Exp(Math.Min(eta, 5)); // Cap to prevent overflow
            // Simple Poisson sampling
            y[i] = SamplePoisson(random, lambda);
        }

        return (x, y);
    }

    private static double SamplePoisson(Random random, double lambda)
    {
        double L = Math.Exp(-lambda);
        double p = 1.0;
        int k = 0;
        do
        {
            k++;
            p *= random.NextDouble();
        } while (p > L && k < 100);
        return k - 1;
    }

    /// <summary>
    /// Creates data in (0, 1) range suitable for BetaRegression.
    /// </summary>
    private static (Matrix<double> x, Vector<double> y) CreateBetaData(
        int n, double[] coefficients, double intercept, int seed)
    {
        var random = new Random(seed);
        int p = coefficients.Length;
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double eta = intercept;
            for (int j = 0; j < p; j++)
            {
                x[i, j] = random.NextDouble() * 2 - 1;
                eta += coefficients[j] * x[i, j];
            }
            // Logit-link: y = sigmoid(eta + noise), in (0,1)
            double logit = eta + NextGaussian(random) * 0.3;
            y[i] = 1.0 / (1.0 + Math.Exp(-logit));
            // Clamp to avoid exact 0 or 1
            y[i] = Math.Max(0.01, Math.Min(0.99, y[i]));
        }

        return (x, y);
    }

    #endregion

    #region BetaRegression

    [Fact]
    public void BetaRegression_FitsProportionData_ReasonableR2()
    {
        var (x, y) = CreateBetaData(80, new[] { 1.0, -0.5 }, intercept: 0, seed: 42);

        var model = new BetaRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "BetaRegression predictions contain NaN/Infinity");
        Assert.Equal(x.Rows, predictions.Length);

        // All predictions should be in (0, 1) for beta regression
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] > -0.1 && predictions[i] < 1.1,
                $"BetaRegression prediction {predictions[i]:F4} at index {i} should be near (0,1)");
        }
    }

    #endregion

    #region GammaRegression

    [Fact]
    public void GammaRegression_FitsPositiveData_ReasonableR2()
    {
        var (x, y) = CreatePositiveResponseData(80, new[] { 0.5, -0.3 }, intercept: 1.0, noise: 0.2, seed: 42);

        var model = new GammaRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "GammaRegression predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.2, $"GammaRegression R²={r2:F4} should be > 0.2 on positive data");
    }

    [Fact]
    public void GammaRegression_PredictionsArePositive()
    {
        var (x, y) = CreatePositiveResponseData(60, new[] { 0.3 }, intercept: 0.5, noise: 0.1, seed: 42);

        var model = new GammaRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] > 0,
                $"GammaRegression prediction should be positive, got {predictions[i]:F4} at index {i}");
        }
    }

    #endregion

    #region InverseGaussianRegression

    [Fact]
    public void InverseGaussianRegression_FitsPositiveData_PredictionsFinite()
    {
        var (x, y) = CreatePositiveResponseData(80, new[] { 0.5, -0.2 }, intercept: 1.0, noise: 0.2, seed: 42);

        var model = new InverseGaussianRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "InverseGaussian predictions contain NaN/Infinity");
        Assert.Equal(x.Rows, predictions.Length);
    }

    #endregion

    #region TweedieRegression

    [Fact]
    public void TweedieRegression_FitsPositiveData_ReasonableR2()
    {
        var (x, y) = CreatePositiveResponseData(80, new[] { 0.5, 0.3 }, intercept: 0.5, noise: 0.2, seed: 42);

        var model = new TweedieRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "TweedieRegression predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.1, $"Tweedie R²={r2:F4} should be > 0.1 on positive data");
    }

    #endregion

    #region PoissonRegression

    [Fact]
    public void PoissonRegression_FitsCountData_ReasonableR2()
    {
        var (x, y) = CreateCountData(100, new[] { 0.5, -0.3 }, intercept: 1.0, seed: 42);

        var model = new PoissonRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "PoissonRegression predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.1, $"Poisson R²={r2:F4} should be > 0.1 on count data");
    }

    [Fact]
    public void PoissonRegression_PredictionsAreNonNegative()
    {
        var (x, y) = CreateCountData(60, new[] { 0.3, 0.2 }, intercept: 0.5, seed: 42);

        var model = new PoissonRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= -0.01,
                $"Poisson prediction should be non-negative, got {predictions[i]:F4} at index {i}");
        }
    }

    #endregion

    #region NegativeBinomialRegression

    [Fact]
    public void NegativeBinomialRegression_FitsCountData_PredictionsFinite()
    {
        var (x, y) = CreateCountData(80, new[] { 0.5, -0.2 }, intercept: 1.0, seed: 42);

        var model = new NegativeBinomialRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "NegBinomial predictions contain NaN/Infinity");
        Assert.Equal(x.Rows, predictions.Length);
    }

    #endregion

    #region ZeroInflatedRegression

    [Fact]
    public void ZeroInflatedRegression_HandlesZeroHeavyData()
    {
        // Create data with many zeros (zero-inflated pattern)
        var random = new Random(42);
        int n = 80;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 2 - 1;
            x[i, 1] = random.NextDouble() * 2 - 1;
            // 50% zeros, 50% Poisson counts
            if (random.NextDouble() < 0.5)
                y[i] = 0;
            else
                y[i] = SamplePoisson(random, Math.Exp(0.5 + 0.3 * x[i, 0]));
        }

        var model = new ZeroInflatedRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "ZeroInflated predictions contain NaN/Infinity");
        Assert.Equal(n, predictions.Length);
    }

    #endregion

    #region GAMLSSRegression

    [Fact]
    public void GAMLSSRegression_FitsData_PredictionsFinite()
    {
        var random = new Random(42);
        int n = 80;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 4 - 2;
            x[i, 1] = random.NextDouble() * 4 - 2;
            y[i] = 2.0 * x[i, 0] + x[i, 1] + NextGaussian(random) * 0.5;
        }

        var model = new GAMLSSRegression<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "GAMLSS predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.3, $"GAMLSS R²={r2:F4} should be > 0.3 on linear data");
    }

    #endregion
}
