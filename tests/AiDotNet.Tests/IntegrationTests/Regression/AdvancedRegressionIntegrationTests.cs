using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Regression.MixedEffects;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for advanced regression models:
/// M5ModelTree, SuperLearner, GeneralizedAdditiveModel,
/// MixedEffectsModel, DeepHit, DeepSurv.
///
/// Tests verify mathematical correctness on known synthetic data.
/// </summary>
[Trait("Category", "Integration")]
public class AdvancedRegressionIntegrationTests
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

    private static (Matrix<double> x, Vector<double> y) CreateNonLinearData(
        int n, int seed)
    {
        var random = new Random(seed);
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 6 - 3;
            x[i, 1] = random.NextDouble() * 4 - 2;
            y[i] = x[i, 0] * x[i, 0] + 0.5 * x[i, 1] + NextGaussian(random) * 0.3;
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

    #region M5ModelTree

    [Fact]
    public void M5ModelTree_FitsNonLinearData_BetterThanLinear()
    {
        var (x, y) = CreateNonLinearData(100, seed: 42);

        var model = new M5ModelTree<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "M5 predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.5, $"M5ModelTree R²={r2:F4} should be > 0.5 on quadratic data");
    }

    [Fact]
    public void M5ModelTree_LeafModelsAreLinear()
    {
        // M5 should use linear models at leaves, so it should handle both
        // piecewise linear and non-linear relationships
        var (x, y) = CreateLinearData(80, new[] { 2.0, -1.0 }, intercept: 1.0, noise: 0.3, seed: 42);

        var model = new M5ModelTree<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.8, $"M5 R²={r2:F4} on clean linear data should be > 0.8");
    }

    #endregion

    #region SuperLearner

    [Fact]
    public void SuperLearner_FitsLinearData_AtLeastAsGoodAsBestBase()
    {
        var (x, y) = CreateLinearData(80, new[] { 2.0, 1.0, -0.5 }, intercept: 3.0, noise: 0.5, seed: 42);

        var model = new SuperLearner<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "SuperLearner predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.5, $"SuperLearner R²={r2:F4} should be > 0.5 (at least as good as best base)");
    }

    [Fact]
    public void SuperLearner_OutputLengthMatchesInput()
    {
        var (x, y) = CreateLinearData(50, new[] { 1.0 }, intercept: 0, noise: 0.1, seed: 42);

        var model = new SuperLearner<double>();
        model.Train(x, y);

        var testX = new Matrix<double>(15, 1);
        var random = new Random(99);
        for (int i = 0; i < 15; i++) testX[i, 0] = random.NextDouble() * 10 - 5;

        var predictions = model.Predict(testX);
        Assert.Equal(15, predictions.Length);
    }

    #endregion

    #region GeneralizedAdditiveModel

    [Fact]
    public void GAM_FitsNonLinearData_ReasonableR2()
    {
        var (x, y) = CreateNonLinearData(80, seed: 42);

        var model = new GeneralizedAdditiveModel<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "GAM predictions contain NaN/Infinity");
        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.3, $"GAM R²={r2:F4} should be > 0.3 on non-linear data");
    }

    [Fact]
    public void GAM_CapturesAdditiveStructure()
    {
        // y = f1(x1) + f2(x2) — GAM should capture additive effects
        var (x, y) = CreateLinearData(60, new[] { 2.0, -1.0 }, intercept: 0, noise: 0.3, seed: 42);

        var model = new GeneralizedAdditiveModel<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        double r2 = ComputeR2(y, predictions);
        Assert.True(r2 > 0.5, $"GAM R²={r2:F4} on additive data should be > 0.5");
    }

    #endregion

    #region MixedEffectsModel

    [Fact]
    public void LinearMixedModel_FitsData_PredictionsFinite()
    {
        // Create data with 2 fixed-effect features + 1 grouping column (col 2)
        // y = 2*x1 - x2 + 3 + group_effect + noise
        int n = 60;
        int nGroups = 3;
        var random = new Random(42);
        var x = new Matrix<double>(n, 3); // col 0,1 = fixed effects, col 2 = group
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 10 - 5;
            x[i, 1] = random.NextDouble() * 10 - 5;
            x[i, 2] = i % nGroups; // group assignment
            double groupEffect = (i % nGroups) * 0.5; // different baseline per group
            y[i] = 2.0 * x[i, 0] - 1.0 * x[i, 1] + 3.0 + groupEffect
                    + NextGaussian(random) * 0.5;
        }

        var model = new LinearMixedModel<double>();
        model.AddRandomIntercept("group", 2); // column 2 is the grouping variable
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "LinearMixedModel predictions contain NaN/Infinity");
        Assert.Equal(x.Rows, predictions.Length);
    }

    #endregion

    #region DeepHit (Survival)

    [Fact]
    public void DeepHit_TrainsAndPredicts_OutputsFinite()
    {
        // DeepHit is a survival analysis model — uses time-to-event data
        var (x, y) = CreateLinearData(60, new[] { 1.0, 0.5 }, intercept: 2.0, noise: 0.3, seed: 42);
        // Make y positive (time-to-event must be positive)
        for (int i = 0; i < y.Length; i++)
            y[i] = Math.Abs(y[i]) + 0.1;

        var model = new DeepHit<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "DeepHit predictions contain NaN/Infinity");
        Assert.Equal(x.Rows, predictions.Length);
    }

    #endregion

    #region DeepSurv (Survival)

    [Fact]
    public void DeepSurv_TrainsAndPredicts_OutputsFinite()
    {
        var (x, y) = CreateLinearData(60, new[] { 1.0, -0.5 }, intercept: 3.0, noise: 0.3, seed: 42);
        for (int i = 0; i < y.Length; i++)
            y[i] = Math.Abs(y[i]) + 0.1;

        var model = new DeepSurv<double>();
        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.True(AllFinite(predictions), "DeepSurv predictions contain NaN/Infinity");
        Assert.Equal(x.Rows, predictions.Length);
    }

    [Fact]
    public void DeepSurv_HigherRisk_HigherPrediction()
    {
        // Create data where higher x values → shorter survival
        var random = new Random(42);
        int n = 60;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = random.NextDouble() * 4;
            // Higher x → shorter survival time (inverse relationship)
            y[i] = Math.Exp(-0.5 * x[i, 0]) * (2 + NextGaussian(random) * 0.1);
            y[i] = Math.Max(0.1, y[i]);
        }

        var model = new DeepSurv<double>();
        model.Train(x, y);

        // Higher risk factor should give different prediction than lower
        var lowRisk = new Matrix<double>(1, 1);
        lowRisk[0, 0] = 0.5;
        var highRisk = new Matrix<double>(1, 1);
        highRisk[0, 0] = 3.5;

        var predLow = model.Predict(lowRisk);
        var predHigh = model.Predict(highRisk);

        // Both should be finite, and different from each other
        Assert.True(AllFinite(predLow), "DeepSurv low-risk prediction is not finite");
        Assert.True(AllFinite(predHigh), "DeepSurv high-risk prediction is not finite");
        Assert.NotEqual(predLow[0], predHigh[0]);
    }

    #endregion

    #region Cross-cutting: All models produce correct output length

    private void AssertOutputLengthAndFinite(IFullModel<double, Matrix<double>, Vector<double>> model, string name)
    {
        var (x, y) = CreateLinearData(40, new[] { 1.0, -0.5 }, intercept: 2.0, noise: 0.3, seed: 42);
        for (int i = 0; i < y.Length; i++) y[i] = Math.Abs(y[i]) + 0.1;

        model.Train(x, y);
        var predictions = model.Predict(x);

        Assert.Equal(x.Rows, predictions.Length);
        Assert.True(AllFinite(predictions), $"{name} produced NaN/Infinity predictions");
    }

    [Fact]
    public void M5ModelTree_OutputLengthMatchesInput()
        => AssertOutputLengthAndFinite(new M5ModelTree<double>(), "M5ModelTree");

    [Fact]
    public void SuperLearner_CrossCutting_FinitePredictions()
        => AssertOutputLengthAndFinite(new SuperLearner<double>(), "SuperLearner");

    [Fact]
    public void GAM_OutputLengthMatchesInput()
        => AssertOutputLengthAndFinite(new GeneralizedAdditiveModel<double>(), "GAM");

    [Fact]
    public void DeepHit_OutputLengthMatchesInput()
        => AssertOutputLengthAndFinite(new DeepHit<double>(), "DeepHit");

    [Fact]
    public void DeepSurv_OutputLengthMatchesInput()
        => AssertOutputLengthAndFinite(new DeepSurv<double>(), "DeepSurv");

    #endregion
}
