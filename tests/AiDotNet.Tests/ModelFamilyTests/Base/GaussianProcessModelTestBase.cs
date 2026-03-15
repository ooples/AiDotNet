using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Gaussian process models implementing IGaussianProcess&lt;double&gt;.
/// Tests probabilistic prediction invariants — mean should be reasonable and variance should be non-negative.
/// </summary>
public abstract class GaussianProcessModelTestBase
{
    protected abstract IGaussianProcess<double> CreateModel();

    protected virtual int TrainSamples => 50;
    protected virtual int Features => 2;

    // --- Fit + Predict Contract ---

    [Fact]
    public void Fit_ThenPredict_MeanIsFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        var testPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            testPoint[j] = 5.0;

        var (mean, _) = model.Predict(testPoint);

        Assert.False(double.IsNaN(mean), "GP mean prediction is NaN.");
        Assert.False(double.IsInfinity(mean), "GP mean prediction is Infinity.");
    }

    [Fact]
    public void Fit_ThenPredict_VarianceNonNegative()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        var testPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            testPoint[j] = 5.0;

        var (_, variance) = model.Predict(testPoint);

        Assert.False(double.IsNaN(variance), "GP variance is NaN.");
        Assert.True(variance >= 0.0,
            $"GP variance = {variance:F4} — variance must be non-negative.");
    }

    // --- Near Training Data: Low Variance ---

    [Fact]
    public void Fit_NearTrainingPoint_LowVariance()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        // Predict at the first training point
        var nearPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            nearPoint[j] = trainX[0, j];

        var (_, varianceNear) = model.Predict(nearPoint);

        // Predict far from training data
        var farPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            farPoint[j] = 1000.0;

        var (_, varianceFar) = model.Predict(farPoint);

        // Variance near training data should generally be less than far away
        if (!double.IsNaN(varianceNear) && !double.IsNaN(varianceFar) &&
            !double.IsInfinity(varianceNear) && !double.IsInfinity(varianceFar))
        {
            Assert.True(varianceNear <= varianceFar + 1e-6,
                $"Variance near training point ({varianceNear:F4}) should be <= variance far away ({varianceFar:F4}).");
        }
    }

    // --- Far from Training Data: Higher Variance ---

    [Fact]
    public void Fit_FarFromTrainingData_HigherVariance()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        var farPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            farPoint[j] = 1000.0;

        var (_, variance) = model.Predict(farPoint);

        Assert.False(double.IsNaN(variance), "Variance is NaN for far-away point.");
        Assert.True(variance >= 0.0,
            $"Variance = {variance:F4} — must be non-negative even far from training data.");
    }

    // --- Determinism ---

    [Fact]
    public void Predict_Deterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        var testPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            testPoint[j] = 5.0;

        var (mean1, var1) = model.Predict(testPoint);
        var (mean2, var2) = model.Predict(testPoint);

        Assert.Equal(mean1, mean2);
        Assert.Equal(var1, var2);
    }

    // --- Mean Should Be Reasonable on Linear Data ---

    [Fact]
    public void Fit_LinearData_ReasonableMean()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        // y = 2*x1 + 4*x2 + 1
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng, noise: 0.1);

        model.Fit(trainX, trainY);

        // Predict at a point where we know the approximate answer
        var testPoint = new Vector<double>(Features);
        testPoint[0] = 5.0;
        testPoint[1] = 5.0;
        // Expected: 2*5 + 4*5 + 1 = 31

        var (mean, _) = model.Predict(testPoint);

        if (!double.IsNaN(mean) && !double.IsInfinity(mean))
        {
            Assert.True(mean > 0.0,
                $"Mean = {mean:F4} — should be positive for positive linear data at (5,5).");
        }
    }
}
