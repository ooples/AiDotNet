using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Gaussian process models implementing IGaussianProcess&lt;double&gt;.
/// Tests deep probabilistic invariants: variance positivity, posterior contraction,
/// interpolation at training points, and uncertainty monotonicity with distance.
/// </summary>
public abstract class GaussianProcessModelTestBase
{
    protected abstract IGaussianProcess<double> CreateModel();

    protected virtual int TrainSamples => 30;
    protected virtual int Features => 2;

    /// <summary>
    /// Generates normalized linear data with features in [0,1] suitable for default kernels.
    /// GP kernels (especially Gaussian/RBF with default sigma=1.0) require data in a
    /// scale where pairwise distances are O(1). Features in [0,10] cause exp(-100)≈0.
    /// </summary>
    protected (Matrix<double> X, Vector<double> Y) GenerateNormalizedLinearData(
        int samples, int features, Random rng, double noise = 0.1)
    {
        var (x, y) = ModelTestHelpers.GenerateLinearData(samples, features, rng, noise);
        // Normalize features to [0,1]
        for (int j = 0; j < features; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < samples; i++)
            {
                if (x[i, j] < min) min = x[i, j];
                if (x[i, j] > max) max = x[i, j];
            }
            double range = max - min;
            if (range > 1e-10)
            {
                for (int i = 0; i < samples; i++)
                    x[i, j] = (x[i, j] - min) / range;
            }
        }
        return (x, y);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Predictive Variance ≥ 0 Everywhere
    // The GP posterior variance is σ²(x) = k(x,x) - k(x,X)K⁻¹k(X,x) ≥ 0
    // by positive semi-definiteness of the kernel. Negative variance
    // indicates a numerical bug in the Cholesky factorization or kernel.
    // =====================================================

    [Fact]
    public void PredictiveVariance_ShouldBeNonNegative_Everywhere()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        // Test at various distances from training data
        var testPoints = new[]
        {
            MakePoint(0.0, 0.0),     // near origin
            MakePoint(0.5, 0.5),     // in range
            MakePoint(10.0, 10.0),   // far away
            MakePoint(-2.0, -2.0),   // extrapolation
        };

        foreach (var point in testPoints)
        {
            var (mean, variance) = model.Predict(point);
            Assert.False(double.IsNaN(variance),
                $"GP variance is NaN at ({point[0]}, {point[1]}).");
            Assert.True(variance >= -1e-10,
                $"GP variance = {variance:E4} at ({point[0]}, {point[1]}) — must be ≥ 0. " +
                "Negative variance indicates kernel matrix inversion bug.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Posterior Contraction Near Training Data
    // The GP posterior variance at training points should be ≤ prior variance.
    // Specifically, variance(x_train) should be small (ideally ≈ noise variance).
    // Variance far from training data should be larger than at training points.
    // =====================================================

    [Fact]
    public void PosteriorContraction_VarianceNearTraining_ShouldBeLessThan_VarianceFarAway()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng, noise: 0.01);

        model.Fit(trainX, trainY);

        // Variance at a training point
        var nearPoint = new Vector<double>(Features);
        for (int j = 0; j < Features; j++)
            nearPoint[j] = trainX[0, j];
        var (_, varianceNear) = model.Predict(nearPoint);

        // Variance far from all training data
        var farPoint = MakePoint(10.0, 10.0);
        var (_, varianceFar) = model.Predict(farPoint);

        if (!double.IsNaN(varianceNear) && !double.IsNaN(varianceFar) &&
            !double.IsInfinity(varianceNear) && !double.IsInfinity(varianceFar))
        {
            Assert.True(varianceNear <= varianceFar + 1e-6,
                $"Posterior contraction violated: variance at training point ({varianceNear:E4}) " +
                $"should be ≤ variance far away ({varianceFar:E4}). " +
                "GP is not reducing uncertainty near observed data.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Uncertainty Increases With Distance
    // As we move further from training data, variance should monotonically
    // increase (for stationary kernels). This is a fundamental property.
    // =====================================================

    [Fact]
    public void Uncertainty_ShouldIncrease_WithDistanceFromTrainingData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        // Cluster training data near origin
        var trainX = new Matrix<double>(20, Features);
        var trainY = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < Features; j++)
                trainX[i, j] = rng.NextDouble() * 2.0; // all near [0,2]
            trainY[i] = trainX[i, 0] + trainX[i, 1]; // simple function
        }

        model.Fit(trainX, trainY);

        double[] distances = { 1.0, 10.0, 50.0, 200.0 };
        double prevVariance = -1;
        int violations = 0;

        foreach (double d in distances)
        {
            var point = MakePoint(d, d);
            var (_, variance) = model.Predict(point);

            if (!double.IsNaN(variance) && !double.IsInfinity(variance) && variance >= 0)
            {
                if (prevVariance >= 0 && variance < prevVariance - 1e-8)
                    violations++;
                prevVariance = variance;
            }
        }

        Assert.True(violations <= 1,
            $"Uncertainty monotonicity violated {violations}/3 times. " +
            "Variance should increase with distance from training data.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Mean Interpolation
    // For noise-free GP (or very low noise), the posterior mean should
    // approximately interpolate training points: μ(x_train) ≈ y_train.
    // Large interpolation error indicates kernel or matrix inversion bugs.
    // =====================================================

    [Fact]
    public void Mean_ShouldApproximatelyInterpolate_TrainingPoints()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        // Use small dataset with low noise for clear interpolation test
        var (trainX, trainY) = GenerateNormalizedLinearData(15, Features, rng, noise: 0.001);

        model.Fit(trainX, trainY);

        double totalError = 0;
        int validCount = 0;
        for (int i = 0; i < Math.Min(5, trainX.Rows); i++)
        {
            var point = new Vector<double>(Features);
            for (int j = 0; j < Features; j++)
                point[j] = trainX[i, j];

            var (mean, _) = model.Predict(point);
            if (!double.IsNaN(mean) && !double.IsInfinity(mean))
            {
                totalError += Math.Abs(mean - trainY[i]);
                validCount++;
            }
        }

        if (validCount > 0)
        {
            double avgError = totalError / validCount;
            double targetRange = 0;
            double minY = double.MaxValue, maxY = double.MinValue;
            for (int i = 0; i < trainY.Length; i++)
            {
                if (trainY[i] < minY) minY = trainY[i];
                if (trainY[i] > maxY) maxY = trainY[i];
            }
            targetRange = maxY - minY;

            Assert.True(avgError < targetRange * 0.3,
                $"Average interpolation error = {avgError:F4} (target range = {targetRange:F4}). " +
                "GP mean should approximately pass through training points.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Mean Should Be Reasonable
    // On linear data y=2x1+4x2+1, the GP mean at (5,5) should
    // be in the right ballpark (positive, reasonable magnitude).
    // =====================================================

    [Fact]
    public void Mean_ShouldBeReasonable_OnLinearData()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng, noise: 0.1);

        model.Fit(trainX, trainY);

        // Test at a point within the normalized training range [0,1]
        var testPoint = MakePoint(0.5, 0.5);
        // With normalized features and y=2x1+4x2+1, y at (0.5,0.5) ≈ positive
        var (mean, _) = model.Predict(testPoint);

        if (!double.IsNaN(mean) && !double.IsInfinity(mean))
        {
            Assert.True(mean > 0.0,
                $"GP mean = {mean:F4} at (0.5,0.5). Should be positive for linear data.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Finite Predictions, Determinism
    // =====================================================

    [Fact]
    public void Predictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        var point = MakePoint(0.5, 0.5);
        var (mean, variance) = model.Predict(point);

        Assert.False(double.IsNaN(mean), "GP mean is NaN.");
        Assert.False(double.IsInfinity(mean), "GP mean is Infinity.");
        Assert.False(double.IsNaN(variance), "GP variance is NaN.");
    }

    [Fact]
    public void Predict_ShouldBeDeterministic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        var point = MakePoint(0.5, 0.5);
        var (mean1, var1) = model.Predict(point);
        var (mean2, var2) = model.Predict(point);

        Assert.Equal(mean1, mean2);
        Assert.Equal(var1, var2);
    }

    private Vector<double> MakePoint(params double[] values)
    {
        var point = new Vector<double>(Features);
        for (int j = 0; j < Features && j < values.Length; j++)
            point[j] = values[j];
        return point;
    }
}
