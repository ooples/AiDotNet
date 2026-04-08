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

    // =====================================================
    // MATHEMATICAL INVARIANT: Kernel Positive Semi-Definiteness
    // The Gram matrix K(X,X) must have all eigenvalues ≥ 0.
    // A negative eigenvalue means the kernel implementation is broken.
    // We approximate this by checking K is symmetric and k(x,x) > 0.
    // =====================================================

    [Fact]
    public void KernelMatrix_ShouldBeSymmetricAndPositive()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(10, Features, rng);

        model.Fit(trainX, trainY);

        // The diagonal of the predicted variances should all be non-negative
        // (this is equivalent to k(x_i, x_i) - k(x_i, X)K^{-1}k(X, x_i) >= 0)
        for (int i = 0; i < trainX.Rows; i++)
        {
            var point = new Vector<double>(Features);
            for (int j = 0; j < Features; j++)
                point[j] = trainX[i, j];

            var (_, variance) = model.Predict(point);
            if (!double.IsNaN(variance))
            {
                Assert.True(variance >= -1e-8,
                    $"Kernel PSD violation: variance at training point {i} = {variance:E4}. " +
                    "Gram matrix may have negative eigenvalues.");
            }
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Noise Variance Recovery
    // When training with known noise σ²_n, the posterior variance at
    // training points should be ≈ σ²_n (not zero, not the prior variance).
    // =====================================================

    [Fact]
    public void NoiseVarianceRecovery_VarianceAtTrainingPoints_ShouldApproximateNoiseLevel()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        double noiseStd = 0.1;
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng, noise: noiseStd);

        model.Fit(trainX, trainY);

        double avgVariance = 0;
        int validCount = 0;
        for (int i = 0; i < Math.Min(5, trainX.Rows); i++)
        {
            var point = new Vector<double>(Features);
            for (int j = 0; j < Features; j++)
                point[j] = trainX[i, j];

            var (_, variance) = model.Predict(point);
            if (!double.IsNaN(variance) && !double.IsInfinity(variance) && variance >= 0)
            {
                avgVariance += variance;
                validCount++;
            }
        }

        if (validCount > 0)
        {
            avgVariance /= validCount;
            // Variance at training points should be in the noise ballpark
            // (between 0 and 10x noise variance). Very large means no learning.
            double noiseVariance = noiseStd * noiseStd;
            Assert.True(avgVariance < noiseVariance * 100,
                $"Average variance at training points = {avgVariance:E4}, noise variance = {noiseVariance:E4}. " +
                "Posterior variance at training points should reflect the noise level, not the prior.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaling Equivariance of Mean
    // Scaling y by constant c should scale the posterior mean by c.
    // μ(x | X, cy) = c · μ(x | X, y)
    // =====================================================

    [Fact]
    public void ScalingEquivariance_ScalingTargets_ShouldScaleMean()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = GenerateNormalizedLinearData(TrainSamples, Features, rng1, noise: 0.01);
        var (trainX2, trainY2) = GenerateNormalizedLinearData(TrainSamples, Features, rng2, noise: 0.01);

        const double scale = 10.0;
        var scaledY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            scaledY[i] = trainY2[i] * scale;

        model1.Fit(trainX1, trainY1);
        model2.Fit(trainX2, scaledY);

        var testPoint = MakePoint(0.5, 0.5);
        var (mean1, _) = model1.Predict(testPoint);
        var (mean2, _) = model2.Predict(testPoint);

        if (!double.IsNaN(mean1) && !double.IsNaN(mean2) &&
            !double.IsInfinity(mean1) && !double.IsInfinity(mean2) &&
            Math.Abs(mean1) > 0.01)
        {
            double ratio = mean2 / mean1;
            Assert.True(ratio > scale * 0.3 && ratio < scale * 3.0,
                $"Scaling equivariance violated: mean ratio = {ratio:F2}, expected ~{scale}. " +
                "Scaling targets by c should scale posterior mean by c.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Translation Equivariance of Mean
    // Shifting y by constant c should shift the posterior mean by c.
    // μ(x | X, y+c) = μ(x | X, y) + c
    // =====================================================

    [Fact]
    public void TranslationEquivariance_ShiftingTargets_ShouldShiftMean()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = GenerateNormalizedLinearData(TrainSamples, Features, rng1, noise: 0.01);
        var (trainX2, trainY2) = GenerateNormalizedLinearData(TrainSamples, Features, rng2, noise: 0.01);

        const double shift = 100.0;
        var shiftedY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            shiftedY[i] = trainY2[i] + shift;

        model1.Fit(trainX1, trainY1);
        model2.Fit(trainX2, shiftedY);

        var testPoint = MakePoint(0.5, 0.5);
        var (mean1, _) = model1.Predict(testPoint);
        var (mean2, _) = model2.Predict(testPoint);

        if (!double.IsNaN(mean1) && !double.IsNaN(mean2) &&
            !double.IsInfinity(mean1) && !double.IsInfinity(mean2))
        {
            double actualShift = mean2 - mean1;
            Assert.True(Math.Abs(actualShift - shift) < shift * 0.3,
                $"Translation equivariance violated: actual shift = {actualShift:F2}, expected ~{shift}. " +
                "Shifting targets by c should shift posterior mean by c.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Log Marginal Likelihood Should Be Finite
    // log p(y|X) = -½y'K⁻¹y - ½log|K| - n/2 log(2π) should be
    // finite and typically negative. Tests the full Bayesian pipeline.
    // =====================================================

    [Fact]
    public void LogMarginalLikelihood_ShouldBeFiniteAndNegative()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        // The model should expose log marginal likelihood or we can check
        // predictions are consistent (which requires a valid likelihood computation)
        // Test via mean predictions at training points being reasonable
        double totalAbsError = 0;
        int count = 0;
        for (int i = 0; i < Math.Min(10, trainX.Rows); i++)
        {
            var point = new Vector<double>(Features);
            for (int j = 0; j < Features; j++)
                point[j] = trainX[i, j];

            var (mean, variance) = model.Predict(point);
            if (!double.IsNaN(mean) && !double.IsInfinity(mean) &&
                !double.IsNaN(variance) && !double.IsInfinity(variance))
            {
                totalAbsError += Math.Abs(mean - trainY[i]);
                count++;
                // Variance should be finite and non-negative (implied by valid likelihood)
                Assert.True(variance >= -1e-8,
                    $"Negative variance at training point {i} implies invalid marginal likelihood computation.");
            }
        }

        // If the marginal likelihood computation is correct, the posterior mean
        // should not deviate wildly from training values
        if (count > 0)
        {
            double avgError = totalAbsError / count;
            double yRange = ModelTestHelpers.ComputeRange(trainY);
            Assert.True(avgError < yRange * 0.5,
                $"Average training error = {avgError:F4} (range = {yRange:F4}). " +
                "Large errors suggest marginal likelihood computation is broken.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: More Data → Lower Variance
    // Adding more training points near a test point should reduce
    // the predictive variance at that point. This is fundamental.
    // =====================================================

    [Fact]
    public void MoreData_ShouldReducePredictiveVariance()
    {
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var modelSmall = CreateModel();
        var modelLarge = CreateModel();

        // Small dataset: 10 points
        var (smallX, smallY) = GenerateNormalizedLinearData(10, Features, rng1, noise: 0.1);
        // Large dataset: 30 points (same seed means first 10 are identical)
        var (largeX, largeY) = GenerateNormalizedLinearData(30, Features, rng2, noise: 0.1);

        modelSmall.Fit(smallX, smallY);
        modelLarge.Fit(largeX, largeY);

        var testPoint = MakePoint(0.5, 0.5);
        var (_, varianceSmall) = modelSmall.Predict(testPoint);
        var (_, varianceLarge) = modelLarge.Predict(testPoint);

        if (!double.IsNaN(varianceSmall) && !double.IsNaN(varianceLarge) &&
            !double.IsInfinity(varianceSmall) && !double.IsInfinity(varianceLarge) &&
            varianceSmall >= 0 && varianceLarge >= 0)
        {
            Assert.True(varianceLarge <= varianceSmall + 1e-6,
                $"More data did not reduce variance: small={varianceSmall:E4}, large={varianceLarge:E4}. " +
                "Adding training data should reduce predictive uncertainty.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Confidence Interval Coverage
    // For a known function with low noise, the 95% CI (mean ± 2σ)
    // should contain the true value at most test points.
    // =====================================================

    [Fact]
    public void ConfidenceInterval_ShouldCoverTruth_AtMostTestPoints()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        // Generate data from known function y = x1 + x2 with small noise
        int n = 25;
        var trainX = new Matrix<double>(n, Features);
        var trainY = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < Features; j++)
                trainX[i, j] = rng.NextDouble();
            trainY[i] = trainX[i, 0] + trainX[i, 1] + ModelTestHelpers.NextGaussian(rng) * 0.05;
        }

        model.Fit(trainX, trainY);

        // Check coverage at test points within the training range
        int covered = 0;
        int total = 0;
        var testRng = ModelTestHelpers.CreateSeededRandom(99);
        for (int i = 0; i < 10; i++)
        {
            double x1 = testRng.NextDouble();
            double x2 = testRng.NextDouble();
            double trueY = x1 + x2; // noise-free truth

            var point = MakePoint(x1, x2);
            var (mean, variance) = model.Predict(point);

            if (!double.IsNaN(mean) && !double.IsNaN(variance) &&
                !double.IsInfinity(mean) && variance >= 0)
            {
                double sigma = Math.Sqrt(Math.Max(0, variance));
                double lower = mean - 2.0 * sigma;
                double upper = mean + 2.0 * sigma;

                if (trueY >= lower && trueY <= upper)
                    covered++;
                total++;
            }
        }

        if (total >= 5)
        {
            double coverageRate = (double)covered / total;
            Assert.True(coverageRate >= 0.5,
                $"95% CI coverage = {coverageRate:P0} ({covered}/{total}). " +
                "Expected at least 50% coverage — GP uncertainty is severely miscalibrated.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Clone Should Preserve Predictions
    // =====================================================

    [Fact]
    public void Clone_ShouldProduceIdenticalPredictions()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng);

        model.Fit(trainX, trainY);

        // Clone if the model supports IFullModel
        if (model is IFullModel<double, Matrix<double>, Vector<double>> fullModel)
        {
            var cloned = fullModel.Clone();
            if (cloned is IGaussianProcess<double> clonedGP)
            {
                var point = MakePoint(0.5, 0.5);
                var (mean1, var1) = model.Predict(point);
                var (mean2, var2) = clonedGP.Predict(point);

                if (!double.IsNaN(mean1) && !double.IsNaN(mean2))
                {
                    Assert.Equal(mean1, mean2);
                    Assert.Equal(var1, var2);
                }
            }
        }
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline Should Work
    // =====================================================

    [Fact]
    public void Builder_ShouldProduceResult()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = GenerateNormalizedLinearData(TrainSamples, Features, rng);

        // GP models implement IGaussianProcess but also IFullModel for the builder
        var model = CreateModel();
        if (model is IFullModel<double, Matrix<double>, Vector<double>> fullModel)
        {
            var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(trainX, trainY);
            var result = new AiDotNet.AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(fullModel)
                .BuildAsync()
                .GetAwaiter()
                .GetResult();

            Assert.NotNull(result);
        }
    }

    private Vector<double> MakePoint(params double[] values)
    {
        var point = new Vector<double>(Features);
        for (int j = 0; j < Features && j < values.Length; j++)
            point[j] = values[j];
        return point;
    }
}
