using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for regression models implementing IFullModel&lt;T, Matrix&lt;T&gt;, Vector&lt;T&gt;&gt;.
/// Tests deep mathematical invariants that any correctly implemented regression model must satisfy.
/// </summary>
/// <remarks>
/// Fixtures and assertion math remain in double precision so the invariants retain their original
/// meaning. Only the model's Train/Predict boundary is converted to <typeparamref name="T"/>.
/// </remarks>
public abstract class RegressionModelTestBase<T> : System.IDisposable
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected static Matrix<T> ToT(Matrix<double> matrix)
    {
        if (typeof(T) == typeof(double)) return (Matrix<T>)(object)matrix;

        var converted = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int row = 0; row < matrix.Rows; row++)
            for (int column = 0; column < matrix.Columns; column++)
                converted[row, column] = NumOps.FromDouble(matrix[row, column]);
        return converted;
    }

    protected static Vector<T> ToT(Vector<double> vector)
    {
        if (typeof(T) == typeof(double)) return (Vector<T>)(object)vector;

        var converted = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            converted[i] = NumOps.FromDouble(vector[i]);
        return converted;
    }

    protected static Vector<double> ToD(Vector<T> vector)
    {
        if (typeof(T) == typeof(double)) return (Vector<double>)(object)vector;

        var converted = new Vector<double>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
            converted[i] = NumOps.ToDouble(vector[i]);
        return converted;
    }

    /// <summary>
    /// Reclaim memory between tests (shared model-family teardown). xUnit constructs a fresh
    /// test-class instance per test and calls Dispose() afterward, so this clears the
    /// InferenceWeightCache and compacts the LOH between model classes — keeping committed memory
    /// from accumulating across a shard. Pure hygiene; no test-observable behavior change.
    /// </summary>
    public virtual void Dispose()
    {
        // Reclaim must be unconditional: a throwing derived DisposeCore() must not skip the
        // shared GC gate, or heavy shards reintroduce cross-test memory buildup / OOM.
        try
        {
            DisposeCore();
        }
        finally
        {
            ModelFamilyTestGcGate.ReclaimBetweenTests();
        }
    }

    /// <summary>
    /// Override in a derived test class to add its own teardown while preserving the
    /// shared <see cref="ModelFamilyTestGcGate.ReclaimBetweenTests"/> call.
    /// </summary>
    protected virtual void DisposeCore()
    {
    }

    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateModel();

    /// <summary>
    /// Converts a generated target into the domain this model accepts, then into <typeparamref name="T"/>.
    /// </summary>
    /// <param name="y">The continuous linear target this base class generates.</param>
    /// <returns>The target actually passed to <c>Train</c>.</returns>
    /// <remarks>
    /// <para>
    /// This harness generates one shape of target: an unrestricted continuous response. Most estimators
    /// here accept that directly, and the default implementation passes it straight through.
    /// </para>
    /// <para>
    /// Some do not. A classifier needs class labels, and Beta regression needs a proportion in (0,1);
    /// handed a continuous unbounded target, the honest response is to reject it. Overriding this lets
    /// such a model keep running EVERY structural invariant below -- output dimension, determinism,
    /// cloning, serialization, metadata, collinear features -- on data it can actually be fitted to,
    /// rather than opting out of them.
    /// </para>
    /// <para>
    /// Only the mapping of the target changes; the features, the seeds and the assertions are untouched,
    /// so a model that overrides this is held to the same contract as every other. Override it whenever
    /// the model's response domain is narrower than "any real number", and say in the override which
    /// domain it requires.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> ToTarget(Vector<double> y) => ToT(y);

    /// <summary>
    /// Thresholds a continuous target at its median, giving two balanced classes labelled 0 and 1.
    /// </summary>
    /// <param name="y">The continuous target.</param>
    /// <returns>A binary target preserving the ordering of <paramref name="y"/>.</returns>
    /// <remarks>
    /// The median split keeps the classes balanced and keeps the label monotone in the original target,
    /// so invariants that expect predictions to rise with a feature still mean what they meant before.
    /// </remarks>
    protected static Vector<double> ThresholdAtMedian(Vector<double> y)
    {
        var sorted = new double[y.Length];
        for (int i = 0; i < y.Length; i++) sorted[i] = y[i];
        Array.Sort(sorted);
        double median = sorted[sorted.Length / 2];

        var labels = new Vector<double>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            labels[i] = y[i] >= median ? 1.0 : 0.0;
        }

        return labels;
    }

    /// <summary>
    /// Bins a continuous target into <paramref name="classCount"/> equal-width, ordered classes.
    /// </summary>
    /// <param name="y">The continuous target.</param>
    /// <param name="classCount">How many classes to produce.</param>
    /// <returns>A target holding class labels 0..classCount-1, monotone in <paramref name="y"/>.</returns>
    protected static Vector<double> BinIntoClasses(Vector<double> y, int classCount)
    {
        double min = double.MaxValue;
        double max = double.MinValue;
        for (int i = 0; i < y.Length; i++)
        {
            if (y[i] < min) min = y[i];
            if (y[i] > max) max = y[i];
        }

        double range = max - min;
        if (range < 1e-10) range = 1.0;

        var labels = new Vector<double>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            int bin = (int)((y[i] - min) / range * classCount);
            labels[i] = Math.Max(0, Math.Min(classCount - 1, bin));
        }

        return labels;
    }

    /// <summary>
    /// Squashes a continuous target into the open interval (0,1) with a logistic transform.
    /// </summary>
    /// <param name="y">The continuous target.</param>
    /// <returns>A proportion strictly inside (0,1), monotone in <paramref name="y"/>.</returns>
    /// <remarks>
    /// Standardizing first and then applying the logistic function keeps the result strictly interior --
    /// never exactly 0 or 1, where the Beta density is undefined -- and strictly increasing, so ordering
    /// is preserved.
    /// </remarks>
    protected static Vector<double> ToProportion(Vector<double> y)
    {
        double mean = 0.0;
        for (int i = 0; i < y.Length; i++) mean += y[i];
        mean /= y.Length;

        double variance = 0.0;
        for (int i = 0; i < y.Length; i++) variance += (y[i] - mean) * (y[i] - mean);
        double std = Math.Sqrt(variance / y.Length);
        if (std < 1e-10) std = 1.0;

        var proportions = new Vector<double>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            proportions[i] = 1.0 / (1.0 + Math.Exp(-(y[i] - mean) / std));
        }

        return proportions;
    }

    /// <summary>
    /// True when the monotonic-response invariant applies to this model.
    /// </summary>
    /// <remarks>
    /// The invariant trains on linear data and then probes a feature from -5 to 15 while holding
    /// the others fixed, expecting predictions to rise. That assumes the model extrapolates
    /// linearly, which a KERNEL method does not: an RBF kernel's influence decays with distance,
    /// so outside the training support its prediction falls back toward the bias rather than
    /// continuing to climb. Non-monotonicity there is the defining behaviour of the kernel, not a
    /// failure to learn the coefficient direction. Override to <c>false</c> for kernel models, with
    /// a comment naming the kernel involved.
    /// </remarks>
    protected virtual bool MonotonicResponseInvariantApplicable => true;

    /// <summary>
    /// True when the held-out predictive-quality invariants (positive R-squared, coefficient signs
    /// matching the data-generating process) apply to this model as the harness configures it.
    /// </summary>
    /// <remarks>
    /// These assume the model can be fitted meaningfully to the generic continuous linear data this
    /// base class generates. That is not true for a model whose required configuration is degenerate
    /// against that data — a mixed model, for instance, must be given a grouping column, and the
    /// harness only has continuous features to offer, so every observation becomes its own group.
    /// The random effects then absorb all the variance, held-out predictions collapse, and the
    /// failure measures the harness's data rather than the estimator. Override to <c>false</c> with
    /// a comment naming the specific mismatch.
    /// </remarks>
    protected virtual bool PredictiveQualityInvariantsApplicable => true;

    /// <summary>
    /// True when this model is an additive, identity-link estimator over an unrestricted continuous
    /// response — the assumption behind the equivariance and generic-quality invariants below.
    /// </summary>
    /// <remarks>
    /// A generalized linear model with a non-identity link is <b>multiplicative</b>, not additive:
    /// under a log link, adding a constant to every target does not add that constant to the
    /// predictions, it rescales them. Such models also restrict their response domain — Gamma and
    /// Inverse Gaussian require strictly positive targets, Negative Binomial requires counts — while
    /// this harness generates unrestricted linear data that can be negative. Asserting translation
    /// equivariance, residual-mean-zero on the response scale, or R-squared against that data
    /// measures the mismatch rather than the estimator.
    ///
    /// These models are exercised on domain-appropriate data by
    /// <c>GLMFamilyRegressionIntegrationTests</c> instead, which is where their correctness is
    /// actually established. Override to <c>false</c> for any non-identity-link GLM.
    /// </remarks>
    protected virtual bool IdentityLinkInvariantsApplicable => true;

    protected virtual int TrainSamples => 100;
    protected virtual int TestSamples => 30;
    protected virtual int Features => 3;

    /// <summary>
    /// Column indices that the model consumes as STRUCTURE rather than as predictors, and which
    /// therefore have no fixed effect of their own.
    /// </summary>
    /// <remarks>
    /// A mixed model given <c>y ~ x + (1|group)</c> treats the grouping column as a factor, not as
    /// a numeric predictor — exactly as lme4 does — so asserting that it has a positive marginal
    /// effect asks the model for something it is not meant to provide. Naming those columns here
    /// keeps the invariant enforced on every genuine predictor instead of switching it off wholesale.
    /// </remarks>
    protected virtual ISet<int> StructuralFeatureIndices => new HashSet<int>();

    /// <summary>
    /// Whether the model is expected to reproduce the ORDER of the generating coefficients, not
    /// merely their signs.
    /// </summary>
    /// <remarks>
    /// GenerateLinearData uses increasing coefficients, so a linear fit's marginal effects come out
    /// in the same order anywhere you probe them. A nonlinear model has no such guarantee: its
    /// marginal effect varies from point to point, and at one probe location the ordering can
    /// invert while the fit as a whole is perfectly good. The sign assertions still apply and are
    /// the substantive check; this only governs the comparison between two features.
    /// </remarks>
    protected virtual bool CoefficientOrderingInvariantApplicable => true;

    // =====================================================
    // MATHEMATICAL INVARIANT: Translation Equivariance
    // Shifting all targets by constant C must shift predictions by C.
    // Any regression model violating this has a bias bug.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task TranslationEquivariance_ShiftingTargets_ShiftsPredictions()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng1, noise: 0.01);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng2, noise: 0.01);

        const double shift = 1000.0;
        var shiftedY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            shiftedY[i] = trainY2[i] + shift;

        model1.Train(ToT(trainX1), ToTarget(trainY1));
        model2.Train(ToT(trainX2), ToTarget(shiftedY));

        var testX = ModelTestHelpers.GenerateLinearData(10, Features, ModelTestHelpers.CreateSeededRandom(99), noise: 0.0).X;
        var pred1 = ToD(model1.Predict(ToT(testX)));
        var pred2 = ToD(model2.Predict(ToT(testX)));

        Assert.True(
            ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2),
            "Translation-equivariance predictions must be finite.");
        for (int i = 0; i < pred1.Length; i++)
        {
            double actualShift = pred2[i] - pred1[i];
            Assert.True(Math.Abs(actualShift - shift) < shift * 0.3,
                $"Translation equivariance violated: predicted shift = {actualShift:F2}, expected ~{shift}. " +
                $"pred_original={pred1[i]:F4}, pred_shifted={pred2[i]:F4}");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaling Equivariance
    // Scaling all targets by factor K must scale predictions by K.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task ScalingEquivariance_ScalingTargets_ScalesPredictions()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX1, trainY1) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng1, noise: 0.01);
        var (trainX2, trainY2) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng2, noise: 0.01);

        const double scale = 100.0;
        var scaledY = new Vector<double>(trainY2.Length);
        for (int i = 0; i < trainY2.Length; i++)
            scaledY[i] = trainY2[i] * scale;

        model1.Train(ToT(trainX1), ToTarget(trainY1));
        model2.Train(ToT(trainX2), ToTarget(scaledY));

        var testX = ModelTestHelpers.GenerateLinearData(10, Features, ModelTestHelpers.CreateSeededRandom(99), noise: 0.0).X;
        var pred1 = ToD(model1.Predict(ToT(testX)));
        var pred2 = ToD(model2.Predict(ToT(testX)));

        Assert.True(
            ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2),
            "Scaling-equivariance predictions must be finite.");
        for (int i = 0; i < pred1.Length; i++)
        {
            if (Math.Abs(pred1[i]) > 0.01)
            {
                double ratio = pred2[i] / pred1[i];
                Assert.True(ratio > scale * 0.5 && ratio < scale * 2.0,
                    $"Scaling equivariance violated at sample {i}: ratio = {ratio:F2}, expected ~{scale}. " +
                    $"pred_original={pred1[i]:F4}, pred_scaled={pred2[i]:F4}");
            }
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Error ≤ Test Error
    // On average, the model should fit training data at least as well as unseen test data.
    // Violation indicates a bug in Train or Predict.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task TrainingError_ShouldNotExceedTestError_OnAverage()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng, noise: 0.5);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng, noise: 0.5);

        model.Train(ToT(trainX), ToTarget(trainY));
        var trainPred = ToD(model.Predict(ToT(trainX)));
        var testPred = ToD(model.Predict(ToT(testX)));

        Assert.True(
            ModelTestHelpers.AllFinite(trainPred) && ModelTestHelpers.AllFinite(testPred),
            "Training- and test-error predictions must be finite.");
        double trainMSE = ModelTestHelpers.CalculateMSE(trainY, trainPred);
        double testMSE = ModelTestHelpers.CalculateMSE(testY, testPred);

        // Training MSE should generally be ≤ test MSE (allow 2x slack for variance)
        Assert.True(trainMSE <= testMSE * 2.0 + 1e-10,
            $"Training MSE ({trainMSE:F4}) is much higher than test MSE ({testMSE:F4}). " +
            "This suggests the model is not actually fitting the training data.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: More Data → Better or Equal Fit
    // Doubling training data should not make R² worse by more than noise.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task MoreData_ShouldNotDegrade_R2()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        if (!PredictiveQualityInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var (trainX1, trainY1) = ModelTestHelpers.GenerateLinearData(30, Features, rng1, noise: 0.1);

        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model2 = CreateModel();
        var (trainX2, trainY2) = ModelTestHelpers.GenerateLinearData(120, Features, rng2, noise: 0.1);

        // Fixed test set
        var rngTest = ModelTestHelpers.CreateSeededRandom(99);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(50, Features, rngTest, noise: 0.1);

        model1.Train(ToT(trainX1), ToTarget(trainY1));
        model2.Train(ToT(trainX2), ToTarget(trainY2));

        var pred1 = ToD(model1.Predict(ToT(testX)));
        var pred2 = ToD(model2.Predict(ToT(testX)));

        Assert.True(
            ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2),
            "More-data invariant predictions must be finite.");
        double r2Small = ModelTestHelpers.CalculateR2(testY, pred1);
        double r2Large = ModelTestHelpers.CalculateR2(testY, pred2);

        // Model with 4x data should be at least as good (allow 0.15 margin for stochasticity)
        Assert.True(r2Large >= r2Small - 0.15,
            $"4x more data made R² worse: R²(30)={r2Small:F4}, R²(120)={r2Large:F4}. " +
            "Model may not be correctly learning from additional data.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Irrelevant Feature Should Not Help
    // Adding a random noise feature should not improve predictions.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task IrrelevantFeature_ShouldNotImprove_Predictions()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        if (!PredictiveQualityInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        if (Features < 2)
        {
            // Univariate models (e.g., SimpleRegression) can't compare N vs N+1 features
            // since they only accept exactly 1 feature. Skip this test.
            return;
        }

        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX_real, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng1, noise: 0.1);
        var (testX_real, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng2, noise: 0.1);

        // Create version with 1 added noise feature
        var rngNoise = ModelTestHelpers.CreateSeededRandom(77);
        var trainX_noisy = new Matrix<double>(TrainSamples, Features + 1);
        var testX_noisy = new Matrix<double>(TestSamples, Features + 1);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                trainX_noisy[i, j] = trainX_real[i, j];
            trainX_noisy[i, Features] = rngNoise.NextDouble() * 100.0; // pure noise
        }
        for (int i = 0; i < TestSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                testX_noisy[i, j] = testX_real[i, j];
            testX_noisy[i, Features] = rngNoise.NextDouble() * 100.0;
        }

        model1.Train(ToT(trainX_real), ToTarget(trainY));
        model2.Train(ToT(trainX_noisy), ToTarget(trainY));

        var pred1 = ToD(model1.Predict(ToT(testX_real)));
        var pred2 = ToD(model2.Predict(ToT(testX_noisy)));

        Assert.True(
            ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2),
            "Irrelevant-feature invariant predictions must be finite.");
        double r2Real = ModelTestHelpers.CalculateR2(testY, pred1);
        double r2Noisy = ModelTestHelpers.CalculateR2(testY, pred2);

        // Adding noise feature should not improve R² substantially
        Assert.True(r2Noisy <= r2Real + 0.15,
            $"Adding irrelevant noise feature improved R²: clean={r2Real:F4}, noisy={r2Noisy:F4}. " +
            "Model may be overfitting to noise.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Monotonic Response
    // For data y = 2*x1 + 4*x2 + 1, increasing x1 while holding x2 constant
    // must increase prediction. Tests the model learned correct sign/direction.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task MonotonicResponse_IncreasingFeature_IncreasesPrediction()
    {
        await Task.Yield();
        if (!MonotonicResponseInvariantApplicable) return;

        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        int nFeatures = Math.Max(Features, 1);
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(200, nFeatures, rng, noise: 0.01);

        model.Train(ToT(trainX), ToTarget(trainY));

        // Create probe with x0 varying from -5 to 15, other features fixed at 5
        var probe = new Matrix<double>(5, nFeatures);
        for (int i = 0; i < 5; i++)
        {
            probe[i, 0] = i * 5.0 - 5.0; // -5, 0, 5, 10, 15
            for (int j = 1; j < nFeatures; j++)
                probe[i, j] = 5.0;
        }

        var predictions = ToD(model.Predict(ToT(probe)));
        Assert.True(
            ModelTestHelpers.AllFinite(predictions),
            "Monotonic-response predictions must be finite.");
        int monotoneViolations = 0;
        for (int i = 1; i < predictions.Length; i++)
        {
            if (predictions[i] < predictions[i - 1])
                monotoneViolations++;
        }
        Assert.True(monotoneViolations <= 1,
            $"Monotonicity violated {monotoneViolations}/4 times. " +
            $"Predictions: [{string.Join(", ", Enumerable.Range(0, predictions.Length).Select(i => predictions[i].ToString("F2")))}]. " +
            "Model failed to learn positive coefficient direction.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Residual Mean ≈ 0
    // For unbiased estimators, the mean of residuals should be near zero.
    // Large residual mean indicates systematic bias in the model.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task ResidualMean_ShouldBeNearZero()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(200, Features, rng, noise: 0.5);

        model.Train(ToT(trainX), ToTarget(trainY));
        var predictions = ToD(model.Predict(ToT(trainX)));

        Assert.True(
            ModelTestHelpers.AllFinite(predictions),
            "Residual-mean predictions must be finite.");
        double residualSum = 0;
        double minY = double.MaxValue, maxY = double.MinValue;
        for (int i = 0; i < trainY.Length; i++)
        {
            residualSum += trainY[i] - predictions[i];
            if (trainY[i] < minY) minY = trainY[i];
            if (trainY[i] > maxY) maxY = trainY[i];
        }
        double targetRange = maxY - minY;
        double meanResidual = residualSum / trainY.Length;

        // Mean residual should be small relative to the target range
        Assert.True(Math.Abs(meanResidual) < targetRange * 0.1,
            $"Mean residual = {meanResidual:F4} is large relative to target range {targetRange:F4}. " +
            "Model has systematic prediction bias.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Coefficient Sign Recovery
    // For y = 2*x1 + 4*x2 + 1 with low noise, probing must show
    // both features have positive effect on prediction.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task CoefficientSigns_ShouldMatchDataGeneratingProcess()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        if (!PredictiveQualityInvariantsApplicable) return;

        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        int nFeatures = Math.Max(Features, 1);
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(200, nFeatures, rng, noise: 0.01);

        model.Train(ToT(trainX), ToTarget(trainY));

        // Probe: baseline at origin, then increase each feature independently
        int probeRows = 1 + nFeatures; // baseline + one per feature
        var probe = new Matrix<double>(probeRows, nFeatures);
        // Row 0 = baseline (all zeros)
        for (int f = 0; f < nFeatures; f++)
        {
            probe[1 + f, f] = 10.0; // increase feature f by 10
        }

        var predictions = ToD(model.Predict(ToT(probe)));
        Assert.True(
            ModelTestHelpers.AllFinite(predictions),
            "Coefficient-sign predictions must be finite.");
        // Each feature should have positive effect (GenerateLinearData uses positive coefficients).
        // Structural columns are skipped: a mixed model's grouping column is a factor rather than
        // a predictor, so it has no fixed effect to check the sign of.
        var structural = StructuralFeatureIndices;

        for (int f = 0; f < nFeatures; f++)
        {
            if (structural.Contains(f)) continue;

            double effect = predictions[1 + f] - predictions[0];
            Assert.True(effect > 0,
                $"Feature x{f} effect = {effect:F4}, expected positive. " +
                "Model learned wrong sign.");
        }

        // GenerateLinearData uses increasing coefficients, so a later predictor should outweigh an
        // earlier one. Compare the first two columns that are actually predictors.
        var predictors = Enumerable.Range(0, nFeatures).Where(f => !structural.Contains(f)).ToArray();

        if (predictors.Length >= 2 && CoefficientOrderingInvariantApplicable)
        {
            int lower = predictors[0];
            int upper = predictors[1];

            double effectLower = predictions[1 + lower] - predictions[0];
            double effectUpper = predictions[1 + upper] - predictions[0];

            Assert.True(effectUpper > effectLower,
                $"Feature x{upper} effect ({effectUpper:F4}) should be larger than "
                + $"x{lower} ({effectLower:F4}).");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Permutation Consistency
    // Permuting feature columns and correspondingly permuting any learned
    // structure should give equivalent predictions.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task FeaturePermutation_ShouldGiveConsistentPredictions()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        if (Features < 2)
        {
            // Feature permutation requires at least 2 features to swap.
            // Univariate models pass trivially (no permutation possible).
            return;
        }

        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);
        var model1 = CreateModel();
        var model2 = CreateModel();

        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng1, noise: 0.01);

        // Create permuted version: swap columns 0 and 1
        var permutedX = new Matrix<double>(TrainSamples, Features);
        for (int i = 0; i < TrainSamples; i++)
        {
            permutedX[i, 0] = trainX[i, 1]; // swap first two
            permutedX[i, 1] = trainX[i, 0];
            for (int j = 2; j < Features; j++)
                permutedX[i, j] = trainX[i, j]; // keep rest
        }

        model1.Train(ToT(trainX), ToTarget(trainY));
        model2.Train(ToT(permutedX), ToTarget(trainY));

        // Test with a specific point
        var testOrig = new Matrix<double>(1, Features);
        var testPerm = new Matrix<double>(1, Features);
        testOrig[0, 0] = 3.0; testOrig[0, 1] = 7.0;
        testPerm[0, 0] = 7.0; testPerm[0, 1] = 3.0; // swapped
        for (int j = 2; j < Features; j++)
        {
            testOrig[0, j] = 5.0;
            testPerm[0, j] = 5.0;
        }

        var pred1 = ToD(model1.Predict(ToT(testOrig)));
        var pred2 = ToD(model2.Predict(ToT(testPerm)));

        Assert.True(
            ModelTestHelpers.AllFinite(pred1) && ModelTestHelpers.AllFinite(pred2),
            "Feature-permutation predictions must be finite.");
        Assert.True(Math.Abs(pred1[0] - pred2[0]) < Math.Abs(pred1[0]) * 0.2 + 1.0,
            $"Feature permutation inconsistency: pred_orig={pred1[0]:F4}, pred_permuted={pred2[0]:F4}. " +
            "Swapping feature columns and correspondingly swapping test inputs should give ~same prediction.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: R² > 0 on Linear Data
    // Any regression model should outperform the mean baseline on data
    // that is actually linear. R²≤0 means the model is worse than guessing the mean.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task R2_ShouldBePositive_OnLinearData()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        if (!PredictiveQualityInvariantsApplicable) return;

        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng, noise: 0.1);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng, noise: 0.1);

        model.Train(ToT(trainX), ToTarget(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

        Assert.True(
            ModelTestHelpers.AllFinite(predictions),
            "Linear-data predictions must be finite.");
        double r2 = ModelTestHelpers.CalculateR2(testY, predictions);
        Assert.True(r2 > 0.0,
            $"R² = {r2:F4} on linear data — model is worse than predicting the mean. " +
            "Either the model is not learning, or Train/Predict has a bug.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Predictions Are Finite
    // No NaN, no Infinity. Violations indicate numerical instability.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Predictions_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));
        var predictions = ToD(model.Predict(ToT(testX)));

        Assert.Equal(TestSamples, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN — numerical instability in model.");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction[{i}] is Infinity — overflow in model computation.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Deterministic Prediction
    // Same trained model + same input = same output. Always.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));
        var pred1 = ToD(model.Predict(ToT(testX)));
        var pred2 = ToD(model.Predict(ToT(testX)));

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Dimension
    // Predict(N×F matrix) must return length-N vector.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task OutputDimension_ShouldMatchInputRows()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));

        // Test with various sample counts
        foreach (int n in new[] { 1, 5, 50 })
        {
            var testX = ModelTestHelpers.GenerateLinearData(n, Features, ModelTestHelpers.CreateSeededRandom(n), noise: 0.0).X;
            var pred = ToD(model.Predict(ToT(testX)));
            Assert.Equal(n, pred.Length);
        }
    }

    // =====================================================
    // CONTRACT: Clone Produces Identical Predictions
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceIdenticalPredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, _) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));
        var cloned = model.Clone();

        var pred1 = ToD(model.Predict(ToT(testX)));
        var pred2 = ToD(cloned.Predict(ToT(testX)));

        for (int i = 0; i < pred1.Length; i++)
            Assert.Equal(pred1[i], pred2[i]);
    }

    // =====================================================
    // CONTRACT: Metadata Should Exist After Training
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Metadata_ShouldExistAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));
        Assert.NotNull(model.GetModelMetadata());
    }

    // =====================================================
    // CONTRACT: Parameters Should Be Non-Empty After Training
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmpty_AfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));
        if (model is not IParameterizable<T, Matrix<T>, Vector<T>> paramModel)
        {
            // Tree/ensemble models don't implement IParameterizable — skip
            return;
        }
        var parameters = paramModel.GetParameters();
        Assert.True(parameters.Length > 0, "Trained model should have learnable parameters.");
    }

    // =====================================================
    // CONTRACT: Active Feature Indices Should Be Valid
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task ActiveFeatureIndices_ShouldBeValid()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);

        model.Train(ToT(trainX), ToTarget(trainY));
        if (model is not IFeatureAware featureAware)
        {
            // Tree/ensemble models don't implement IFeatureAware — skip
            return;
        }
        var activeFeatures = featureAware.GetActiveFeatureIndices().ToList();

        Assert.True(activeFeatures.Count > 0, "Trained model should have at least one active feature.");
        foreach (var idx in activeFeatures)
        {
            Assert.True(idx >= 0 && idx < Features,
                $"Active feature index {idx} is out of bounds [0, {Features}).");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Intercept Recovery
    // On constant data y = C, all predictions should equal C.
    // If not, the bias/intercept term is broken.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task InterceptRecovery_ConstantTarget_ShouldPredictConstant()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        int n = TrainSamples;
        var x = new Matrix<double>(n, Features);
        var y = new Vector<double>(n);
        const double constant = 7.5;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = rng.NextDouble() * 10.0;
            y[i] = constant;
        }

        model.Train(ToT(x), ToTarget(y));
        var predictions = ToD(model.Predict(ToT(x)));

        Assert.True(
            ModelTestHelpers.AllFinite(predictions),
            "Intercept-recovery predictions must be finite.");
        double meanPred = 0;
        for (int i = 0; i < predictions.Length; i++) meanPred += predictions[i];
        meanPred /= predictions.Length;

        Assert.True(Math.Abs(meanPred - constant) < constant * 0.3,
            $"Mean prediction = {meanPred:F4} on constant data (y={constant}). " +
            "Intercept/bias term may be broken.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Collinear Features Should Not Crash
    // Perfectly correlated features should not cause NaN/Infinity.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task CollinearFeatures_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        int n = TrainSamples;
        var x = new Matrix<double>(n, Features);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double val = rng.NextDouble() * 10.0;
            for (int j = 0; j < Features; j++)
                x[i, j] = val + j * 0.001; // nearly perfectly collinear
            y[i] = val * 2.0 + 1.0;
        }

        model.Train(ToT(x), ToTarget(y));
        var predictions = ToD(model.Predict(ToT(x)));

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN with collinear features — numerical instability.");
            Assert.False(double.IsInfinity(predictions[i]),
                $"Prediction[{i}] is Infinity with collinear features.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Single Feature Should Work
    // Regression model should handle 1-dimensional input.
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task SingleFeature_ShouldWork()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        int n = TrainSamples;
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = rng.NextDouble() * 10.0;
            y[i] = 3.0 * x[i, 0] + 1.0 + ModelTestHelpers.NextGaussian(rng) * 0.1;
        }

        model.Train(ToT(x), ToTarget(y));
        var predictions = ToD(model.Predict(ToT(x)));
        Assert.Equal(n, predictions.Length);

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.False(double.IsNaN(predictions[i]),
                $"Prediction[{i}] is NaN for 1-feature input.");
        }
    }

    // =====================================================
    // INTEGRATION: Builder Pipeline Produces Valid Result
    // =====================================================

    [Fact(Timeout = 60000)]
    public async Task Builder_ShouldProduceResult()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(ToT(trainX), ToTarget(trainY));

        var result = new AiDotNet.AiModelBuilder<T, Matrix<T>, Vector<T>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        Assert.NotNull(result);
    }

    [Fact(Timeout = 60000)]
    public async Task Builder_R2ShouldBePositive()
    {
        await Task.Yield();
        if (!IdentityLinkInvariantsApplicable) return;
        if (!PredictiveQualityInvariantsApplicable) return;
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var (trainX, trainY) = ModelTestHelpers.GenerateLinearData(TrainSamples, Features, rng);
        var (testX, testY) = ModelTestHelpers.GenerateLinearData(TestSamples, Features, rng);
        var loader = AiDotNet.Data.Loaders.DataLoaders.FromMatrixVector(ToT(trainX), ToTarget(trainY));

        var result = new AiDotNet.AiModelBuilder<T, Matrix<T>, Vector<T>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(CreateModel())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        var predictions = ToD(result.Predict(ToT(testX)));
        double r2 = ModelTestHelpers.CalculateR2(testY, predictions);
        Assert.True(r2 > 0.0,
            $"Builder pipeline R² = {r2:F4} — should be positive on linear data.");
    }
}

/// <summary>Double-precision compatibility shim for existing regression fixtures.</summary>
public abstract class RegressionModelTestBase : RegressionModelTestBase<double> { }
