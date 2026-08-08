using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for anomaly detection models.
/// Tests mathematical invariants: normal data scores, outlier detection,
/// score finiteness, determinism, monotonicity, and clone consistency.
/// </summary>
/// <remarks>
/// Anomaly detectors use IFullModel&lt;T, Matrix&lt;T&gt;, Vector&lt;T&gt;&gt; where
/// the output vector contains anomaly scores (higher = more anomalous).
/// </remarks>
/// <typeparam name="T">
/// Numeric precision of the generated fixture. Generic so a training-bound detector can be routed to
/// <c>&lt;float&gt;</c> (the scaffold's Fp32 selection) to fit the shard budget instead of being
/// deferred out of it; the non-generic <see cref="AnomalyDetectorTestBase"/> alias below keeps every
/// existing <c>&lt;double&gt;</c> fixture source-compatible.
/// </typeparam>
public abstract class AnomalyDetectorTestBase<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a double literal into the fixture's numeric type.</summary>
    protected static T ToT(double value) => NumOps.FromDouble(value);

    /// <summary>Converts a fixture-typed value back to double for finiteness / magnitude asserts.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateModel();

    protected virtual int TrainSamples => 100;
    protected virtual int Features => 3;

    private (Matrix<T> X, Vector<T> Y) GenerateNormalData(Random rng)
    {
        var x = new Matrix<T>(TrainSamples, Features);
        var y = new Vector<T>(TrainSamples);
        for (int i = 0; i < TrainSamples; i++)
        {
            for (int j = 0; j < Features; j++)
                x[i, j] = ToT(ModelTestHelpers.NextGaussian(rng) * 1.0); // centered at 0
            y[i] = NumOps.Zero; // normal label
        }
        return (x, y);
    }

    [Fact(Timeout = 60000)]
    public async Task Outliers_ShouldHaveHigherScores()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        // Normal test points
        var normalX = new Matrix<T>(5, Features);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < Features; j++)
                normalX[i, j] = ToT(ModelTestHelpers.NextGaussian(rng) * 1.0);

        // Outlier test points (far from training distribution)
        var outlierX = new Matrix<T>(5, Features);
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < Features; j++)
                outlierX[i, j] = ToT(50.0 + ModelTestHelpers.NextGaussian(rng) * 0.1);

        var normalScores = model.Predict(normalX);
        var outlierScores = model.Predict(outlierX);

        // NON-FINITE IS A FAILURE, NOT A REASON TO SKIP. Wrapping the invariant in a
        // finiteness check meant a detector returning NaN or Infinity satisfied this test
        // by producing nothing checkable -- the worse the model behaved, the more certainly
        // it passed. Finiteness is asserted first, then the invariant runs unconditionally.
        Assert.True(ModelTestHelpers.AllFinite(normalScores),
            "Anomaly scores for normal points contain NaN or Infinity.");
        Assert.True(ModelTestHelpers.AllFinite(outlierScores),
            "Anomaly scores for outlier points contain NaN or Infinity.");

        {
            double normalMean = 0, outlierMean = 0;
            for (int i = 0; i < normalScores.Length; i++) normalMean += ToD(normalScores[i]);
            for (int i = 0; i < outlierScores.Length; i++) outlierMean += ToD(outlierScores[i]);
            normalMean /= normalScores.Length;
            outlierMean /= outlierScores.Length;

            // DIRECTIONAL, NOT MERELY DIFFERENT. |outlier - normal| > 1e-6 was satisfied by a
            // detector that scored outliers LOWER than normal points -- an inverted detector,
            // which is worse than an insensitive one because every downstream threshold then
            // selects exactly the wrong rows. The method is named Outliers_ShouldHaveHigherScores
            // and that is the contract asserted here.
            Assert.True(outlierMean > normalMean,
                $"Normal mean score = {normalMean:F4}, outlier mean = {outlierMean:F4}. " +
                "Outliers must score HIGHER than normal points; an equal score means the detector " +
                "does not distinguish them, and a lower score means it ranks them backwards.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Scores_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        var scores = model.Predict(trainX);
        for (int i = 0; i < scores.Length; i++)
        {
            double s = ToD(scores[i]);
            Assert.False(double.IsNaN(s), $"Anomaly score[{i}] is NaN.");
            Assert.False(double.IsInfinity(s), $"Anomaly score[{i}] is Infinity.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        var scores1 = model.Predict(trainX);
        var scores2 = model.Predict(trainX);
        for (int i = 0; i < scores1.Length; i++)
            Assert.Equal(scores1[i], scores2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Clone_ShouldProduceSameScores()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);

        var cloned = model.Clone();
        var scores1 = model.Predict(trainX);
        var scores2 = cloned.Predict(trainX);
        for (int i = 0; i < scores1.Length; i++)
            Assert.Equal(scores1[i], scores2[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task OutputDimension_ShouldMatchInputRows()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);
        Assert.Equal(TrainSamples, model.Predict(trainX).Length);
    }

    [Fact(Timeout = 60000)]
    public async Task Metadata_ShouldExistAfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 60000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var (trainX, trainY) = GenerateNormalData(rng);
        model.Train(trainX, trainY);
        Assert.True(((IParameterizable<T, Matrix<T>, Vector<T>>)model).GetParameters().Length > 0, "Trained anomaly detector should have parameters.");
    }
}

/// <summary>
/// Double-precision alias so existing generated anomaly-detector fixtures keep deriving from a
/// non-generic base; <c>&lt;float&gt;</c> fixtures derive from <see cref="AnomalyDetectorTestBase{T}"/>
/// directly.
/// </summary>
public abstract class AnomalyDetectorTestBase : AnomalyDetectorTestBase<double> { }
