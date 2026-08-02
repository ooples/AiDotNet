using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Verifies that <see cref="LFTAlgorithm{T, TInput, TOutput}"/> implements Learned Feature-Wise
/// Transformation (Tseng et al., arXiv:2001.08735) rather than a generic perturbation scheme.
/// </summary>
/// <remarks>
/// Four properties define the method and none is visible from a loss-is-finite smoke test: the
/// transform is centred on the identity so it perturbs without biasing; its spread passes through
/// softplus so it stays positive; it is RESAMPLED every application, which is what simulates many
/// domains rather than one extra fixed layer; and it is absent at inference. Each is asserted here.
/// </remarks>
public class LFTMechanismTests
{
    private const int C = 8;

    private static FeatureWiseTransformation<double> Make(
        double scale = 0.3, double bias = 0.5, int seed = 11, int channels = C)
        => new(channels, scale, bias, new Random(seed));

    private static Vector<double> Ones(int n = C)
    {
        var v = new Vector<double>(n);
        for (int i = 0; i < n; i++) v[i] = 1.0;
        return v;
    }

    // ------------------------------------------------- the transformation

    [Fact]
    public void PaperHyperparameterInitialization()
    {
        // "theta_gamma = 0.3, theta_beta = 0.5" — the paper's pre-determined values.
        var o = new LFTOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3));
        Assert.Equal(0.3, o.InitialScaleHyperparameter);
        Assert.Equal(0.5, o.InitialBiasHyperparameter);
        Assert.True(o.LearnTransformationHyperparameters);

        var fwt = Make();
        for (int c = 0; c < C; c++)
        {
            Assert.Equal(0.3, fwt.ScaleHyperparameters[c], 12);
            Assert.Equal(0.5, fwt.BiasHyperparameters[c], 12);
        }
    }

    [Fact]
    public void SpreadIsSoftplusOfTheHyperparameter_AndAlwaysPositive()
    {
        // softplus keeps the standard deviation positive under unconstrained optimization, so a
        // hyper-parameter driven negative by the learning-to-learn step cannot produce a negative
        // (meaningless) spread.
        var fwt = Make(scale: 0.3, bias: 0.5);
        Assert.Equal(Math.Log(1 + Math.Exp(0.3)), fwt.EffectiveScaleStdDev(0), 12);
        Assert.Equal(Math.Log(1 + Math.Exp(0.5)), fwt.EffectiveBiasStdDev(0), 12);

        var negative = Make(scale: -5.0, bias: -5.0);
        Assert.True(negative.EffectiveScaleStdDev(0) > 0.0);
        Assert.True(negative.EffectiveBiasStdDev(0) > 0.0);

        // And it must not overflow for a large hyper-parameter.
        var large = Make(scale: 1000.0, bias: 1000.0);
        Assert.True(double.IsFinite(large.EffectiveScaleStdDev(0)));
        Assert.Equal(1000.0, large.EffectiveScaleStdDev(0), 6);
    }

    [Fact]
    public void TransformIsCentredOnTheIdentity()
    {
        // gamma ~ N(1, .), beta ~ N(0, .), so E[gamma * z + beta] = z. Averaged over many draws the
        // transform must leave the feature where it was. A gamma centred on 0 — the easy mistake —
        // would drive the mean to zero instead.
        var fwt = Make(seed: 5);
        var input = Ones();

        const int draws = 20000;
        var mean = new double[C];
        for (int d = 0; d < draws; d++)
        {
            var outp = fwt.Apply(input);
            for (int c = 0; c < C; c++) mean[c] += outp[c];
        }
        for (int c = 0; c < C; c++)
        {
            mean[c] /= draws;
            Assert.Equal(1.0, mean[c], 1);   // input was 1.0; identity in expectation
        }
    }

    [Fact]
    public void TransformIsResampledEveryApplication()
    {
        // The variation ACROSS draws is the mechanism — it is what "simulate various feature
        // distributions under different domains" means. A fixed draw would be one more deterministic
        // layer for the encoder to absorb.
        var fwt = Make(seed: 7);
        var input = Ones();

        var a = fwt.Apply(input);
        var b = fwt.Apply(input);

        bool differ = Enumerable.Range(0, C).Any(c => Math.Abs(a[c] - b[c]) > 1e-12);
        Assert.True(differ, "Two applications produced identical output; gamma/beta were not resampled.");
    }

    [Fact]
    public void LargerHyperparameterProducesWiderSpread()
    {
        // The hyper-parameter IS the spread, so raising it must widen the distribution — that is the
        // quantity the learning-to-learn stage is tuning.
        double Spread(double theta)
        {
            var fwt = Make(scale: theta, bias: theta, seed: 3);
            var input = Ones();
            const int draws = 4000;
            double sum = 0, sumSq = 0;
            for (int d = 0; d < draws; d++)
            {
                double v = fwt.Apply(input)[0];
                sum += v; sumSq += v * v;
            }
            double mean = sum / draws;
            return Math.Sqrt(sumSq / draws - mean * mean);
        }

        Assert.True(Spread(2.0) > Spread(0.3),
            "A larger hyper-parameter must widen the sampled transform's spread.");
    }

    [Fact]
    public void ChannelsBeyondTheDeclaredWidthArePassedThrough()
    {
        // Truncating would change the encoder's output width rather than perturb it.
        var fwt = Make(channels: 3);
        var input = new Vector<double>(6);
        for (int i = 0; i < 6; i++) input[i] = i + 1.0;

        var output = fwt.Apply(input);
        Assert.Equal(6, output.Length);
        for (int i = 3; i < 6; i++) Assert.Equal(input[i], output[i], 12);
    }

    [Fact]
    public void Transformation_RejectsNonPositiveWidth()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FeatureWiseTransformation<double>(0, 0.3, 0.5, new Random(1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FeatureWiseTransformation<double>(-4, 0.3, 0.5, new Random(1)));
    }

    [Fact]
    public void SeededTransformationIsReproducible()
    {
        var a = Make(seed: 99);
        var b = Make(seed: 99);
        var input = Ones();
        for (int i = 0; i < 5; i++)
        {
            var x = a.Apply(input);
            var y = b.Apply(input);
            for (int c = 0; c < C; c++) Assert.Equal(x[c], y[c], 12);
        }
    }

    // ------------------------------------------------- the algorithm

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task(int seed)
    {
        var rng = new Random(seed);
        var sx = new Matrix<double>(4, 3);
        var qx = new Matrix<double>(4, 3);
        var sy = new Vector<double>(4);
        var qy = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++) { sx[i, j] = rng.NextDouble() - 0.5; qx[i, j] = rng.NextDouble() - 0.5; }
            sy[i] = i % 2; qy[i] = i % 2;
        }
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = sx, SupportSetY = sy, QuerySetX = qx, QuerySetY = qy,
            NumWays = 2, NumShots = 2, NumQueryPerClass = 2,
        };
    }

    private static LFTAlgorithm<double, Matrix<double>, Vector<double>> Algorithm(
        Action<LFTOptions<double, Matrix<double>, Vector<double>>>? tweak = null)
    {
        var options = new LFTOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
        {
            InnerLearningRate = 0.01,
            OuterLearningRate = 0.01,
            FeatureDimension = 1,   // LinearVectorModel emits one value per row
            Seed = 17,
        };
        tweak?.Invoke(options);
        return new LFTAlgorithm<double, Matrix<double>, Vector<double>>(options);
    }

    [Fact]
    public void TransformationIsInactiveOutsideTraining()
    {
        // "removing the feature-wise transformation layers from the model" — inference must be
        // deterministic and unperturbed.
        var algorithm = Algorithm();
        Assert.False(algorithm.TransformationActive);

        var task = Task(3);
        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { task }));
        Assert.False(algorithm.TransformationActive, "Transformation left active after MetaTrain.");

        algorithm.Adapt(task);
        Assert.False(algorithm.TransformationActive, "Adapt must never activate the transformation.");
    }

    [Fact]
    public void AdaptIsDeterministic_BecauseTheTransformIsRemoved()
    {
        // If the transform leaked into inference, two identical Adapt+Predict calls would differ.
        var algorithm = Algorithm();
        var task = Task(8);

        var first = algorithm.Adapt(task).Predict(task.QuerySetX);
        var second = algorithm.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i], 12);
    }

    [Fact]
    public void LearningToLearn_MovesTheHyperparameters()
    {
        var algorithm = Algorithm();
        var before = algorithm.Transformation.ScaleHyperparameters[0];

        // Needs at least two tasks so the batch can split into non-overlapping domains.
        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(
            new[] { Task(1), Task(2), Task(3), Task(4) });
        algorithm.MetaTrain(batch);

        Assert.NotEqual(before, algorithm.Transformation.ScaleHyperparameters[0]);
    }

    [Fact]
    public void HyperparametersAreFrozenWhenLearningIsDisabled()
    {
        // The paper's hand-tuned ablation: keep the transform, stop learning its spread.
        var algorithm = Algorithm(o => o.LearnTransformationHyperparameters = false);
        var before = algorithm.Transformation.ScaleHyperparameters[0];

        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(
            new[] { Task(1), Task(2), Task(3), Task(4) }));

        Assert.Equal(before, algorithm.Transformation.ScaleHyperparameters[0], 12);
    }

    [Fact]
    public void SingleTaskBatch_SkipsTheSecondStageRatherThanMeasuringTrainingLoss()
    {
        // With one task there is no held-out domain. Splitting anyway would make stage 2 score the
        // hyper-parameters on data stage 1 just trained on, which is the one thing the pseudo-unseen
        // split exists to prevent.
        var algorithm = Algorithm();
        var before = algorithm.Transformation.ScaleHyperparameters[0];

        var loss = algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { Task(5) }));

        Assert.False(double.IsNaN(loss));
        Assert.Equal(before, algorithm.Transformation.ScaleHyperparameters[0], 12);
    }

    [Fact]
    public void Constructor_RejectsInvalidConfiguration()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new LFTAlgorithm<double, Matrix<double>, Vector<double>>(null!));

        // ThrowsAny: MetaLearnerBase validates IMetaLearnerOptions before this class's own checks,
        // so the exact ArgumentException subtype is the base's choice. What matters is rejection.
        Assert.ThrowsAny<ArgumentException>(() => Algorithm(o => o.FeatureDimension = 0));
        Assert.ThrowsAny<ArgumentException>(() => Algorithm(o => o.PseudoSeenFraction = 0.0));
        Assert.ThrowsAny<ArgumentException>(() => Algorithm(o => o.PseudoSeenFraction = 1.0));
    }

    [Fact]
    public void PseudoSeenFractionMustLeaveBothDomainsNonEmpty()
    {
        // A fraction of exactly 0 or 1 empties one side, which makes the split meaningless rather
        // than merely unbalanced — so it is rejected at construction, not silently clamped.
        var ex = Assert.ThrowsAny<ArgumentException>(() => Algorithm(o => o.PseudoSeenFraction = 1.0));
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
    }
}
