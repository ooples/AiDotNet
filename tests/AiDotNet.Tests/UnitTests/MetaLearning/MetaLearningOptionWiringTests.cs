using System;
using AiDotNet.Data.Structures;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for meta-learning options that used to be accepted but never read: each one is now wired into
/// its algorithm the way its paper defines it.
/// </summary>
public class MetaLearningOptionWiringTests
{
    private const double Tolerance = 1e-10;

    /// <summary>A linear model that counts its forward passes.</summary>
    private sealed class CountingModel : LinearVectorModel
    {
        public CountingModel() : base(3)
        {
        }

        public int Predictions { get; private set; }

        public override Vector<double> Predict(Matrix<double> input)
        {
            Predictions++;
            return base.Predict(input);
        }
    }

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateRegressionTask(int seed)
    {
        const int rows = 4;
        const int featureCount = 3;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var supportX = new Matrix<double>(rows, featureCount);
        var supportY = new Vector<double>(rows);
        var queryX = new Matrix<double>(rows, featureCount);
        var queryY = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                supportX[i, j] = random.NextDouble() - 0.5;
                queryX[i, j] = random.NextDouble() - 0.5;
            }

            supportY[i] = random.NextDouble() * 2.0 - 1.0;
            queryY[i] = random.NextDouble() * 2.0 - 1.0;
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY,
            NumWays = 2,
            NumShots = 2,
            NumQueryPerClass = 2,
        };
    }

    private static TaskBatch<double, Matrix<double>, Vector<double>> Batch()
        => new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { CreateRegressionTask(3), CreateRegressionTask(5) });

    [Fact]
    public void NP_GaussianKL_MatchesTheClosedForm()
    {
        // KL(N(1, 1) || N(0, 4)) = 0.5 * (ln 4 + (1 + 1) / 4 - 1) per dimension (Kingma & Welling 2014, App. B).
        var mean1 = new Vector<double>(new[] { 1.0, 0.0 });
        var logvar1 = new Vector<double>(new[] { 0.0, 0.3 });
        var mean2 = new Vector<double>(new[] { 0.0, 0.0 });
        var logvar2 = new Vector<double>(new[] { Math.Log(4.0), 0.3 });

        double kl = NPAlgorithm<double, Matrix<double>, Vector<double>>.GaussianKL(mean1, logvar1, mean2, logvar2);

        Assert.Equal(0.5 * (Math.Log(4.0) + 0.5 - 1.0), kl, Tolerance);
    }

    private static int NPForwardPasses(int latentSamples)
    {
        var model = new CountingModel();
        var algorithm = new NPAlgorithm<double, Matrix<double>, Vector<double>>(
            new NPOptions<double, Matrix<double>, Vector<double>>(model)
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                NumLatentSamples = latentSamples,
                RepresentationDim = 4,
                LatentDim = 2,
                RandomSeed = 11,
            });
        algorithm.MetaTrain(Batch());
        return model.Predictions;
    }

    [Fact]
    public void NP_ReconstructionTerm_IsAMonteCarloAverageOverNumLatentSamples()
    {
        // Garnelo et al. (2018), Eq. 4: E_{q(z|C,T)}[log p(y_T | z, x_T)] is estimated with NumLatentSamples
        // posterior draws, each decoding the targets once, so the forward passes grow linearly with the count.
        int one = NPForwardPasses(1);
        int two = NPForwardPasses(2);
        int four = NPForwardPasses(4);

        Assert.True(two > one, $"NumLatentSamples is not read: {one} forward passes for 1 sample and {two} for 2.");
        Assert.Equal(3 * (two - one), four - one);
    }

    private static PACOHOptions<double, Matrix<double>, Vector<double>> PacohOptions(
        LinearVectorModel model, double delta, int posteriorSamples)
        => new PACOHOptions<double, Matrix<double>, Vector<double>>(model)
        {
            LossFunction = new MeanSquaredErrorLoss<double>(),
            AdaptationSteps = 2,
            Delta = delta,
            NumPosteriorSamples = posteriorSamples,
            RandomSeed = 17,
        };

    [Fact]
    public void PACOH_Delta_EntersTheBoundAsItsConfidenceTerm()
    {
        // Rothfuss et al. (2021), Thm. 1: the bound carries ln(2 sqrt(n) / delta) next to the KL term, weighted by
        // KLCoefficient / n. Two runs that differ only in delta differ by exactly that weight times ln(d2 / d1).
        var tight = new PACOHAlgorithm<double, Matrix<double>, Vector<double>>(PacohOptions(new LinearVectorModel(3), 0.05, 3));
        var loose = new PACOHAlgorithm<double, Matrix<double>, Vector<double>>(PacohOptions(new LinearVectorModel(3), 0.5, 3));

        double tightLoss = tight.MetaTrain(Batch());
        double looseLoss = loose.MetaTrain(Batch());

        Assert.Equal(0.1 / 2 * Math.Log(0.5 / 0.05), tightLoss - looseLoss, 1e-9);
    }

    [Fact]
    public void PACOH_NumPosteriorSamples_DrawsThatManyPosteriorParametersPerTask()
    {
        // The Gibbs risk E_{theta ~ Q}[L(theta; D^q)] is a Monte Carlo average over NumPosteriorSamples draws,
        // each evaluated on the query set once, so the forward passes grow linearly with the count.
        int Passes(int samples)
        {
            var model = new CountingModel();
            new PACOHAlgorithm<double, Matrix<double>, Vector<double>>(PacohOptions(model, 0.05, samples)).MetaTrain(Batch());
            return model.Predictions;
        }

        int one = Passes(1);
        int two = Passes(2);
        int four = Passes(4);

        Assert.True(two > one, $"NumPosteriorSamples is not read: {one} forward passes for 1 sample and {two} for 2.");
        Assert.Equal(3 * (two - one), four - one);
    }

    [Fact]
    public void PACOH_RejectsADeltaOutsideTheUnitInterval()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new PACOHAlgorithm<double, Matrix<double>, Vector<double>>(PacohOptions(new LinearVectorModel(3), 1.5, 1)));
    }

    [Fact]
    public void MetaPACOH_Delta_EntersTheBoundAsItsConfidenceTerm()
    {
        // Same confidence term as PACOH, added to the hierarchical KL: KLCoefficient / n * ln(2 sqrt(n) / delta).
        MetaPACOHOptions<double, Matrix<double>, Vector<double>> Options(double delta)
            => new MetaPACOHOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                AdaptationSteps = 2,
                Delta = delta,
                RandomSeed = 17,
            };

        double tightLoss = new MetaPACOHAlgorithm<double, Matrix<double>, Vector<double>>(Options(0.05)).MetaTrain(Batch());
        double looseLoss = new MetaPACOHAlgorithm<double, Matrix<double>, Vector<double>>(Options(0.5)).MetaTrain(Batch());

        Assert.Equal(0.1 / 2 * Math.Log(0.5 / 0.05), tightLoss - looseLoss, 1e-9);
    }

    private static MetaSGDAlgorithm<double, Matrix<double>, Vector<double>> MetaSgd(
        bool schedule, MetaSGDLearningRateScheduleType type = MetaSGDLearningRateScheduleType.CosineAnnealing)
        => new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(
            new MetaSGDOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                OuterLearningRate = 0.1,
                UseLearningRateSchedule = schedule,
                LearningRateSchedule = type,
                ScheduleWarmupEpisodes = 4,
                NumMetaIterations = 12,
            });

    [Theory]
    [InlineData(MetaSGDLearningRateScheduleType.Constant, 0.1)]
    [InlineData(MetaSGDLearningRateScheduleType.StepDecay, 0.025)]
    [InlineData(MetaSGDLearningRateScheduleType.Exponential, 0.01)]
    [InlineData(MetaSGDLearningRateScheduleType.CosineAnnealing, 0.05)]
    [InlineData(MetaSGDLearningRateScheduleType.Cyclical, 0.1)]
    public void MetaSGD_LearningRateSchedule_ShapesTheMetaStepAfterWarmup(
        MetaSGDLearningRateScheduleType type, double expected)
    {
        // Warmup ramps linearly over 4 iterations (Goyal et al. 2017); the schedule then runs over the remaining 8.
        var learner = MetaSgd(schedule: true, type);
        Assert.Equal(0.1 * 1 / 4, learner.ScheduledMetaLearningRate(0), Tolerance);
        Assert.Equal(0.1 * 4 / 4, learner.ScheduledMetaLearningRate(3), Tolerance);

        // Halfway (progress 1/2): step decay has halved twice, the exponential is at 0.01^(1/2) = 0.1 of the rate,
        // cosine at (1 + cos(pi / 2)) / 2. The cyclical policy (four cycles) peaks at progress 1/8.
        int iteration = type == MetaSGDLearningRateScheduleType.Cyclical ? 4 + 1 : 4 + 4;
        Assert.Equal(expected, learner.ScheduledMetaLearningRate(iteration), Tolerance);
    }

    [Fact]
    public void MetaSGD_LearningRateSchedule_ScalesTheThetaStep()
    {
        // Li et al. (2017) keep the meta step fixed (the default). With the schedule on, the first iteration of a
        // four-iteration warmup steps theta by a quarter of the fixed step along the same meta-gradient.
        var fixedRate = MetaSgd(schedule: false);
        var warmup = MetaSgd(schedule: true);
        var start = fixedRate.GetMetaModel().GetParameters();

        fixedRate.MetaTrain(Batch());
        warmup.MetaTrain(Batch());

        var fixedStep = fixedRate.GetMetaModel().GetParameters();
        var warmupStep = warmup.GetMetaModel().GetParameters();
        for (int i = 0; i < start.Length; i++)
        {
            Assert.Equal(0.25 * (fixedStep[i] - start[i]), warmupStep[i] - start[i], 1e-12);
        }
    }

    [Fact]
    public void MetaSGD_Defaults_DrawAlphaFromThePaperRange()
    {
        // Li et al. (2017), Sec. 4: alpha is initialised randomly in [0.005, 0.1]; meta step 0.001; one inner step.
        var options = new MetaSGDOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3)) { RandomSeed = 5 };
        var learner = new MetaSGDAlgorithm<double, Matrix<double>, Vector<double>>(options);
        var alpha = learner.LearnedOptimizer.LearningRates;

        Assert.Equal(0.001, options.OuterLearningRate);
        Assert.Equal(1, options.InnerSteps);
        for (int i = 0; i < alpha.Length; i++)
        {
            Assert.InRange(alpha[i], 0.005, 0.1);
        }

        Assert.True(alpha[0] != alpha[1] || alpha[1] != alpha[2], "alpha must be drawn per parameter, not shared.");
    }
}
