using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// ICM-Fusion's meta-gradient is computed in closed form, so it is checked against central differences of the
/// meta-loss it claims to differentiate (Shao et al. 2025, Algorithm 1, Eq. 12).
/// </summary>
public class ICMFusionGradientTests
{
    private const double Step = 1e-5;

    private static MetaLearningTask<double, Matrix<double>, Vector<double>> Task(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        Matrix<double> Features()
        {
            var m = new Matrix<double>(4, 3);
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) m[i, j] = rng.NextDouble() - 0.5;
            return m;
        }

        var labels = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = Features(), SupportSetY = labels, QuerySetX = Features(), QuerySetY = labels,
            NumWays = 2, NumShots = 2, NumQueryPerClass = 2, Name = $"icm-{seed}",
        };
    }

    private static ICMFusionAlgorithm<double, Matrix<double>, Vector<double>> TrainedLearner(bool firstOrder)
    {
        var learner = new ICMFusionAlgorithm<double, Matrix<double>, Vector<double>>(
            new ICMFusionOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                LatentDim = 4,
                AdaptationSteps = 3,
                InnerLearningRate = 0.1,
                OuterLearningRate = 0.05,
                UseFirstOrder = firstOrder,
                RandomSeed = 11,
            });

        // The decoder starts at zero, which would leave every path through it untested.
        for (int i = 0; i < 3; i++)
            learner.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { Task(20 + i), Task(40 + i) }));
        Assert.Contains(learner.DecoderParametersForTesting, value => value != 0.0);
        return learner;
    }

    private static double[] Noise(int seed, int length)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        return Enumerable.Range(0, length).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray();
    }

    private static void AssertMatchesCentralDifferences(
        ICMFusionAlgorithm<double, Matrix<double>, Vector<double>> learner,
        MetaLearningTask<double, Matrix<double>, Vector<double>> task,
        double[] firstNoise, double[] secondNoise,
        double[] encoderGradient, double[] decoderGradient, double[]? frozenAdapted)
    {
        var encoder = learner.EncoderParametersForTesting;
        var decoder = learner.DecoderParametersForTesting;

        void Check(double[] parameters, double[] analytic, string name)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                double saved = parameters[i];
                parameters[i] = saved + Step;
                double plus = learner.MetaObjectiveForTesting(task, encoder, decoder, firstNoise, secondNoise, frozenAdapted);
                parameters[i] = saved - Step;
                double minus = learner.MetaObjectiveForTesting(task, encoder, decoder, firstNoise, secondNoise, frozenAdapted);
                parameters[i] = saved;

                double numeric = (plus - minus) / (2 * Step);
                double tolerance = 1e-6 + 1e-4 * Math.Max(Math.Abs(numeric), Math.Abs(analytic[i]));
                Assert.True(Math.Abs(numeric - analytic[i]) <= tolerance,
                    $"{name}[{i}]: analytic {analytic[i]:G10}, central difference {numeric:G10}.");
            }
        }

        Check(encoder, encoderGradient, "encoder");
        Check(decoder, decoderGradient, "decoder");
    }

    [Fact]
    public void SecondOrderGradient_MatchesCentralDifferencesOfTheMetaLoss()
    {
        var learner = TrainedLearner(firstOrder: false);
        var task = Task(7);
        var firstNoise = Noise(1, 4);
        var secondNoise = Noise(2, 4);

        var (encoderGradient, decoderGradient, _) = learner.MetaGradientForTesting(task, firstNoise, secondNoise, secondOrder: true);

        AssertMatchesCentralDifferences(learner, task, firstNoise, secondNoise, encoderGradient, decoderGradient, frozenAdapted: null);
    }

    [Fact]
    public void FirstOrderGradient_IsTheGradientWithTheRefinedAdapterHeldFixed()
    {
        var learner = TrainedLearner(firstOrder: true);
        var task = Task(8);
        var firstNoise = Noise(3, 4);
        var secondNoise = Noise(4, 4);

        var (encoderGradient, decoderGradient, adapted) = learner.MetaGradientForTesting(task, firstNoise, secondNoise, secondOrder: false);

        AssertMatchesCentralDifferences(learner, task, firstNoise, secondNoise, encoderGradient, decoderGradient, frozenAdapted: adapted);
    }

    [Fact]
    public void SecondOrderGradient_DiffersFromFirstOrder_WhenRefinementDependsOnTheVae()
    {
        // A control arm: if the refinement path contributed nothing, the second-order test above could not tell a
        // dropped term from a correct one.
        var learner = TrainedLearner(firstOrder: false);
        var task = Task(9);
        var firstNoise = Noise(5, 4);
        var secondNoise = Noise(6, 4);

        var full = learner.MetaGradientForTesting(task, firstNoise, secondNoise, secondOrder: true);
        var truncated = learner.MetaGradientForTesting(task, firstNoise, secondNoise, secondOrder: false);

        double difference = full.Decoder.Zip(truncated.Decoder, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(difference > 1e-6, $"The refinement path changed the decoder gradient by only {difference:G6}.");
    }

    [Fact]
    public void MetaModel_StartsAtThePretrainedWeights_AndMovesToThePretrainedPlusTheFusedAdapter()
    {
        var model = new LinearVectorModel(3);
        var pretrained = model.GetParameters().Clone();
        var learner = new ICMFusionAlgorithm<double, Matrix<double>, Vector<double>>(
            new ICMFusionOptions<double, Matrix<double>, Vector<double>>(model)
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                LatentDim = 4,
                RandomSeed = 5,
            });

        Assert.Equal(pretrained.ToArray(), learner.GetMetaModel().GetParameters().ToArray());

        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Vector<double>>(new[] { Task(1), Task(2) }));

        Assert.NotEqual(pretrained.ToArray(), learner.GetMetaModel().GetParameters().ToArray());
        Assert.Equal(pretrained.ToArray(), learner.PretrainedParametersForTesting.ToArray());
    }
}
