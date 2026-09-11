using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// BOIL as Oh et al. 2021 state it: the body adapted alone in the inner loop behind a frozen head, and body and head
/// meta-learned on the exact meta-gradient - with MAML++'s per-layer per-step rates when asked (#2155).
/// </summary>
public class BOILExactGradientTests
{
    private const int Features = 3;
    private const int Width = 3;
    private const int Classes = 2;

    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTask(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var centres = Enumerable.Range(0, Classes)
            .Select(_ => Enumerable.Range(0, Features).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray()).ToArray();

        (Matrix<double> X, Tensor<double> Y) Rows(int perClass)
        {
            int rows = perClass * Classes;
            var x = new Matrix<double>(rows, Features);
            var y = new Tensor<double>(new[] { rows });
            for (int r = 0; r < rows; r++)
            {
                int label = r % Classes;
                for (int f = 0; f < Features; f++) x[r, f] = centres[label][f] + 0.3 * (rng.NextDouble() - 0.5);
                y[r] = label;
            }

            return (x, y);
        }

        var support = Rows(3);
        var query = Rows(2);
        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support.X, SupportSetY = support.Y, QuerySetX = query.X, QuerySetY = query.Y,
            NumWays = Classes, NumShots = 3, NumQueryPerClass = 2, Name = $"boil-{seed}",
        };
    }

    private static BOILAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<BOILOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new BOILOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            NumClasses = Classes,
            FeatureDimension = Width,
            AdaptationSteps = 2,
            InnerLearningRate = 0.3,
            OuterLearningRate = 0.05,
        };
        configure?.Invoke(options);
        return new BOILAlgorithm<double, Matrix<double>, Tensor<double>>(options);
    }

    private static double CentralDifference(Func<double> loss, Action<double> shift, double h)
    {
        shift(h);
        double plus = loss();
        shift(-2 * h);
        double minus = loss();
        shift(h);
        return (plus - minus) / (2 * h);
    }

    private static void AssertClose(double analytic, double numeric, string what)
        => Assert.True(Math.Abs(analytic - numeric) <= 1e-6 + 1e-4 * Math.Abs(numeric),
            $"{what}: analytic {analytic:G10} vs central difference {numeric:G10}.");

    [Theory]
    [InlineData(false, 0.0)]
    [InlineData(true, 0.0)]
    [InlineData(true, 0.1)]
    public void TaskMetaGradient_MatchesCentralDifferences(bool layerwiseRates, double bodyL2)
    {
        var learner = CreateLearner(o =>
        {
            o.UseLayerwiseLearningRates = layerwiseRates;
            o.BodyL2Regularization = bodyL2;
        });
        var task = CreateTask(seed: 3);
        var (_, body, weights, bias, rates) = learner.TaskMetaGradientForTesting(task);
        double Loss() => learner.TaskLossForTesting(task);

        var model = learner.GetMetaModel();
        var parameters = model.GetParameters();
        Assert.Equal(parameters.Length, body.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            int index = i;
            AssertClose(body[i], CentralDifference(Loss, h =>
            {
                var shifted = model.GetParameters();
                shifted[index] += h;
                model.SetParameters(shifted);
            }, 1e-6), $"dL/dbody[{i}]");
        }

        for (int i = 0; i < weights.Length; i++)
        {
            int index = i;
            AssertClose(weights[i], CentralDifference(Loss, h =>
            {
                var shifted = learner.HeadWeightsForTesting;
                shifted[index] += h;
                learner.HeadWeightsForTesting = shifted;
            }, 1e-6), $"dL/dW[{i}]");
        }

        for (int i = 0; i < bias.Length; i++)
        {
            int index = i;
            AssertClose(bias[i], CentralDifference(Loss, h =>
            {
                var shifted = learner.HeadBiasForTesting;
                shifted[index] += h;
                learner.HeadBiasForTesting = shifted;
            }, 1e-6), $"dL/db[{i}]");
        }

        Assert.Equal(layerwiseRates ? 2 : 0, rates.Length);
        for (int i = 0; i < rates.Length; i++)
        {
            int index = i;
            AssertClose(rates[i], CentralDifference(Loss, h =>
            {
                var shifted = learner.LayerStepRatesForTesting;
                shifted[index] += h;
                learner.LayerStepRatesForTesting = shifted;
            }, 1e-6), $"dL/drate[{i}]");
        }
    }

    [Fact]
    public void FirstOrder_DropsTheSecondOrderTerms()
    {
        var task = CreateTask(seed: 4);
        var exact = CreateLearner().TaskMetaGradientForTesting(task);
        var firstOrder = CreateLearner(o => o.UseFirstOrder = true).TaskMetaGradientForTesting(task);

        Assert.True(exact.Body.ToArray().Zip(firstOrder.Body.ToArray(), (a, b) => Math.Abs(a - b)).Max() > 1e-9,
            "Dropping the second-order terms changed nothing.");
    }

    [Fact]
    public void Adapt_ScoresEveryClassForEachExample_AndLeavesTheMetaModelAlone()
    {
        var learner = CreateLearner();
        var task = CreateTask(seed: 5);
        var before = learner.GetMetaModel().GetParameters().ToArray();

        var adapted = learner.Adapt(task);
        var scores = adapted.Predict(task.QuerySetX);
        adapted.Predict(task.QuerySetX);

        Assert.Equal(new[] { task.QuerySetX.Rows, Classes }, scores.Shape.ToArray());
        Assert.Equal(before, learner.GetMetaModel().GetParameters().ToArray());
    }

    [Fact]
    public void Defaults_KeepSecondOrderTerms_AndNoLearnedRates()
    {
        var options = new BOILOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width));

        Assert.False(options.UseFirstOrder);
        Assert.False(options.UseLayerwiseLearningRates);
        Assert.Equal(1.0, options.BodyAdaptationFraction);
    }

    [Fact]
    public void MetaTrain_LowersTheQueryLossOnSeparableTasks()
    {
        var learner = CreateLearner();
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(
            Enumerable.Range(0, 4).Select(i => CreateTask(100 + i)).ToArray());

        double first = learner.MetaTrain(batch);
        double last = first;
        for (int step = 0; step < 30; step++) last = learner.MetaTrain(batch);

        Assert.True(last < first, $"Meta-training on a fixed batch went from {first} to {last}.");
    }

    [Fact]
    public void LayerwiseRates_AreMetaLearned()
    {
        var learner = CreateLearner(o => o.UseLayerwiseLearningRates = true);
        var before = learner.LayerStepRatesForTesting.ToArray();

        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(8), CreateTask(9) }));

        Assert.NotEqual(before, learner.LayerStepRatesForTesting.ToArray());
    }

    [Fact]
    public void RejectsABodyWhoseWidthIsNotFeatureDimension()
    {
        var learner = CreateLearner(o => o.FeatureDimension = Width + 1);

        Assert.Throws<InvalidOperationException>(() => learner.MetaTrain(
            new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(7) })));
    }
}
