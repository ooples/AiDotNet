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
/// ANIL as Raghu et al. 2020 state it: a per-example head adapted alone in the inner loop, and body and head
/// initialisation meta-learned on the exact meta-gradient, second-order terms included (#2155).
/// </summary>
public class ANILExactGradientTests
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
            NumWays = Classes, NumShots = 3, NumQueryPerClass = 2, Name = $"anil-{seed}",
        };
    }

    private static ANILAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<ANILOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new ANILOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            NumClasses = Classes,
            FeatureDimension = Width,
            AdaptationSteps = 2,
            InnerLearningRate = 0.5,
            OuterLearningRate = 0.05,
            UseHeadBias = true,
        };
        configure?.Invoke(options);
        return new ANILAlgorithm<double, Matrix<double>, Tensor<double>>(options);
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
    [InlineData(false, 0.1)]
    [InlineData(true, 0.0)]
    public void TaskMetaGradient_MatchesCentralDifferences(bool firstOrder, double headL2)
    {
        var learner = CreateLearner(o =>
        {
            o.UseFirstOrder = firstOrder;
            o.HeadL2Regularization = headL2;
        });
        var task = CreateTask(seed: 3);
        var (_, body, weights, bias) = learner.TaskMetaGradientForTesting(task);
        if (firstOrder)
        {
            // First-order drops the second-order terms, so it is compared with the exact gradient rather than with
            // the loss: the two must differ, and only the exact one matches the central difference (checked below).
            var exact = CreateLearner(o => o.HeadL2Regularization = headL2).TaskMetaGradientForTesting(task);
            Assert.True(exact.Body.ToArray().Zip(body.ToArray(), (a, b) => Math.Abs(a - b)).Max() > 1e-9,
                "Dropping the second-order terms changed nothing.");
            return;
        }

        double Loss() => learner.TaskLossForTesting(task);
        var model = learner.GetMetaModel();
        var parameters = model.GetParameters();
        Assert.Equal(parameters.Length, body.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(Loss, h =>
            {
                var shifted = model.GetParameters();
                shifted[index] += h;
                model.SetParameters(shifted);
            }, 1e-6);
            AssertClose(body[i], numeric, $"dL/dbody[{i}]");
        }

        for (int i = 0; i < weights.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(Loss, h =>
            {
                var shifted = learner.HeadWeightsForTesting;
                shifted[index] += h;
                learner.HeadWeightsForTesting = shifted;
            }, 1e-6);
            AssertClose(weights[i], numeric, $"dL/dW[{i}]");
        }

        for (int i = 0; i < bias.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(Loss, h =>
            {
                var shifted = learner.HeadBiasForTesting;
                shifted[index] += h;
                learner.HeadBiasForTesting = shifted;
            }, 1e-6);
            AssertClose(bias[i], numeric, $"dL/db[{i}]");
        }
    }

    [Fact]
    public void Adapt_ScoresEveryClassForEachExample()
    {
        var learner = CreateLearner();
        var task = CreateTask(seed: 5);

        var scores = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { task.QuerySetX.Rows, Classes }, scores.Shape.ToArray());
    }

    [Fact]
    public void Defaults_KeepThePapersSecondOrderTerms()
    {
        var options = new ANILOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width));

        Assert.False(options.UseFirstOrder);
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
    public void RejectsABodyWhoseWidthIsNotFeatureDimension()
    {
        var learner = CreateLearner(o => o.FeatureDimension = Width + 1);

        Assert.Throws<InvalidOperationException>(() => learner.MetaTrain(
            new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(7) })));
    }
}
