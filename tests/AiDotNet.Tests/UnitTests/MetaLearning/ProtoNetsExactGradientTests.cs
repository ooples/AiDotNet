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
/// Prototypical networks as Snell et al. 2017 state them - squared Euclidean distance to class-mean prototypes, trained
/// through the prototypes - and their learned extensions on the same exact gradient (#2155).
/// </summary>
public class ProtoNetsExactGradientTests
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
            NumWays = Classes, NumShots = 3, NumQueryPerClass = 2, Name = $"proto-{seed}",
        };
    }

    private static ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<ProtoNetsOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            OuterLearningRate = 0.05,
        };
        configure?.Invoke(options);
        return new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(options);
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

    private static void CheckVector(Func<double> loss, double[] analytic, Func<double[]> get, Action<double[]> set, string name)
    {
        for (int i = 0; i < analytic.Length; i++)
        {
            int index = i;
            AssertClose(analytic[i], CentralDifference(loss, h =>
            {
                var shifted = get();
                shifted[index] += h;
                set(shifted);
            }, 1e-6), $"dL/d{name}[{i}]");
        }
    }

    [Theory]
    [InlineData(ProtoNetsDistanceFunction.Euclidean, false, false, false)]
    [InlineData(ProtoNetsDistanceFunction.Cosine, false, false, true)]
    [InlineData(ProtoNetsDistanceFunction.Mahalanobis, true, true, false)]
    [InlineData(ProtoNetsDistanceFunction.Euclidean, true, true, true)]
    public void EpisodeGradient_MatchesCentralDifferences(
        ProtoNetsDistanceFunction distance, bool attention, bool classScaling, bool normalize)
    {
        var learner = CreateLearner(o =>
        {
            o.DistanceFunction = distance;
            o.UseAttentionMechanism = attention;
            o.UseAdaptiveClassScaling = classScaling;
            o.NormalizeFeatures = normalize;
            o.MahalanobisScaling = 1.5;
        });
        var task = CreateTask(seed: 3);

        // Move the learned metric off its initial point, so its gradient is not checked only at a symmetric start.
        var first = learner.EpisodeGradientForTesting(task);
        if (first.Attention.Length > 0) learner.AttentionQueryForTesting = new Vector<double>(Enumerable.Range(0, first.Attention.Length).Select(i => 0.3 * (i + 1)).ToArray());
        if (first.Mahalanobis.Length > 0) learner.MahalanobisLogScaleForTesting = new Vector<double>(Enumerable.Range(0, first.Mahalanobis.Length).Select(i => 0.1 * (i - 1)).ToArray());
        if (first.Classes.Length > 0) learner.ClassLogScaleForTesting = new Vector<double>(Enumerable.Range(0, first.Classes.Length).Select(i => 0.2 * (i + 1)).ToArray());

        var (_, body, attn, maha, classes) = learner.EpisodeGradientForTesting(task);
        double Loss() => learner.EpisodeLossForTesting(task);

        var model = learner.GetMetaModel();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(), v => model.SetParameters(new Vector<double>(v)), "body");
        CheckVector(Loss, attn.ToArray(), () => learner.AttentionQueryForTesting.ToArray(), v => learner.AttentionQueryForTesting = new Vector<double>(v), "attention");
        CheckVector(Loss, maha.ToArray(), () => learner.MahalanobisLogScaleForTesting.ToArray(), v => learner.MahalanobisLogScaleForTesting = new Vector<double>(v), "rho");
        CheckVector(Loss, classes.ToArray(), () => learner.ClassLogScaleForTesting.ToArray(), v => learner.ClassLogScaleForTesting = new Vector<double>(v), "kappa");
        Assert.Equal(attention ? Width : 0, attn.Length);
        Assert.Equal(distance == ProtoNetsDistanceFunction.Mahalanobis ? Width : 0, maha.Length);
        Assert.Equal(classScaling ? Classes : 0, classes.Length);
    }

    [Fact]
    public void Euclidean_IsTheSquaredDistance()
    {
        // One support example per class and one query: the prototypes ARE the support embeddings, so the softmax over
        // -d is checkable by hand from the embedding model's own outputs.
        var body = new LinearEmbeddingModel(Features, Width);
        var options = new ProtoNetsOptions<double, Matrix<double>, Tensor<double>>(body);
        var learner = new ProtoNetsAlgorithm<double, Matrix<double>, Tensor<double>>(options);
        var support = new Matrix<double>(new[,] { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 } });
        var query = new Matrix<double>(new[,] { { 0.5, 0.2, 0.1 } });
        var task = new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support, SupportSetY = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.0, 1.0 })),
            QuerySetX = query, QuerySetY = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 0.0 })),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 1, Name = "hand",
        };

        var p = body.Predict(support);
        var q = body.Predict(query);
        double D(int c) => Enumerable.Range(0, Width).Sum(f => Math.Pow(q[f] - p[c * Width + f], 2));
        double expected = -Math.Log(Math.Exp(-D(0)) / (Math.Exp(-D(0)) + Math.Exp(-D(1))));

        Assert.Equal(expected, learner.EpisodeLossForTesting(task), 10);
    }

    [Fact]
    public void AttentionAtZero_IsThePapersMean()
    {
        var task = CreateTask(seed: 4);
        var plain = CreateLearner();
        var attentive = CreateLearner(o => o.UseAttentionMechanism = true);

        Assert.Equal(plain.EpisodeLossForTesting(task), attentive.EpisodeLossForTesting(task), 12);
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
    public void LearnedMetric_IsTrained()
    {
        var learner = CreateLearner(o =>
        {
            o.UseAttentionMechanism = true;
            o.UseAdaptiveClassScaling = true;
            o.DistanceFunction = ProtoNetsDistanceFunction.Mahalanobis;
        });
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(8), CreateTask(9) });

        learner.MetaTrain(batch);
        learner.MetaTrain(batch);

        Assert.Contains(learner.AttentionQueryForTesting.ToArray(), v => v != 0.0);
        Assert.Contains(learner.MahalanobisLogScaleForTesting.ToArray(), v => v != 0.0);
        Assert.Contains(learner.ClassLogScaleForTesting.ToArray(), v => v != 0.0);
    }

    [Fact]
    public void Adapt_ReturnsOneProbabilityRowPerExample()
    {
        var learner = CreateLearner();
        var task = CreateTask(seed: 5);

        var probabilities = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { task.QuerySetX.Rows, Classes }, probabilities.Shape.ToArray());
        for (int r = 0; r < task.QuerySetX.Rows; r++)
        {
            double sum = Enumerable.Range(0, Classes).Sum(c => probabilities[r * Classes + c]);
            Assert.Equal(1.0, sum, 10);
        }
    }

    [Fact]
    public void RejectsAQueryClassWithoutSupport()
    {
        var learner = CreateLearner();
        var task = CreateTask(seed: 6);
        task.QuerySetY[0] = 5;

        Assert.Throws<ArgumentException>(() => learner.MetaTrain(
            new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task })));
    }
}
