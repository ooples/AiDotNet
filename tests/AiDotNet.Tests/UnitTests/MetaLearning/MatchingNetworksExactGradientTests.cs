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
/// Matching Networks as Vinyals et al. 2016 state them - cosine attention over the support set, trained on
/// -log P(y | x, S) - with full context embeddings (appendix A) and the learned kernel on the same exact gradient
/// (#2155).
/// </summary>
public class MatchingNetworksExactGradientTests
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

        var support = Rows(2);
        var query = Rows(2);
        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support.X, SupportSetY = support.Y, QuerySetX = query.X, QuerySetY = query.Y,
            NumWays = Classes, NumShots = 2, NumQueryPerClass = 2, Name = $"matching-{seed}",
        };
    }

    private static MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            NumClasses = Classes,
            OuterLearningRate = 0.05,
            RandomSeed = 11,
        };
        configure?.Invoke(options);
        return new MatchingNetworksAlgorithm<double, Matrix<double>, Tensor<double>>(options);
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

    private static void CheckVector(Func<double> loss, double[] analytic, Func<double[]> get, Action<double[]> set, string name)
    {
        for (int i = 0; i < analytic.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(loss, h =>
            {
                var shifted = get();
                shifted[index] += h;
                set(shifted);
            }, 1e-6);
            Assert.True(Math.Abs(analytic[i] - numeric) <= 1e-6 + 1e-4 * Math.Abs(numeric),
                $"dL/d{name}[{i}]: analytic {analytic[i]:G10} vs central difference {numeric:G10}.");
        }
    }

    [Theory]
    [InlineData(MatchingNetworksAttentionFunction.Cosine, false, false)]
    [InlineData(MatchingNetworksAttentionFunction.DotProduct, false, false)]
    [InlineData(MatchingNetworksAttentionFunction.Euclidean, false, false)]
    [InlineData(MatchingNetworksAttentionFunction.Learned, false, false)]
    [InlineData(MatchingNetworksAttentionFunction.Cosine, true, false)]
    [InlineData(MatchingNetworksAttentionFunction.Cosine, false, true)]
    [InlineData(MatchingNetworksAttentionFunction.Learned, true, true)]
    public void EpisodeGradient_MatchesCentralDifferences(
        MatchingNetworksAttentionFunction kernel, bool bidirectional, bool fullContext)
    {
        var learner = CreateLearner(o =>
        {
            o.AttentionFunction = kernel;
            o.UseBidirectionalEncoding = bidirectional;
            o.UseFullContextEmbedding = fullContext;
            o.ProcessingSteps = 3;
            o.Temperature = 0.7;
        });
        var task = CreateTask(seed: 3);

        // Move a learned kernel off the identity, so its gradient is not checked only at the symmetric start.
        var first = learner.EpisodeGradientForTesting(task);
        if (first.Kernel.Length > 0)
        {
            learner.KernelWeightsForTesting = new Vector<double>(
                Enumerable.Range(0, first.Kernel.Length).Select(i => (i % (Width + 1) == 0 ? 1.0 : 0.0) + 0.1 * Math.Sin(i + 1)).ToArray());
        }

        var (_, body, kernelGradient, support, query) = learner.EpisodeGradientForTesting(task);
        double Loss() => learner.EpisodeLossForTesting(task);

        var model = learner.GetMetaModel();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(), v => model.SetParameters(new Vector<double>(v)), "body");
        CheckVector(Loss, kernelGradient.ToArray(), () => learner.KernelWeightsForTesting.ToArray(),
            v => learner.KernelWeightsForTesting = new Vector<double>(v), "W");
        CheckVector(Loss, support.ToArray(), () => learner.SupportContextWeightsForTesting.ToArray(),
            v => learner.SupportContextWeightsForTesting = new Vector<double>(v), "g-LSTM");
        CheckVector(Loss, query.ToArray(), () => learner.QueryContextWeightsForTesting.ToArray(),
            v => learner.QueryContextWeightsForTesting = new Vector<double>(v), "attLSTM");

        Assert.Equal(kernel == MatchingNetworksAttentionFunction.Learned ? Width * Width : 0, kernelGradient.Length);
        Assert.Equal(bidirectional || fullContext, support.Length > 0);
        Assert.Equal(fullContext, query.Length > 0);
    }

    [Fact]
    public void Loss_IsMinusLogOfTheAttentionWeightedLabels()
    {
        // Eq. 1 and 2 by hand: P(y | x, S) = sum_i softmax_i(cos(f(x), g(x_i))) y_i and the loss is -log P(y | x, S).
        var learner = CreateLearner();
        var task = CreateTask(seed: 4);
        var body = learner.GetMetaModel();
        var support = body.Predict(task.SupportSetX);
        var query = body.Predict(task.QuerySetX);

        double Cosine(int q, int s)
        {
            double dot = 0, qq = 0, ss = 0;
            for (int f = 0; f < Width; f++)
            {
                double a = query[q * Width + f], b = support[s * Width + f];
                dot += a * b;
                qq += a * a;
                ss += b * b;
            }

            return dot / Math.Sqrt(qq * ss);
        }

        int supportRows = task.SupportSetX.Rows, queryRows = task.QuerySetX.Rows;
        double expected = 0;
        for (int q = 0; q < queryRows; q++)
        {
            var e = Enumerable.Range(0, supportRows).Select(s => Math.Exp(Cosine(q, s))).ToArray();
            double p = Enumerable.Range(0, supportRows).Where(s => task.SupportSetY[s] == task.QuerySetY[q]).Sum(s => e[s]) / e.Sum();
            expected -= Math.Log(p);
        }

        Assert.Equal(expected / queryRows, learner.EpisodeLossForTesting(task), 10);
    }

    [Fact]
    public void LearnedKernelAtIdentity_IsThePapersCosine()
    {
        var task = CreateTask(seed: 5);
        var cosine = CreateLearner();
        var learned = CreateLearner(o => o.AttentionFunction = MatchingNetworksAttentionFunction.Learned);

        Assert.Equal(cosine.EpisodeLossForTesting(task), learned.EpisodeLossForTesting(task), 12);
    }

    [Fact]
    public void ProcessingSteps_DefaultsToFive()
    {
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width));

        Assert.Equal(5, options.ProcessingSteps);
    }

    [Fact]
    public void ProcessingSteps_BelowOne_IsInvalid()
    {
        Assert.Throws<ArgumentException>(() => CreateLearner(o => o.ProcessingSteps = 0));
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
    public void FullContextEmbeddings_AreTrained()
    {
        var learner = CreateLearner(o =>
        {
            o.UseFullContextEmbedding = true;
            o.AttentionFunction = MatchingNetworksAttentionFunction.Learned;
        });
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(8), CreateTask(9) });
        learner.MetaTrain(batch);
        var kernel = learner.KernelWeightsForTesting.ToArray();
        var support = learner.SupportContextWeightsForTesting.ToArray();
        var query = learner.QueryContextWeightsForTesting.ToArray();

        learner.MetaTrain(batch);

        Assert.NotEqual(kernel, learner.KernelWeightsForTesting.ToArray());
        Assert.NotEqual(support, learner.SupportContextWeightsForTesting.ToArray());
        Assert.NotEqual(query, learner.QueryContextWeightsForTesting.ToArray());
    }

    [Fact]
    public void Adapt_GivesAClassWithoutSupportZeroProbability()
    {
        var learner = CreateLearner(o => o.NumClasses = 3);
        var task = CreateTask(seed: 6);

        var probabilities = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { task.QuerySetX.Rows, 3 }, probabilities.Shape.ToArray());
        for (int r = 0; r < task.QuerySetX.Rows; r++)
        {
            Assert.Equal(0.0, probabilities[r * 3 + 2]);
            Assert.Equal(1.0, probabilities[r * 3] + probabilities[r * 3 + 1], 10);
        }
    }

    [Fact]
    public void ModelBuiltDirectly_RefusesUntrainedContextEmbeddings()
    {
        var body = new LinearEmbeddingModel(Features, Width);
        var options = new MatchingNetworksOptions<double, Matrix<double>, Tensor<double>>(body)
        {
            NumClasses = Classes,
            UseFullContextEmbedding = true,
        };
        var task = CreateTask(seed: 7);

        Assert.Throws<ArgumentException>(() => new MatchingNetworksModel<double, Matrix<double>, Tensor<double>>(
            body, task.SupportSetX, task.SupportSetY, options, MathHelper.GetNumericOperations<double>()));
    }
}
