using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// MCL as its documented algorithm states it - the prototypical episodic loss plus the supervised contrastive loss
/// of Khosla et al. 2020 over a real projection head - differentiated exactly rather than by perturbation (#2155).
/// </summary>
public class MCLExactGradientTests
{
    private const int Features = 3;
    private const int Width = 3;
    private const int Classes = 2;

    /// <summary>A small head: the gradient check finite-differences every one of its weights.</summary>
    private const int Projection = 3;

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
            NumWays = Classes, NumShots = 2, NumQueryPerClass = 2, Name = $"mcl-{seed}",
        };
    }

    private static MCLAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<MCLOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new MCLOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            NumWays = Classes,
            ProjectionDim = Projection,
            ContrastiveWeight = 0.5,
            ContrastiveTemperature = 0.07,
            OuterLearningRate = 0.05,
            GradientClipThreshold = null,
            RandomSeed = 7,
        };
        configure?.Invoke(options);
        return new MCLAlgorithm<double, Matrix<double>, Tensor<double>>(options);
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
    [InlineData(0.5, 0.07)]
    [InlineData(0.0, 0.07)]
    [InlineData(2.0, 0.5)]
    public void EpisodeGradient_MatchesCentralDifferences(double contrastiveWeight, double temperature)
    {
        var learner = CreateLearner(o =>
        {
            o.ContrastiveWeight = contrastiveWeight;
            o.ContrastiveTemperature = temperature;
        });
        var task = CreateTask(seed: 3);

        var (_, body, projection) = learner.EpisodeGradientForTesting(task);
        double Loss() => learner.EpisodeLossForTesting(task);

        var model = learner.GetMetaModel();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(),
            v => model.SetParameters(new Vector<double>(v)), "body");
        CheckVector(Loss, projection.ToArray(), () => learner.ProjectionWeightsForTesting.ToArray(),
            v => learner.ProjectionWeightsForTesting = new Vector<double>(v), "projection");
    }

    [Fact]
    public void ContrastiveLoss_IsZeroWhenEveryAnchorsOnlyNegativeIsItsPositive()
    {
        // Eq. 2 for two examples of one class: P(i) and A(i) are the same single index, so the log ratio is
        // log(exp(s/tau) / exp(s/tau)) = 0 for both anchors, whatever the embedding is. The term the old code
        // computed - a "cosine similarity" between two scalars, over classes inferred from position - could not
        // satisfy this, because it did not read the labels at all.
        var x = new Matrix<double>(1, Features);
        for (int f = 0; f < Features; f++) x[0, f] = 0.4 * (f + 1);
        var other = new Matrix<double>(1, Features);
        for (int f = 0; f < Features; f++) other[0, f] = -0.7 * (f + 1);
        var y = new Tensor<double>(new[] { 1 });

        var task = new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = x, SupportSetY = y, QuerySetX = other, QuerySetY = y,
            NumWays = 1, NumShots = 1, NumQueryPerClass = 1, Name = "mcl-single-class",
        };

        double loss = CreateLearner(o => o.NumWays = 1).ContrastiveLossForTesting(task);
        Assert.Equal(0.0, loss, 10);
    }

    [Fact]
    public void ContrastiveLoss_IsZeroWhenNoAnchorHasAPositive()
    {
        // Every example is its own class, so P(i) is empty for all of them and the paper's sum has no terms.
        var support = new Matrix<double>(1, Features);
        for (int f = 0; f < Features; f++) support[0, f] = 0.2 * (f + 1);
        var query = new Matrix<double>(1, Features);
        for (int f = 0; f < Features; f++) query[0, f] = -0.5 * (f + 1);
        var supportY = new Tensor<double>(new[] { 1 });
        var queryY = new Tensor<double>(new[] { 1 });
        queryY[0] = 1;

        var task = new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support, SupportSetY = supportY, QuerySetX = query, QuerySetY = queryY,
            NumWays = Classes, NumShots = 1, NumQueryPerClass = 1, Name = "mcl-all-singletons",
        };

        Assert.Equal(0.0, CreateLearner().ContrastiveLossForTesting(task), 10);
    }

    [Fact]
    public void ContrastiveLoss_FallsWhenSameClassExamplesAreBroughtTogether()
    {
        // The objective's whole point: same-class projections closer together lowers it. Checked through the
        // embedding, since that is what MCL actually trains.
        var learner = CreateLearner();
        var spread = CreateTask(seed: 3);
        double before = learner.ContrastiveLossForTesting(spread);

        // Collapse every example of a class onto its centre, leaving the classes where they were.
        var tight = CreateTask(seed: 3);
        void Collapse(Matrix<double> x)
        {
            for (int c = 0; c < Classes; c++)
            {
                var mean = new double[Features];
                int count = 0;
                for (int r = c; r < x.Rows; r += Classes)
                {
                    for (int f = 0; f < Features; f++) mean[f] += x[r, f];
                    count++;
                }

                for (int r = c; r < x.Rows; r += Classes)
                {
                    for (int f = 0; f < Features; f++) x[r, f] = mean[f] / count;
                }
            }
        }

        Collapse(tight.SupportSetX);
        Collapse(tight.QuerySetX);

        Assert.True(learner.ContrastiveLossForTesting(tight) < before,
            $"collapsing each class onto its centre did not lower the contrastive loss: {before:G6} to " +
            $"{learner.ContrastiveLossForTesting(tight):G6}.");
    }

    [Fact]
    public void Adapt_LeavesTheMetaModelAlone()
    {
        // Adaptation used to rescale the shared backbone's parameters by a factor derived from the projections,
        // and the adapted model re-applied them on every Predict, so adapting to a task changed the meta-model
        // itself and every later task saw the previous one's rescaling.
        var learner = CreateLearner();
        var task = CreateTask(seed: 5);
        var model = learner.GetMetaModel();
        var before = model.GetParameters().ToArray();

        var adapted = learner.Adapt(task);
        var predictions = adapted.Predict(task.QuerySetX);
        var after = model.GetParameters().ToArray();

        Assert.Equal(before, after);
        Assert.Equal(task.QuerySetX.Rows, predictions.Shape[0]);
        Assert.Equal(Classes, predictions.Shape[1]);
    }

    [Fact]
    public void MetaTrain_MovesTheProjectionHeadByItsOwnGradient()
    {
        // The head used to be updated by simultaneous perturbation. It now has an exact gradient, so a step moves
        // it in the direction that gradient points.
        var learner = CreateLearner();
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(1), CreateTask(2) });

        var (_, _, gradient) = learner.EpisodeGradientForTesting(CreateTask(1));
        Assert.True(gradient.ToArray().Any(g => Math.Abs(g) > 1e-12), "the projection head has no gradient at all.");

        var before = learner.ProjectionWeightsForTesting.ToArray();
        learner.MetaTrain(batch);
        var after = learner.ProjectionWeightsForTesting.ToArray();

        Assert.Equal(before.Length, after.Length);
        Assert.True(before.Zip(after, (b, a) => Math.Abs(b - a)).Any(d => d > 1e-12),
            "meta-training left the projection head where it was.");
    }
}
