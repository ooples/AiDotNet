using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Relation Networks as Sung et al. 2018 state them - class embeddings summed, a concatenation relation module ending
/// in a sigmoid unit, trained by mean squared error onto the match indicators - and the learned extensions, all on the
/// same exact gradient (#2155).
/// </summary>
public class RelationNetworkExactGradientTests
{
    private const int Features = 3;
    private const int Width = 3;
    private const int Classes = 2;
    private const int Hidden = 4;

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
            NumWays = Classes, NumShots = 2, NumQueryPerClass = 2, Name = $"relation-{seed}",
        };
    }

    private static RelationNetworkAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<RelationNetworkOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new RelationNetworkOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            NumClasses = Classes,
            RelationHiddenDimension = Hidden,
            OuterLearningRate = 0.05,
            RandomSeed = 5,
        };
        configure?.Invoke(options);
        return new RelationNetworkAlgorithm<double, Matrix<double>, Tensor<double>>(options);
    }

    private static TaskBatch<double, Matrix<double>, Tensor<double>> Batch(params int[] seeds)
        => new TaskBatch<double, Matrix<double>, Tensor<double>>(seeds.Select(CreateTask).ToArray());

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
    [InlineData(RelationModuleType.Concatenate, RelationAggregationMethod.EmbeddingSum, false, false)]
    [InlineData(RelationModuleType.Convolution, RelationAggregationMethod.Mean, false, false)]
    [InlineData(RelationModuleType.Attention, RelationAggregationMethod.Max, false, false)]
    [InlineData(RelationModuleType.Transformer, RelationAggregationMethod.Attention, false, true)]
    [InlineData(RelationModuleType.Concatenate, RelationAggregationMethod.LearnedWeighting, true, false)]
    [InlineData(RelationModuleType.Attention, RelationAggregationMethod.EmbeddingSum, true, true)]
    public void EpisodeGradient_MatchesCentralDifferences(
        RelationModuleType type, RelationAggregationMethod pooling, bool multiHead, bool transform)
    {
        var learner = CreateLearner(o =>
        {
            o.RelationType = type;
            o.AggregationMethod = pooling;
            o.UseMultiHeadRelation = multiHead;
            o.NumHeads = 2;
            o.ApplyFeatureTransform = transform;
        });
        var task = CreateTask(seed: 3);

        // Move the zero- and identity-initialised parts off their start, so their gradients are not checked only at
        // a symmetric point.
        var first = learner.EpisodeGradientForTesting(task);
        if (first.Pooling.Length > 0)
            learner.PoolingWeightsForTesting = new Vector<double>(Enumerable.Range(0, first.Pooling.Length).Select(i => 0.2 * Math.Sin(i + 1)).ToArray());
        if (first.Shots.Length > 0)
            learner.ShotWeightsForTesting = new Vector<double>(Enumerable.Range(0, first.Shots.Length).Select(i => 0.3 - 0.5 * i).ToArray());
        if (first.Transform.Length > 0)
            learner.FeatureTransformForTesting = new Vector<double>(Enumerable.Range(0, first.Transform.Length)
                .Select(i => (i % (Width + 1) == 0 ? 1.0 : 0.0) + 0.1 * Math.Cos(i)).ToArray());

        var (_, body, relation, transformGradient, poolingGradient, shotGradient) = learner.EpisodeGradientForTesting(task);
        double Loss() => learner.EpisodeLossForTesting(task);

        var model = learner.GetMetaModel();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(), v => model.SetParameters(new Vector<double>(v)), "body");
        CheckVector(Loss, relation.ToArray(), () => learner.RelationWeightsForTesting.ToArray(),
            v => learner.RelationWeightsForTesting = new Vector<double>(v), "g");
        CheckVector(Loss, transformGradient.ToArray(), () => learner.FeatureTransformForTesting.ToArray(),
            v => learner.FeatureTransformForTesting = new Vector<double>(v), "A");
        CheckVector(Loss, poolingGradient.ToArray(), () => learner.PoolingWeightsForTesting.ToArray(),
            v => learner.PoolingWeightsForTesting = new Vector<double>(v), "U");
        CheckVector(Loss, shotGradient.ToArray(), () => learner.ShotWeightsForTesting.ToArray(),
            v => learner.ShotWeightsForTesting = new Vector<double>(v), "w");

        Assert.Equal(transform, transformGradient.Length > 0);
        Assert.Equal(pooling == RelationAggregationMethod.Attention, poolingGradient.Length > 0);
        Assert.Equal(pooling == RelationAggregationMethod.LearnedWeighting, shotGradient.Length > 0);
    }

    [Fact]
    public void PaperConfiguration_IsTheMeanSquaredErrorOfSummedClassRelations()
    {
        // Eq. 1 and 2 by hand: class feature = sum of its support embeddings; r = sigmoid(w2 relu(W1 [c; q] + b1) + b2);
        // loss = mean over (query, class) of (r - 1(y == c))^2.
        var learner = CreateLearner();
        var task = CreateTask(seed: 4);
        double actual = learner.EpisodeLossForTesting(task);
        var w = learner.RelationWeightsForTesting.ToArray();
        var body = learner.GetMetaModel();
        var support = body.Predict(task.SupportSetX);
        var query = body.Predict(task.QuerySetX);

        double Relation(double[] sample, double[] q)
        {
            var z = sample.Concat(q).ToArray();
            int b1 = Hidden * 2 * Width, w2 = b1 + Hidden, b2 = w2 + Hidden;
            double output = w[b2];
            for (int j = 0; j < Hidden; j++)
            {
                double pre = w[b1 + j];
                for (int k = 0; k < 2 * Width; k++) pre += w[j * 2 * Width + k] * z[k];
                output += w[w2 + j] * Math.Max(0.0, pre);
            }

            return 1.0 / (1.0 + Math.Exp(-output));
        }

        double expected = 0;
        int queryRows = task.QuerySetX.Rows, supportRows = task.SupportSetX.Rows;
        for (int qi = 0; qi < queryRows; qi++)
        {
            var q = Enumerable.Range(0, Width).Select(f => query[qi * Width + f]).ToArray();
            for (int c = 0; c < Classes; c++)
            {
                var feature = Enumerable.Range(0, Width).Select(f =>
                    Enumerable.Range(0, supportRows).Where(s => task.SupportSetY[s] == c).Sum(s => support[s * Width + f])).ToArray();
                double target = task.QuerySetY[qi] == c ? 1.0 : 0.0;
                expected += Math.Pow(Relation(feature, q) - target, 2);
            }
        }

        Assert.Equal(expected / (queryRows * Classes), actual, 10);
    }

    [Fact]
    public void Defaults_AreThePapers()
    {
        var options = new RelationNetworkOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width));

        Assert.Equal(RelationModuleType.Concatenate, options.RelationType);
        Assert.Equal(RelationAggregationMethod.EmbeddingSum, options.AggregationMethod);
        Assert.Equal(8, options.RelationHiddenDimension);
    }

    [Theory]
    [InlineData(RelationAggregationMethod.Attention)]
    [InlineData(RelationAggregationMethod.LearnedWeighting)]
    public void LearnedPoolingAtItsStart_IsTheMean(RelationAggregationMethod pooling)
    {
        var task = CreateTask(seed: 5);
        double mean = CreateLearner(o => o.AggregationMethod = RelationAggregationMethod.Mean).EpisodeLossForTesting(task);
        double learned = CreateLearner(o => o.AggregationMethod = pooling).EpisodeLossForTesting(task);

        Assert.Equal(mean, learned, 12);
    }

    [Fact]
    public void MetaTrain_LowersTheRelationLossOnSeparableTasks()
    {
        var learner = CreateLearner();
        var batch = Batch(100, 101, 102, 103);

        double first = learner.MetaTrain(batch);
        double last = first;
        for (int step = 0; step < 30; step++) last = learner.MetaTrain(batch);

        Assert.True(last < first, $"Meta-training on a fixed batch went from {first} to {last}.");
    }

    [Fact]
    public void Dropout_TrainsWhileAdaptationStaysDeterministic()
    {
        var learner = CreateLearner(o => o.RelationDropout = 0.5);
        var task = CreateTask(seed: 6);

        double loss = learner.MetaTrain(Batch(6, 7));
        var adapted = learner.Adapt(task);
        var once = adapted.Predict(task.QuerySetX).ToArray();
        var twice = adapted.Predict(task.QuerySetX).ToArray();

        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
        Assert.Equal(once, twice);
    }

    [Fact]
    public void EncoderDecay_ChangesTheUpdate()
    {
        var plain = CreateLearner();
        var decayed = CreateLearner(o => o.FeatureEncoderL2Reg = 0.5);

        plain.MetaTrain(Batch(8));
        decayed.MetaTrain(Batch(8));

        Assert.NotEqual(plain.GetMetaModel().GetParameters().ToArray(), decayed.GetMetaModel().GetParameters().ToArray());
    }

    [Fact]
    public void Adapt_ScoresAClassWithoutSupportZero()
    {
        var learner = CreateLearner(o => o.NumClasses = 3);
        var task = CreateTask(seed: 9);

        var scores = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { task.QuerySetX.Rows, 3 }, scores.Shape.ToArray());
        for (int r = 0; r < task.QuerySetX.Rows; r++)
        {
            Assert.Equal(0.0, scores[r * 3 + 2]);
            for (int c = 0; c < 2; c++) Assert.InRange(scores[r * 3 + c], 0.0, 1.0);
        }
    }

    [Fact]
    public void InvalidHeadsOrDropout_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => CreateLearner(o => { o.UseMultiHeadRelation = true; o.NumHeads = 0; }));
        Assert.Throws<ArgumentException>(() => CreateLearner(o => o.RelationDropout = 1.0));
    }
}
