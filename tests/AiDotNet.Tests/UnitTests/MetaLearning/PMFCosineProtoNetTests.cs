using System;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// PMF's meta-training stage (Hu et al. 2022) is a ProtoNet with a cosine classifier: prototypes are the support means,
/// and a query's logits are its cosine similarity to each prototype divided by the temperature (the reference
/// implementation's <c>scale_cls = 10</c>). The expected values below come from numpy:
/// <code>
/// P = [S[ys == c].mean(0) for c in (0, 1)]; C = normalize(Q) @ normalize(P).T; L = s * C
/// loss = mean(logsumexp(L, 1) - L[i, yq[i]]);  dloss/ds = mean(sum_k softmax(L)_k C_k - C[i, yq[i]])
/// </code>
/// </summary>
public class PMFCosineProtoNetTests
{
    private const int Width = 2;

    // Identity weights and zero bias: each embedding is its input row, so the prototypes are checkable by hand.
    private static LinearEmbeddingModel IdentityEmbedding()
    {
        var model = new LinearEmbeddingModel(Width, Width);
        model.SetParameters(new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 }));
        return model;
    }

    // Support: class 0 = (2,0), (2,2) -> prototype (2,1); class 1 = (0,1), (0,3) -> prototype (0,2).
    // Query: (1,1) of class 0 and (3,1) of class 1.
    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> HandTask() => new()
    {
        SupportSetX = new Matrix<double>(new[,] { { 2.0, 0.0 }, { 0.0, 1.0 }, { 2.0, 2.0 }, { 0.0, 3.0 } }),
        SupportSetY = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 })),
        QuerySetX = new Matrix<double>(new[,] { { 1.0, 1.0 }, { 3.0, 1.0 } }),
        QuerySetY = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.0, 1.0 })),
        NumWays = 2, NumShots = 2, NumQueryPerClass = 1, Name = "pmf-hand",
    };

    private static PMFAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<PMFOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new PMFOptions<double, Matrix<double>, Tensor<double>>(IdentityEmbedding());
        configure?.Invoke(options);
        return new PMFAlgorithm<double, Matrix<double>, Tensor<double>>(options);
    }

    [Fact]
    public void Defaults_AreThePapersCosineClassifier()
    {
        var options = new PMFOptions<double, Matrix<double>, Tensor<double>>(IdentityEmbedding());

        Assert.Equal(PMFDistanceMetric.Cosine, options.DistanceMetric);
        Assert.Equal(0.1, options.Temperature);
        Assert.True(options.LearnTemperature);
        Assert.Equal(10.0, CreateLearner().CosineScaleForTesting[0], 12);
    }

    [Theory]
    [InlineData(0.1, 9.486832980505136, 7.071067811865475, 9.899494936611665, 3.1622776601683795, 3.4119684359387685)]
    [InlineData(0.5, 1.8973665961010273, 1.414213562373095, 1.9798989873223332, 0.6324555320336759, 1.0294748360240713)]
    public void Logits_AreCosineToSupportMeans_OverTemperature(
        double temperature, double l00, double l01, double l10, double l11, double expectedLoss)
    {
        var learner = CreateLearner(o => o.Temperature = temperature);
        var task = HandTask();

        // Closed forms: cos((1,1),(2,1)) = 3/sqrt(10), cos((1,1),(0,2)) = 1/sqrt(2),
        // cos((3,1),(2,1)) = 7/sqrt(50), cos((3,1),(0,2)) = 1/sqrt(10).
        double scale = 1.0 / temperature;
        Assert.Equal(scale * 3.0 / Math.Sqrt(10.0), l00, 12);
        Assert.Equal(scale / Math.Sqrt(2.0), l01, 12);
        Assert.Equal(scale * 7.0 / Math.Sqrt(50.0), l10, 12);
        Assert.Equal(scale / Math.Sqrt(10.0), l11, 12);

        var logits = learner.EpisodeLogitsForTesting(task);
        Assert.Equal(2, logits.Shape[0]);
        Assert.Equal(2, logits.Shape[1]);
        Assert.Equal(l00, logits[0, 0], 9);
        Assert.Equal(l01, logits[0, 1], 9);
        Assert.Equal(l10, logits[1, 0], 9);
        Assert.Equal(l11, logits[1, 1], 9);

        Assert.Equal(expectedLoss, learner.EpisodeGradientForTesting(task).Loss, 9);
    }

    [Theory]
    [InlineData(0.1, 0.3265598116137726)]
    [InlineData(0.5, 0.2212886211426987)]
    public void TemperatureGradient_MatchesNumpy(double temperature, double expected)
    {
        var learner = CreateLearner(o => o.Temperature = temperature);

        var scale = learner.EpisodeGradientForTesting(HandTask()).Scale;

        Assert.Equal(1, scale.Length);
        Assert.Equal(expected, scale[0], 9);
    }

    [Fact]
    public void SquaredEuclidean_IsNegativeSquaredDistanceToSupportMeans()
    {
        var learner = CreateLearner(o => o.DistanceMetric = PMFDistanceMetric.SquaredEuclidean);

        var logits = learner.EpisodeLogitsForTesting(HandTask());

        // |(1,1)-(2,1)|^2 = 1, |(1,1)-(0,2)|^2 = 2, |(3,1)-(2,1)|^2 = 1, |(3,1)-(0,2)|^2 = 10.
        Assert.Equal(-1.0, logits[0, 0], 9);
        Assert.Equal(-2.0, logits[0, 1], 9);
        Assert.Equal(-1.0, logits[1, 0], 9);
        Assert.Equal(-10.0, logits[1, 1], 9);
    }

    [Fact]
    public void MetaTrain_LearnsTheTemperature_OnlyWhenAsked()
    {
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { HandTask() });

        var learned = CreateLearner(o => o.OuterLearningRate = 0.1);
        learned.MetaTrain(batch);
        Assert.NotEqual(10.0, learned.CosineScaleForTesting[0]);

        var fixedScale = CreateLearner(o =>
        {
            o.OuterLearningRate = 0.1;
            o.LearnTemperature = false;
        });
        fixedScale.MetaTrain(batch);
        Assert.Equal(10.0, fixedScale.CosineScaleForTesting[0], 12);
    }

    [Fact]
    public void Constructor_RejectsNonPositiveTemperature()
    {
        Assert.Throws<ArgumentException>(() => CreateLearner(o => o.Temperature = 0.0));
    }
}
