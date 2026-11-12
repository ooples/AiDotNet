using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage;
using AiDotNet.VisionLanguage.Encoders;
using AiDotNet.VisionLanguage.Foundational;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Reference checks for the vision-language engine-op conversions and the bugs fixed with them.
/// </summary>
public class VisionLanguageEngineSweepTests
{
    private const double Tolerance = 1e-9;

    private static Tensor<double> Random(int seed, params int[] shape)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    [Fact]
    public void CosineToEach_MatchesVectorHelperForEveryCandidate()
    {
        var query = Random(1, 6);
        var candidates = new[] { Random(2, 6), Random(3, 6), new Tensor<double>(new[] { 6 }), Random(4, 6) };

        var scores = EmbeddingScores<double>.CosineToEach(AiDotNetEngine.Current, query, candidates);

        Assert.Equal(new[] { 4 }, scores.Shape.ToArray());
        for (int c = 0; c < candidates.Length; c++)
            Assert.Equal(VectorHelper.CosineSimilarity(query.ToVector(), candidates[c].ToVector()), scores[c], Tolerance);
        Assert.Equal(0.0, scores[2]);
    }

    [Fact]
    public void CosineToEach_LargeValues_DoNotOverflowFloat()
    {
        // Squares of 1e20 overflow float; the pre-scaled rows still give the exact cosine.
        var query = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1e20f, 0f, 1e20f }));
        var candidate = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1e20f, 0f, 0f }));

        var scores = EmbeddingScores<float>.CosineToEach(AiDotNetEngine.Current, query, new[] { candidate });

        Assert.Equal(1.0 / Math.Sqrt(2.0), scores[0], 1e-6);
    }

    [Fact]
    public void SigmoidPairScores_ScoresEachPairOnItsOwn()
    {
        // SigLIP (Zhai et al., 2023): p(match) = sigmoid(cos / temperature + bias) per pair, not renormalized.
        var query = Random(5, 8);
        var candidates = new[] { Random(6, 8), Random(7, 8), Random(8, 8) };
        const double temperature = 0.1, bias = -10.0;

        var all = EmbeddingScores<double>.SigmoidPairScores(AiDotNetEngine.Current, query, candidates, temperature, bias);
        var first = EmbeddingScores<double>.SigmoidPairScores(AiDotNetEngine.Current, query, candidates.Take(1).ToArray(),
            temperature, bias);

        for (int c = 0; c < candidates.Length; c++)
        {
            double cos = VectorHelper.CosineSimilarity(query.ToVector(), candidates[c].ToVector());
            Assert.Equal(Sigmoid(cos / temperature + bias), all[c], Tolerance);
        }
        Assert.Equal(all[0], first[0], Tolerance);
    }

    [Fact]
    public void SigLIPDefaults_UsePaperInitialLogitScaleAndBias()
    {
        // t' = log 10 (t = 10) and b = -10; Temperature is 1 / t.
        Assert.Equal(0.1, new SigLIPOptions().Temperature, precision: 12);
        Assert.Equal(-10.0, new SigLIPOptions().SigmoidBias, precision: 12);
        Assert.Equal(0.1, new SigLIP2Options().Temperature, precision: 12);
        Assert.Equal(-10.0, new SigLIP2Options().SigmoidBias, precision: 12);
    }

    private static SigLIP2<double> SmallSigLIP2() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification, inputHeight: 32, inputWidth: 32, inputDepth: 3,
            outputSize: 16),
        new SigLIP2Options
        {
            ImageSize = 32,
            PatchSize = 16,
            VisionEmbeddingDim = 16,
            TextEmbeddingDim = 16,
            ProjectionDim = 16,
            NumVisionLayers = 1,
            NumTextLayers = 1,
            NumVisionHeads = 2,
            NumTextHeads = 2,
            CaptioningDecoderDim = 16,
            NumCaptioningDecoderLayers = 1,
            NumCaptioningDecoderHeads = 2,
            MimDecoderDim = 16,
            NumMimDecoderLayers = 1,
            VocabSize = 40,
            MaxSequenceLength = 8,
            MaxCaptionLength = 3,
            DropoutRate = 0.0
        });

    [Fact]
    public void SigLIP2GenerateCaption_DecodesEveryStepAtTheDecoderWidth()
    {
        using var model = SmallSigLIP2();

        var logits = model.GenerateCaption(Random(12, 3, 32, 32));

        Assert.Equal(3 * 40, logits.Length);
        for (int i = 0; i < logits.Length; i++)
            Assert.False(double.IsNaN(logits[i]) || double.IsInfinity(logits[i]));
    }

    [Fact]
    public void ViLBERTMeanPool_Rank4_AveragesEveryTokenAxis()
    {
        var input = Random(13, 2, 3, 2, 4);

        var pooled = ViLBERT<double>.MeanPoolOverTokens(AiDotNetEngine.Current, input);

        Assert.Equal(new[] { 2, 4 }, pooled.Shape.ToArray());
        for (int b = 0; b < 2; b++)
            for (int k = 0; k < 4; k++)
            {
                double sum = 0;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        sum += input[b, i, j, k];
                Assert.Equal(sum / 6, pooled[b, k], Tolerance);
            }
    }

    [Fact]
    public void GroundingSelect_SuppressesInDescendingConfidenceOrder()
    {
        // Two heavily overlapping boxes; the later one is more confident and must be the one kept.
        var boxes = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.9 }));
        var confidences = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 0.6, 0.9 }));
        var eligible = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 1.0, 1.0 }));

        var kept = GroundingDetections<double>.Select(AiDotNetEngine.Current, boxes, confidences, eligible, 10, 0.5);

        Assert.Equal(new[] { 0.0, 0.0, 1.0, 0.9, 0.9 }, kept.ToArray());
    }

    [Fact]
    public void GroundingSelect_ConsidersOnlyTheFirstEligibleCandidates()
    {
        var boxes = new Tensor<double>(new[] { 3, 4 }, new Vector<double>(new[]
        {
            0.0, 0.0, 0.2, 0.2,
            0.5, 0.5, 0.7, 0.7,
            0.8, 0.8, 1.0, 1.0
        }));
        var confidences = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 0.7, 0.8, 0.9 }));
        var eligible = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 0.0, 1.0, 1.0 }));

        var kept = GroundingDetections<double>.Select(AiDotNetEngine.Current, boxes, confidences, eligible, 1, 0.5);

        Assert.Equal(new[] { 0.5, 0.5, 0.7, 0.7, 0.8 }, kept.ToArray());
    }
}
