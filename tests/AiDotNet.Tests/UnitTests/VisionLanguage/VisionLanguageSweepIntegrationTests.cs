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
public class VisionLanguageSweepIntegrationTests
{
    private const double Tolerance = 1e-9;
    private static Tensor<double> Random(int seed, params int[] shape)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
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
    public void GroundingSelect_TiedConfidences_KeepCandidateOrder()
    {
        // Twenty disjoint boxes with one confidence, except that box 4 repeats box 3. Greedy NMS visits equal
        // confidences in candidate order, so box 3 is kept, box 4 is suppressed, and the rest come out in order.
        // An unstable sort (introsort permutes equal keys past 16 items) reorders them.
        const int count = 20;
        var boxData = new double[count * 4];
        for (int i = 0; i < count; i++)
        {
            int slot = i == 4 ? 3 : i;
            boxData[i * 4] = slot * 0.05;
            boxData[i * 4 + 1] = 0.0;
            boxData[i * 4 + 2] = slot * 0.05 + 0.04;
            boxData[i * 4 + 3] = 0.04;
        }
        var boxes = new Tensor<double>(new[] { count, 4 }, new Vector<double>(boxData));
        var confidences = Tensor<double>.CreateDefault(new[] { count }, 0.5);
        var eligible = Tensor<double>.CreateDefault(new[] { count }, 1.0);

        var kept = GroundingDetections<double>.Select(AiDotNetEngine.Current, boxes, confidences, eligible, count, 0.5);

        var expected = new List<double>();
        for (int i = 0; i < count; i++)
        {
            if (i == 4) continue;
            for (int f = 0; f < 4; f++) expected.Add(boxData[i * 4 + f]);
            expected.Add(0.5);
        }
        Assert.Equal(expected.ToArray(), kept.ToArray());
    }

    [Fact]
    public void CosineToEach_Half_LongRowNormDoesNotSaturateAt2048()
    {
        // 4096 entries of 1/64 scale to 4096 ones, whose Half running sum stops at 2048: the query norm came out
        // as sqrt(2048) / 64 instead of 1, and the cosine to its first half as 1 instead of 1 / sqrt(2).
        var query = Tensor<Half>.CreateDefault(new[] { 4096 }, (Half)(1.0 / 64));
        var candidate = new Tensor<Half>(new[] { 4096 });
        for (int i = 0; i < 2048; i++) candidate[i] = (Half)(1.0 / 64);

        var scores = EmbeddingScores<Half>.CosineToEach(AiDotNetEngine.Current, query, new[] { candidate, query });

        Assert.Equal(1.0 / Math.Sqrt(2.0), (double)scores[0], 1e-3);
        Assert.Equal(1.0, (double)scores[1], 1e-3);
    }
}
