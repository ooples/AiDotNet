using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Medical SAM 2's self-sorting memory bank (Zhu et al. 2024, eqs. 2-5, and the official code's alternatives) and the
/// model's image-set, volume and training surfaces.
/// </summary>
public class MedicalSam2MemoryBankTests
{
    private static Tensor<double> Vector(params double[] values) => new(new[] { values.Length }, new Vector<double>(values));

    private static Sam2SortedMemoryBank<double> Bank(int capacity, double? threshold = null,
        MemoryBankUpdateRule rule = MemoryBankUpdateRule.TopKDissimilarity, MemoryBankResampling resampling = MemoryBankResampling.NormalizedSimilarity)
        => new(new Sam2BankSettings(capacity, threshold, rule, resampling));

    [Fact]
    public void TopKDissimilarity_KeepsTheMostMutuallyDissimilarCandidates()
    {
        // Memories along directions; of the near-duplicates 0° and 10°, the bank of three drops the one less dissimilar to the
        // rest when the fourth arrives.
        double Rad(double degrees) => degrees * Math.PI / 180;
        var bank = Bank(3);
        var memories = new[] { 0.0, 10.0, 90.0, 200.0 }.Select(d => Vector(Math.Cos(Rad(d)), Math.Sin(Rad(d)))).ToList();
        foreach (var m in memories) bank.Add(m, m, 0.9);
        Assert.Equal(3, bank.Count);
        var kept = bank.Entries.Select(e => Math.Round(Math.Atan2(e.MemoryKey[1], e.MemoryKey[0]) * 180 / Math.PI)).ToList();
        Assert.Contains(90.0, kept);
        Assert.Contains(-160.0, kept);
        // D(0°) = (1 - cos 10°) + 1 + (1 - cos 200°) = 2.955, D(10°) = (1 - cos 10°) + (1 - cos 80°) + (1 - cos 190°) = 2.826,
        // D(90°) = 3.168 and D(200°) = 5.267: 10° goes.
        Assert.Contains(0.0, kept);
        Assert.DoesNotContain(10.0, kept);
    }

    [Fact]
    public void ConfidenceThreshold_KeepsLowConfidenceMemoriesOut()
    {
        var bank = Bank(4, threshold: 0.6);
        bank.Add(Vector(1, 0), Vector(1, 0), 0.5);
        Assert.Equal(0, bank.Count);
        bank.Add(Vector(1, 0), Vector(1, 0), 0.7);
        Assert.Equal(1, bank.Count);
    }

    [Fact]
    public void ReplaceRedundant_IsTheOfficialCodesRule()
    {
        var bank = Bank(2, rule: MemoryBankUpdateRule.ReplaceRedundant);
        bank.Add(Vector(1, 0), Vector(1, 0), 0.9);
        bank.Add(Vector(1, 0.1), Vector(1, 0.1), 0.9);
        // The new memory is less similar to its least-similar entry than that entry is to the other: it replaces the other.
        bank.Add(Vector(-1, 0), Vector(-1, 0), 0.9);
        Assert.Equal(2, bank.Count);
        Assert.Equal(-1.0, bank.Entries[1].MemoryKey[0], 12);
    }

    [Fact]
    public void Resampling_DrawsInProportionToSimilarity_AndRejectsNegativeSimilarities()
    {
        var bank = Bank(2);
        bank.Add(Vector(1, 0), Vector(1, 0), 0.9);
        bank.Add(Vector(0, 1), Vector(0.6, 0.8), 0.9);
        // Similarities to (1, 0): 1 and 0.6, so p = 1/1.6 and 0.6/1.6.
        var random = new Random(3);
        int first = 0, draws = 0;
        for (int i = 0; i < 4000; i++)
            foreach (var e in bank.Resample(Vector(1, 0), random)) { draws++; if (e.EmbeddingKey[0] == 1.0) first++; }
        Assert.InRange(first / (double)draws, 1 / 1.6 - 0.02, 1 / 1.6 + 0.02);
        Assert.Throws<InvalidOperationException>(() => bank.Resample(Vector(-1, 0.1), random));
        var softmax = Bank(2, resampling: MemoryBankResampling.Softmax);
        softmax.Add(Vector(1, 0), Vector(1, 0), 0.9);
        softmax.Add(Vector(0, 1), Vector(0, 1), 0.9);
        Assert.Equal(2, softmax.Resample(Vector(-1, 0.1), random).Count);
    }

    private static MedicalSAM2Options Small() => new()
    {
        ImageSize = 32, MemoryBankCapacity = 3, HieraEmbeddingDimension = 8, HieraStageDepths = new[] { 1, 1, 1, 1 },
        HieraInitialHeadCount = 1, HieraWindowSizes = new[] { 4, 4, 2, 1 }, HieraGlobalAttentionBlockIndexes = new[] { 2 },
        ModelDimension = 32, MemoryDimension = 16, DecoderHeadCount = 2, MaskDecoderMlpDimension = 32, MemoryAttentionLayerCount = 1,
        MemoryAttentionFeedForwardDimension = 32, MemoryFuserDimension = 32, MemoryBankSize = 3, Seed = 5,
        MemoryResampling = MemoryBankResampling.Softmax, WarmupSteps = 0, LearningRate = 1e-3, ImageEncoderLearningRate = 1e-3, CorrectionPoints = 1,
    };

    private static NeuralNetworkArchitecture<double> Architecture(int channels) => new(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.BinaryClassification,
        inputHeight: 32, inputWidth: 32, inputDepth: channels, outputSize: 1);

    [Fact(Timeout = 600000)]
    public async Task ImageSets_AndVolumes_GiveProbabilitiesAtTheirSize()
    {
        await Task.Yield();
        using var model = new MedicalSAM2<double>(Architecture(1), options: Small());
        var images = Enumerable.Range(0, 4).Select(t =>
        {
            var image = new Tensor<double>(new[] { 1, 32, 32 });
            for (int i = 0; i < image.Length; i++) image[i] = 0.5 + 0.5 * Math.Sin(0.11 * i + t);
            return image;
        }).ToList();
        var masks = model.SegmentImageSet(images, 0, new float[,] { { 16, 12 } }, new[] { 1 });
        Assert.Equal(4, masks.Count);
        Assert.All(masks, m => Assert.Equal(new[] { 32, 32 }, m.Shape.ToArray()));
        var volume = new Tensor<double>(new[] { 1, 6, 32, 32 });
        for (int i = 0; i < volume.Length; i++) volume[i] = 0.5 + 0.5 * Math.Cos(0.07 * i);
        var axial = new Tensor<double>(new[] { 32, 32 });
        for (int y = 8; y < 20; y++) for (int x = 10; x < 22; x++) axial[y, x] = 1;
        var coronal = new Tensor<double>(new[] { 6, 32 });
        for (int z = 1; z < 5; z++) for (int x = 10; x < 22; x++) coronal[z, x] = 1;
        var result = model.SegmentVolume(volume, new Dictionary<int, Tensor<double>> { [3] = axial }, new Dictionary<int, Tensor<double>> { [12] = coronal });
        Assert.Equal(new[] { 6, 32, 32 }, result.Labels.Shape.ToArray());
        Assert.All(result.Probabilities.ToArray(), v => Assert.InRange(v, 0.0, 1.0));
    }

    [Fact(Timeout = 600000)]
    public async Task Training_OnAnImageSet_LowersTheObjective()
    {
        await Task.Yield();
        using var model = new MedicalSAM2<double>(Architecture(3), options: Small());
        var images = new Tensor<double>(new[] { 3, 3, 32, 32 });
        for (int i = 0; i < images.Length; i++) images[i] = 0.5 + 0.5 * Math.Sin(0.29 * i);
        var masks = new Tensor<double>(new[] { 3, 32, 32 });
        for (int t = 0; t < 3; t++) for (int y = 8; y < 20; y++) for (int x = 10 + t; x < 22 + t; x++) masks[t, y, x] = 1;
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(images, masks);
        for (int i = 0; i < 15; i++) model.Train(images, masks);
        double after = objective.EvaluateTrainingObjective(images, masks);
        Assert.True(after < before, $"The objective did not fall: {before} -> {after}");
    }
}
