using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using AiDotNet.Video.Segmentation;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Video;

public sealed class SAM2PaperFidelityTests
{
    private readonly ITestOutputHelper _output;

    public SAM2PaperFidelityTests(ITestOutputHelper output) => _output = output;

    private static SAM2Options CreateBoundedOptions() => new()
    {
        HieraEmbeddingDimension = 16,
        HieraStageDepths = [1, 1, 1, 1],
        HieraInitialHeadCount = 1,
        HieraWindowSizes = [8, 4, 2, 1],
        HieraGlobalAttentionBlockIndexes = [2],
        ModelDimension = 16,
        MemoryDimension = 8,
        DecoderHeadCount = 4,
        MemoryAttentionLayerCount = 4,
        MaskDecoderDepth = 2,
        MaskDecoderMlpDimension = 64
    };

    private static NeuralNetworkArchitecture<double> CreateArchitecture() => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.BinaryClassification,
        inputHeight: 32,
        inputWidth: 32,
        inputDepth: 3,
        outputSize: 1)
    {
        RandomSeed = 17,
    };

    private static SAM2<double> CreateModel() => new(
        CreateArchitecture(),
        modelSize: SAM2ModelSize.Tiny,
        memoryBankSize: 2,
        options: CreateBoundedOptions());

    private static Tensor<double> CreateImage(double offset = 0.0)
    {
        var image = new Tensor<double>([1, 3, 32, 32]);
        for (int i = 0; i < image.Length; i++)
            image[i] = offset + ((i % 29) - 14) / 29.0;
        return image;
    }

    private static double WorstDifference(Tensor<double> left, Tensor<double> right)
    {
        Assert.Equal(left.Shape.ToArray(), right.Shape.ToArray());
        double worst = 0.0;
        for (int i = 0; i < left.Length; i++)
            worst = Math.Max(worst, Math.Abs(left[i] - right[i]));
        return worst;
    }

    private static double[] HeadParameters(IEnumerable<DenseLayer<double>> layers) =>
        layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();

    private static double WorstDifference(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        Assert.Equal(left.Count, right.Count);
        double worst = 0.0;
        for (int i = 0; i < left.Count; i++)
            worst = Math.Max(worst, Math.Abs(left[i] - right[i]));
        return worst;
    }

    private static double PublishedMaskLoss(Tensor<double> predicted, Tensor<double> target)
    {
        double focal = new FocalLoss<double>(gamma: 2.0, alpha: 0.25)
            .CalculateLoss(predicted.ToVector(), target.ToVector());
        double intersection = 0.0;
        double predictedArea = 0.0;
        double targetArea = 0.0;
        for (int i = 0; i < predicted.Length; i++)
        {
            intersection += predicted[i] * target[i];
            predictedArea += predicted[i];
            targetArea += target[i];
        }
        double dice = 1.0 - ((2.0 * intersection + 1.0) / (predictedArea + targetArea + 1.0));
        return 20.0 * focal + dice;
    }

    private static double[] CandidateMaskLosses(Tensor<double> candidates, Tensor<double> target)
    {
        var losses = new double[candidates.Shape[1]];
        for (int candidate = 0; candidate < losses.Length; candidate++)
        {
            var mask = new Tensor<double>([candidates.Shape[0], 1, candidates.Shape[2], candidates.Shape[3]]);
            for (int batch = 0; batch < candidates.Shape[0]; batch++)
            for (int h = 0; h < candidates.Shape[2]; h++)
            for (int w = 0; w < candidates.Shape[3]; w++)
                mask[batch, 0, h, w] = candidates[batch, candidate, h, w];
            losses[candidate] = PublishedMaskLoss(mask, target);
        }
        return losses;
    }

    [Fact]
    public async Task PublishedHieraPresets_MatchMetaConfigurations()
    {
        await Task.Yield();

        var tiny = SAM2<double>.ResolveHieraPreset(SAM2ModelSize.Tiny, new SAM2Options());
        Assert.Equal(96, tiny.Embedding);
        Assert.Equal([1, 2, 7, 2], tiny.Depths);
        Assert.Equal(1, tiny.Heads);
        Assert.Equal([8, 4, 14, 7], tiny.Windows);
        Assert.Equal([5, 7, 9], tiny.Globals);

        var small = SAM2<double>.ResolveHieraPreset(SAM2ModelSize.Small, new SAM2Options());
        Assert.Equal(96, small.Embedding);
        Assert.Equal([1, 2, 11, 2], small.Depths);
        Assert.Equal(1, small.Heads);
        Assert.Equal([7, 10, 13], small.Globals);

        var @base = SAM2<double>.ResolveHieraPreset(SAM2ModelSize.Base, new SAM2Options());
        Assert.Equal(112, @base.Embedding);
        Assert.Equal([2, 3, 16, 3], @base.Depths);
        Assert.Equal(2, @base.Heads);
        Assert.Equal([12, 16, 20], @base.Globals);

        var large = SAM2<double>.ResolveHieraPreset(SAM2ModelSize.Large, new SAM2Options());
        Assert.Equal(144, large.Embedding);
        Assert.Equal([2, 6, 36, 4], large.Depths);
        Assert.Equal(2, large.Heads);
        Assert.Equal([8, 4, 16, 8], large.Windows);
        Assert.Equal([23, 33, 43], large.Globals);
    }

    [Fact]
    public async Task BoundedFixture_ScalesCapacityWithoutRemovingPaperModules()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);

        Assert.Equal([1, 1, 1, 1], pipeline.ImageEncoder.StageDepths);
        Assert.Equal([2], pipeline.ImageEncoder.GlobalAttentionBlockIndexes);
        Assert.Equal(16, pipeline.ModelDimension);
        Assert.Equal(8, pipeline.MemoryDimension);
        Assert.Equal(4, pipeline.MemoryAttentionLayerCount);
        Assert.Equal(2, pipeline.DecoderDepth);
        Assert.Equal(4, pipeline.CandidateCount);
    }

    [Fact]
    public async Task Decoder_EmitsFourMasksQualityPresenceAndObjectPointer()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);
        var encoded = pipeline.AddNoMemoryEmbedding(pipeline.EncodeImage(CreateImage()));

        var decoded = pipeline.Decode(encoded, sparsePrompt: null, densePrompt: null);

        Assert.Equal([1, 4, 8, 8], decoded.Masks.Shape.ToArray());
        Assert.Equal([1, 4], decoded.IouScores.Shape.ToArray());
        Assert.Equal([1, 1], decoded.ObjectPresenceScores.Shape.ToArray());
        Assert.Equal([1, 1, 16], decoded.ObjectPointer.Shape.ToArray());
    }

    [Fact]
    public async Task MemoryEncoder_IsConditionedOnThePredictedMask()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);
        var imageFeatures = pipeline.EncodeImage(CreateImage());
        var background = new Tensor<double>([1, 1, 8, 8]);
        var foreground = new Tensor<double>([1, 1, 8, 8]);
        foreground.Fill(1.0);

        var backgroundMemory = pipeline.EncodeMemory(imageFeatures, background);
        pipeline.ResetState();
        var foregroundMemory = pipeline.EncodeMemory(imageFeatures, foreground);

        Assert.Equal([1, 8, 2, 2], backgroundMemory.Shape.ToArray());
        Assert.True(
            WorstDifference(backgroundMemory, foregroundMemory) > 1e-10,
            "The memory encoder ignored the predicted mask and collapsed to an image-feature clone.");
    }

    [Fact]
    public async Task MemoryAttention_ChangesTheCurrentFrameAndUsesObjectPointers()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);
        var imageFeatures = pipeline.EncodeImage(CreateImage());
        var firstFrame = pipeline.Decode(
            pipeline.AddNoMemoryEmbedding(imageFeatures), sparsePrompt: null, densePrompt: null);
        var selected = new Tensor<double>([1, 1, 8, 8]);
        selected.Fill(1.0);
        var memory = pipeline.EncodeMemory(imageFeatures, selected);

        var withoutMemory = pipeline.ApplyMemoryAttention(
            imageFeatures, Array.Empty<Tensor<double>>(), Array.Empty<Tensor<double>>());
        var withMemory = pipeline.ApplyMemoryAttention(
            imageFeatures, [memory], [firstFrame.ObjectPointer]);

        Assert.True(
            WorstDifference(withoutMemory, withMemory) > 1e-10,
            "Memory attention did not affect the current-frame embedding.");
    }

    [Fact]
    public async Task ResetState_ReplaysExactlyWhileDifferentImagesRemainDistinguishable()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);

        var first = pipeline.Forward(CreateImage());
        pipeline.ResetState();
        var replay = pipeline.Forward(CreateImage());
        pipeline.ResetState();
        var changed = pipeline.Forward(CreateImage(offset: 0.75));

        Assert.Equal(0.0, WorstDifference(first, replay));
        Assert.True(
            WorstDifference(first, changed) > 1e-10,
            "The native SAM2 pipeline is insensitive to its image input.");
    }

    [Fact]
    public async Task NativeTraining_ReducesPublishedMaskLossAndUpdatesBothAuxiliaryHeads()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);
        var input = CreateImage();
        var target = new Tensor<double>([1, 1, 8, 8]);
        for (int h = 2; h < 6; h++)
        {
            for (int w = 2; w < 6; w++) target[0, 0, h, w] = 1.0;
        }

        pipeline.ResetState();
        _ = pipeline.Forward(input);
        double[] candidateLossesBefore = CandidateMaskLosses(pipeline.LastMasks, target);
        double beforeLoss = candidateLossesBefore.Min();
        double[] iouScoresBefore = pipeline.LastIouScores.ToArray();
        var iouBefore = HeadParameters(pipeline.IouHeadLayers);
        var presenceBefore = HeadParameters(pipeline.ObjectPresenceHeadLayers);

        for (int step = 0; step < 20; step++)
        {
            model.ResetState();
            model.Train(input, target);
        }

        model.ResetState();
        var afterOutput = pipeline.Forward(input);
        double[] candidateLossesAfter = CandidateMaskLosses(pipeline.LastMasks, target);
        double afterLoss = candidateLossesAfter.Min();
        double[] iouScoresAfter = pipeline.LastIouScores.ToArray();
        var iouAfter = HeadParameters(pipeline.IouHeadLayers);
        var presenceAfter = HeadParameters(pipeline.ObjectPresenceHeadLayers);
        double iouDelta = WorstDifference(iouBefore, iouAfter);
        double presenceDelta = WorstDifference(presenceBefore, presenceAfter);

        _output.WriteLine($"Published oracle mask loss: {beforeLoss:R} -> {afterLoss:R}");
        _output.WriteLine($"Candidate losses before: {string.Join(", ", candidateLossesBefore.Select(x => x.ToString("R")))}");
        _output.WriteLine($"Candidate losses after: {string.Join(", ", candidateLossesAfter.Select(x => x.ToString("R")))}");
        _output.WriteLine($"IoU scores before: {string.Join(", ", iouScoresBefore.Select(x => x.ToString("R")))}");
        _output.WriteLine($"IoU scores after: {string.Join(", ", iouScoresAfter.Select(x => x.ToString("R")))}");
        _output.WriteLine($"IoU-head maximum parameter delta: {iouDelta:R}");
        _output.WriteLine($"Presence-head maximum parameter delta: {presenceDelta:R}");

        Assert.All(afterOutput.ToArray(), value =>
            Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        Assert.Single(model.LastTrainingMaskIndices);
        Assert.InRange(model.LastTrainingMaskIndices[0], 0, pipeline.CandidateCount - 1);
        Assert.True(afterLoss < beforeLoss,
            $"Published oracle 20:1 focal/dice loss did not decrease: {beforeLoss:R} -> {afterLoss:R}.");
        Assert.True(iouDelta > 0.0,
            "The IoU-regression head was disconnected from the native SAM2 training objective.");
        Assert.True(presenceDelta > 0.0,
            "The object-presence head was disconnected from the native SAM2 training objective.");
    }

    [Fact]
    public async Task PipelineClone_PreservesConstructionStateAndParameters()
    {
        await Task.Yield();
        using var model = CreateModel();
        var pipeline = Assert.IsType<SAM2NativePipelineLayer<double>>(model.NativePipeline);
        var clone = Assert.IsType<SAM2NativePipelineLayer<double>>(pipeline.Clone());

        Assert.Equal(pipeline.ParameterCount, clone.ParameterCount);
        Assert.Equal(pipeline.ImageEncoder.StageDepths, clone.ImageEncoder.StageDepths);
        Assert.Equal(
            pipeline.ImageEncoder.GlobalAttentionBlockIndexes,
            clone.ImageEncoder.GlobalAttentionBlockIndexes);

        var input = CreateImage();
        Assert.Equal(0.0, WorstDifference(pipeline.Forward(input), clone.Forward(input)));
    }
}
