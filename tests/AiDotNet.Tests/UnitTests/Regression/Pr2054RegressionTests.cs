using System.Reflection;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Regression;

/// <summary>
/// Focused contracts for the regressions introduced while consolidating parameter manifests and
/// adding normalization-aware convolution bias handling in PR 2054.
/// </summary>
public class Pr2054RegressionTests
{
    [Fact(Timeout = 120000)]
    public async Task LatentDiffusion_NullConditioner_StreamsLiveComponentTensors()
    {
        await Task.Yield();

        using var source = CreateSmallStableDiffusion(seed: 42);
        using var destination = CreateSmallStableDiffusion(seed: 777);

        Assert.Null(source.Conditioner);

        var flat = source.GetParameters();
        var chunks = source.GetParameterChunks().ToList();

        Assert.Equal(source.ParameterCount, (long)flat.Length);
        Assert.Equal(source.ParameterCount, chunks.Sum(chunk => (long)chunk.Length));
        Assert.True(chunks.Count > 2,
            "The U-Net and VAE must remain per-tensor streams, not two component-sized allocations.");
        Assert.All(chunks, chunk => Assert.True(chunk.Length < flat.Length));

        int offset = 0;
        foreach (var chunk in chunks)
        {
            for (int i = 0; i < chunk.Length; i++)
                Assert.Equal(flat[offset++], chunk[i]);
        }
        Assert.Equal(flat.Length, offset);

        long parameterCount = source.ParameterCount;
        _ = source.GetParameterChunks().Sum(chunk => (long)chunk.Length); // warm iterator paths
#if !NETFRAMEWORK
        var predictorAllocations = MeasureChunkEnumerationAllocations(
            () => source.NoisePredictor.GetParameterChunks());
        var vaeAllocations = MeasureChunkEnumerationAllocations(
            () => source.VAE.GetParameterChunks());
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
#endif
        long streamedCount = source.GetParameterChunks().Sum(chunk => (long)chunk.Length);
#if !NETFRAMEWORK
        long streamingAllocations = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
#endif
        Assert.Equal(parameterCount, streamedCount);
#if !NETFRAMEWORK
        Assert.True(streamingAllocations < Math.Max(1_000_000L, parameterCount),
            $"Streaming allocated {streamingAllocations:N0} bytes for {parameterCount:N0} parameters; " +
            $"predictor={predictorAllocations.Total:N0} across {predictorAllocations.ChunkCount} chunks " +
            $"(largest MoveNext={predictorAllocations.LargestMoveNext:N0} bytes at chunk " +
            $"{predictorAllocations.LargestMoveNextIndex}, length " +
            $"{predictorAllocations.LargestMoveNextChunkLength:N0}); " +
            $"VAE={vaeAllocations.Total:N0}. " +
            "That is consistent with rebuilding payloads, not zero-copy per-tensor streaming.");
#endif

        destination.SetParameterChunks(source.GetParameterChunks());
        var restored = destination.GetParameters();
        Assert.Equal(flat.Length, restored.Length);
        for (int i = 0; i < flat.Length; i++)
            Assert.Equal(flat[i], restored[i]);
    }

    [Fact]
    public void BiasRedundancy_IsOptionalAndMatchesNormalizationMathematics()
    {
        Assert.DoesNotContain(typeof(ILayer<double>).GetProperties(),
            property => property.Name == "ProvidesLearnableShift");
        Assert.DoesNotContain(typeof(ILayer<double>).GetProperties(),
            property => property.Name == "AbsorbsUpstreamChannelBias");

        Assert.True(((IUpstreamBiasRedundancy)new BatchNormalizationLayer<double>())
            .MakesUpstreamBiasRedundant);
        Assert.True(((IUpstreamBiasRedundancy)new GroupNormalizationLayer<double>(8, 8))
            .MakesUpstreamBiasRedundant);
        Assert.False(((IUpstreamBiasRedundancy)new GroupNormalizationLayer<double>(4, 8))
            .MakesUpstreamBiasRedundant);
        Assert.True(((IUpstreamBiasRedundancy)new InstanceNormalizationLayer<double>(8, affine: true))
            .MakesUpstreamBiasRedundant);
        Assert.True(((IUpstreamBiasRedundancy)new InstanceNormalizationLayer<double>(8, affine: false))
            .MakesUpstreamBiasRedundant);
        Assert.False(((IUpstreamBiasRedundancy)new LayerNormalizationLayer<double>())
            .MakesUpstreamBiasRedundant);
    }

    [Fact]
    public void Convolution_BiasModeNever_RoundTripsAndReturnsNoBiasTensor()
    {
        var input = new Tensor<double>([1, 3, 8, 8]);
        var layer = new ConvolutionalLayer<double>(
            outputDepth: 2,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            biasMode: BiasMode.Never);

        layer.Forward(input);
        Assert.Null(layer.GetBiases());
        Assert.Equal("Never", layer.GetMetadata()["BiasMode"]);

        var restored = Assert.IsType<ConvolutionalLayer<double>>(
            DeserializationHelper.CreateLayerFromType<double>(
                "ConvolutionalLayer`1", [1, 3, 8, 8], [2], ToObjectMetadata(layer.GetMetadata())));
        restored.Forward(input);

        Assert.Null(restored.GetBiases());
        Assert.Equal(layer.ParameterCount, restored.ParameterCount);
        Assert.Contains(typeof(ConvolutionalLayer<double>).GetConstructors(), constructor =>
        {
            var parameters = constructor.GetParameters();
            return parameters.Length == 8
                && parameters.Take(4).All(parameter => parameter.ParameterType == typeof(int))
                && parameters[4].ParameterType == typeof(IActivationFunction<double>)
                && parameters[5].ParameterType == typeof(IInitializationStrategy<double>)
                && parameters[6].ParameterType == typeof(IActivationFunction<double>)
                && parameters[7].ParameterType == typeof(int);
        });
    }

    [Fact]
    public void PatchGAN_DefaultPreservesLegacyLayout_AndAutoRoundTrips()
    {
        var input = new Tensor<double>([1, 3, 32, 32]);
        var legacyDefault = new PatchGANDiscriminator<double>(
            numLayers: 2, numFilters: 8, kernelSize: 3, leakySlope: 0.1, applySigmoid: false);
        var explicitAlways = new PatchGANDiscriminator<double>(
            numLayers: 2, numFilters: 8, kernelSize: 3, leakySlope: 0.1,
            applySigmoid: false, biasMode: BiasMode.Always);
        var auto = new PatchGANDiscriminator<double>(
            numLayers: 2, numFilters: 8, kernelSize: 3, leakySlope: 0.1,
            applySigmoid: false, biasMode: BiasMode.Auto);

        legacyDefault.Forward(input);
        explicitAlways.Forward(input);
        auto.Forward(input);

        Assert.Equal(explicitAlways.ParameterCount, legacyDefault.ParameterCount);
        Assert.True(auto.ParameterCount < legacyDefault.ParameterCount);
        Assert.Equal("Auto", auto.GetMetadata()["BiasMode"]);

        var restored = Assert.IsType<PatchGANDiscriminator<double>>(
            DeserializationHelper.CreateLayerFromType<double>(
                "PatchGANDiscriminator`1", [1, 3, 32, 32], [1], ToObjectMetadata(auto.GetMetadata())));
        restored.Forward(input);

        Assert.Equal(auto.ParameterCount, restored.ParameterCount);
        Assert.Equal("Auto", restored.GetMetadata()["BiasMode"]);
        Assert.Equal(auto.GetMetadata()["LeakySlope"], restored.GetMetadata()["LeakySlope"]);
        Assert.Contains(typeof(PatchGANDiscriminator<double>).GetConstructors(), constructor =>
            constructor.GetParameters().Length == 5
            && constructor.GetParameters()[0].ParameterType == typeof(int));
    }

    [Fact]
    public void DcnV3_UsesGroupwiseSoftmax_AndRoundTripsProjectionMode()
    {
        var input = new Tensor<double>([1, 4, 4, 4]);
        var layer = new DeformableConvolutionalLayer<double>(
            outputChannels: 4,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            groups: 2,
            deformGroups: 2,
            useModulation: true,
            separableOffsetProjection: true);

        layer.Forward(input);

        var maskField = typeof(DeformableConvolutionalLayer<double>)
            .GetField("_lastMask", BindingFlags.Instance | BindingFlags.NonPublic);
        var mask = Assert.IsType<Tensor<double>>(maskField?.GetValue(layer));
        Assert.Equal(4, mask.Shape.Length);
        Assert.Equal(1, mask.Shape[0]);
        Assert.Equal(18, mask.Shape[1]);
        Assert.Equal(4, mask.Shape[2]);
        Assert.Equal(4, mask.Shape[3]);

        const int kernelPoints = 9;
        int height = mask.Shape[2];
        int width = mask.Shape[3];
        for (int group = 0; group < 2; group++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double sum = 0;
                    for (int point = 0; point < kernelPoints; point++)
                    {
                        int channel = group * kernelPoints + point;
                        int flatIndex = (channel * height + y) * width + x;
                        Assert.True(mask[flatIndex] > 0.0);
                        sum += mask[flatIndex];
                    }
                    Assert.InRange(sum, 1.0 - 1e-10, 1.0 + 1e-10);
                }
            }
        }

        Assert.Equal("True", layer.GetMetadata()["SeparableOffsetProjection"]);
        var restored = Assert.IsType<DeformableConvolutionalLayer<double>>(
            DeserializationHelper.CreateLayerFromType<double>(
                "DeformableConvolutionalLayer`1", [1, 4, 4, 4], [4],
                ToObjectMetadata(layer.GetMetadata())));
        restored.Forward(input);
        Assert.Equal(layer.ParameterCount, restored.ParameterCount);
        Assert.Equal("True", restored.GetMetadata()["SeparableOffsetProjection"]);

        Assert.Throws<ArgumentException>(() => new DeformableConvolutionalLayer<double>(
            outputChannels: 4, kernelSize: 2, padding: 0, groups: 1, deformGroups: 1,
            useModulation: true, separableOffsetProjection: true));
        _ = new DeformableConvolutionalLayer<double>(
            outputChannels: 4, kernelSize: 2, padding: 0, groups: 1, deformGroups: 1,
            useModulation: true, separableOffsetProjection: false);

        Assert.Contains(typeof(DeformableConvolutionalLayer<double>).GetConstructors(), constructor =>
        {
            var parameters = constructor.GetParameters();
            return parameters.Length == 8
                && parameters.Take(6).All(parameter => parameter.ParameterType == typeof(int))
                && parameters[6].ParameterType == typeof(bool)
                && parameters[7].Name == "engine";
        });
    }

    private static StableDiffusion2Model<float> CreateSmallStableDiffusion(int seed)
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 32,
            channelMultipliers: [1, 2],
            numResBlocks: 1,
            attentionResolutions: [1],
            contextDim: 0,
            numHeads: 4,
            inputHeight: 8,
            seed: seed);

        var vae = new StandardVAE<float>(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 32,
            channelMultipliers: [1, 2],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 0.18215,
            seed: seed);

        return new StableDiffusion2Model<float>(unet: unet, vae: vae, seed: seed);
    }

    private static Dictionary<string, object> ToObjectMetadata(Dictionary<string, string> metadata)
        => metadata.ToDictionary(entry => entry.Key, entry => (object)entry.Value);

#if !NETFRAMEWORK
    private static (long Total, int ChunkCount, long LargestMoveNext,
        int LargestMoveNextIndex, int LargestMoveNextChunkLength)
        MeasureChunkEnumerationAllocations(Func<IEnumerable<Tensor<float>>> chunks)
    {
        _ = chunks().Sum(chunk => (long)chunk.Length);
        long start = GC.GetAllocatedBytesForCurrentThread();
        using var enumerator = chunks().GetEnumerator();
        int index = 0;
        long largest = 0;
        int largestIndex = -1;
        int largestChunkLength = 0;
        while (true)
        {
            long beforeMoveNext = GC.GetAllocatedBytesForCurrentThread();
            if (!enumerator.MoveNext()) break;
            long allocated = GC.GetAllocatedBytesForCurrentThread() - beforeMoveNext;
            if (allocated > largest)
            {
                largest = allocated;
                largestIndex = index;
                largestChunkLength = enumerator.Current.Length;
            }
            index++;
        }
        return (GC.GetAllocatedBytesForCurrentThread() - start, index, largest,
            largestIndex, largestChunkLength);
    }
#endif
}
