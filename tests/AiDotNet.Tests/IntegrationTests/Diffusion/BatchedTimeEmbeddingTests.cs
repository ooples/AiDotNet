using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// The fused diffusion training step feeds timesteps as a tensor. Each predictor that declares
/// <see cref="NoisePredictorBase{T}.SupportsBatchedTimeEmbedding"/> must, for a batch whose elements sit at different
/// timesteps, return exactly what it returns for each element alone at its own timestep.
/// </summary>
public class BatchedTimeEmbeddingTests
{
    public static TheoryData<string> Predictors => new()
    {
        "UNet", "DiT", "UViT", "FlagDiT", "VideoUNet",
    };

    private static (NoisePredictorBase<double> Predictor, int[] SampleShape) Create(string name) => name switch
    {
        "UNet" => (new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
            channelMultipliers: [1, 2], numResBlocks: 1, attentionResolutions: [1], contextDim: 0, numHeads: 2,
            inputHeight: 8, seed: 42), [4, 8, 8]),
        "DiT" => (new DiTNoisePredictor<double>(inputChannels: 4, hiddenSize: 16, numLayers: 1, numHeads: 2,
            patchSize: 2, contextDim: 16, latentSpatialSize: 8, seed: 42), [4, 8, 8]),
        "UViT" => (new UViTNoisePredictor<double>(inputChannels: 4, hiddenSize: 16, numLayers: 2, numHeads: 2,
            patchSize: 2, contextDim: 0, latentSpatialSize: 8, seed: 42), [4, 8, 8]),
        "FlagDiT" => (new FlagDiTPredictor<double>(inputChannels: 4, hiddenSize: 32, numLayers: 1, numHeads: 4,
            numKVHeads: 2, contextDim: 32, latentSize: 8, seed: 42), [4, 8, 8]),
        "VideoUNet" => (new VideoUNetPredictor<double>(inputChannels: 2, outputChannels: 2, baseChannels: 8,
            channelMultipliers: new[] { 1 }, numResBlocks: 1, attentionResolutions: new int[0], numTemporalLayers: 1,
            contextDim: 0, numHeads: 1, supportsImageConditioning: false, inputHeight: 4, inputWidth: 4,
            numFrames: 3, clipTokenLength: 1), [2, 3, 4, 4]),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [Theory]
    [MemberData(nameof(Predictors))]
    public void BatchedEmbedding_EqualsEachElementAtItsOwnTimestep(string name)
    {
        var (predictor, sampleShape) = Create(name);
        int[] timesteps = [3, 517, 999];
        int batch = timesteps.Length;

        var batchShape = new int[sampleShape.Length + 1];
        batchShape[0] = batch;
        Array.Copy(sampleShape, 0, batchShape, 1, sampleShape.Length);
        var rng = new Random(7);
        var x = new Tensor<double>(batchShape);
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextDouble() * 2 - 1;

        var t = new Tensor<double>([batch]);
        for (int b = 0; b < batch; b++) t[b] = timesteps[b];
        var batched = predictor.PredictNoiseWithEmbedding(x, predictor.GetTimestepEmbedding(t));

        int per = x.Length / batch;
        var elemShape = (int[])batchShape.Clone();
        elemShape[0] = 1;
        double worst = 0.0;
        for (int b = 0; b < batch; b++)
        {
            var elem = new Tensor<double>(elemShape);
            for (int j = 0; j < per; j++) elem[j] = x[b * per + j];
            var alone = predictor.PredictNoise(elem, timesteps[b]);
            for (int j = 0; j < per; j++) worst = Math.Max(worst, Math.Abs(alone[j] - batched[b * per + j]));
        }

        Assert.True(worst <= 1e-9, $"{name}: batched-embedding prediction differs from per-element by {worst:G6}");
        Assert.True(predictor.SupportsBatchedTimeEmbedding, $"{name} matches but does not declare SupportsBatchedTimeEmbedding.");
    }

    [Fact]
    public void TensorEmbedding_EqualsIntEmbedding()
    {
        var (predictor, _) = Create("DiT");
        int[] timesteps = [0, 1, 250, 999];
        var t = new Tensor<double>([timesteps.Length]);
        for (int b = 0; b < timesteps.Length; b++) t[b] = timesteps[b];
        var rows = predictor.GetTimestepEmbedding(t);
        for (int b = 0; b < timesteps.Length; b++)
        {
            var single = predictor.GetTimestepEmbedding(timesteps[b]);
            for (int j = 0; j < single.Length; j++)
                Assert.True(Math.Abs(single[j] - rows[b * single.Length + j]) <= 1e-12, $"t={timesteps[b]} j={j}");
        }
    }
}