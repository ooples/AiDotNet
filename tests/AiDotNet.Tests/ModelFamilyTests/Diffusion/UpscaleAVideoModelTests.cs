using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class UpscaleAVideoModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 4, 4];
    protected override int[] OutputShape => [1, 4, 4, 4];
    protected override int TrainingIterations => 1;

    // Upscale-A-Video defaults to a 320-base-channel video UNet + temporal VAE: a single forward
    // exceeds the 120s model-family budget. Inject a tiny same-architecture VideoUNet + TemporalVAE —
    // latentChannels (4) and contextDim (1024) stay paper-correct; only base channels / level count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new UpscaleAVideoModel<float>(
            videoUNet: new VideoUNetPredictor<float>(
                inputChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2 },
                numResBlocks: 1, numHeads: 8, contextDim: 1024,
                inputHeight: 4, inputWidth: 4, numFrames: 1, clipTokenLength: 1,
                imageConditionChannels: 3, numClassEmbeddings: 351, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 8,
                channelMultipliers: new[] { 1, 2, 4 }, numTemporalLayers: 1,
                temporalKernelSize: 3, latentScaleFactor: 0.18215, seed: 42),
            conditioner: new TestConditioner(),
            seed: 42);

    protected override void TrainModel(
        IDiffusionModel<float> model,
        Tensor<float> input,
        Tensor<float> expectedOutput)
    {
        var upscale = Assert.IsType<UpscaleAVideoModel<float>>(model);
        upscale.TrainConditioned(
            FilledVideo([1, 1, 3, 4, 4], 0.1f),
            FilledVideo([1, 1, 3, 16, 16], 0.2f),
            "test clip",
            noiseLevel: 20);
    }

    private static Tensor<float> FilledVideo(int[] shape, float value)
    {
        int length = shape.Aggregate(1, (product, dimension) => product * dimension);
        return new Tensor<float>(Enumerable.Repeat(value, length).ToArray(), shape);
    }

    private sealed class TestConditioner : IConditioningModule<float>
    {
        public int EmbeddingDimension => 1024;
        public ConditioningType ConditioningType => ConditioningType.Text;
        public bool ProducesPooledOutput => false;
        public int MaxSequenceLength => 1;

        public Tensor<float> Encode(Tensor<float> input) => EncodeText(input);

        public Tensor<float> EncodeText(
            Tensor<float> tokenIds,
            Tensor<float>? attentionMask = null)
            => new(new float[tokenIds.Shape[0] * 1024], [tokenIds.Shape[0], 1, 1024]);

        public Tensor<float> GetPooledEmbedding(Tensor<float> sequenceEmbeddings)
            => new(new float[sequenceEmbeddings.Shape[0] * 1024],
                [sequenceEmbeddings.Shape[0], 1024]);

        public Tensor<float> GetUnconditionalEmbedding(int batchSize)
            => new(new float[batchSize * 1024], [batchSize, 1, 1024]);

        public Tensor<float> Tokenize(string text) => new(new float[1], [1, 1]);

        public Tensor<float> TokenizeBatch(string[] texts)
            => new(new float[texts.Length], [texts.Length, 1]);
    }
}
