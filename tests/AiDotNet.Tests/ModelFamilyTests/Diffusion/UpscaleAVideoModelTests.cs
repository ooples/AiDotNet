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
    protected override int[] InputShape => [1, 1, 3, 4, 4];
    protected override int[] OutputShape => [1, 1, 3, 16, 16];
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
            input,
            EnsureFourTimesTarget(input, expectedOutput),
            "test clip",
            noiseLevel: 20);
    }

    private static Tensor<float> EnsureFourTimesTarget(
        Tensor<float> input,
        Tensor<float> expectedOutput)
    {
        int targetHeight = input.Shape[3] * 4;
        int targetWidth = input.Shape[4] * 4;
        if (expectedOutput.Rank == 5 &&
            expectedOutput.Shape[0] == input.Shape[0] &&
            expectedOutput.Shape[1] == input.Shape[1] &&
            expectedOutput.Shape[2] == input.Shape[2] &&
            expectedOutput.Shape[3] == targetHeight &&
            expectedOutput.Shape[4] == targetWidth)
        {
            return expectedOutput;
        }

        var target = new Tensor<float>(
            [input.Shape[0], input.Shape[1], input.Shape[2], targetHeight, targetWidth]);
        for (int batch = 0; batch < input.Shape[0]; batch++)
        for (int frame = 0; frame < input.Shape[1]; frame++)
        for (int channel = 0; channel < input.Shape[2]; channel++)
        for (int y = 0; y < targetHeight; y++)
        for (int x = 0; x < targetWidth; x++)
            target[batch, frame, channel, y, x] = input[batch, frame, channel, y / 4, x / 4];
        return target;
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
            => Sequential(tokenIds.Shape[0] * 1024, 0.01f,
                [tokenIds.Shape[0], 1, 1024]);

        public Tensor<float> GetPooledEmbedding(Tensor<float> sequenceEmbeddings)
            => Sequential(sequenceEmbeddings.Shape[0] * 1024, 0.02f,
                [sequenceEmbeddings.Shape[0], 1024]);

        public Tensor<float> GetUnconditionalEmbedding(int batchSize)
            => Sequential(batchSize * 1024, -0.01f, [batchSize, 1, 1024]);

        private static Tensor<float> Sequential(int length, float step, int[] shape)
        {
            var data = new float[length];
            for (int i = 0; i < length; i++) data[i] = (i + 1) * step;
            return new Tensor<float>(data, shape);
        }

        public Tensor<float> Tokenize(string text) => new(new float[1], [1, 1]);

        public Tensor<float> TokenizeBatch(string[] texts)
            => new(new float[texts.Length], [texts.Length, 1]);
    }
}
