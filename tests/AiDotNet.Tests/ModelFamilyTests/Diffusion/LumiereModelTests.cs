using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class LumiereModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // Lumiere defaults to a 320-base-channel video UNet ([1,2,4,4]) + temporal VAE: a single forward
    // exceeds the 120s model-family budget. Inject a tiny same-architecture VideoUNet + TemporalVAE —
    // latentChannels (4) and contextDim (768) stay paper-correct; only base channels / level count shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new LumiereModel<float>(
            predictor: new VideoUNetPredictor<float>(
                inputChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2 },
                numResBlocks: 1, numHeads: 8, contextDim: 768,
                inputHeight: 16, inputWidth: 16, numFrames: 16, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1,
                temporalKernelSize: 3, latentScaleFactor: 0.18215, seed: 42));
}
