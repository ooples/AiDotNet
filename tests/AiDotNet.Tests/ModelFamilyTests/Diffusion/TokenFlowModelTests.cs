using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TokenFlowModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // TokenFlow defaults to a foundation-scale video UNet + temporal VAE that exceeds the 120s
    // model-family budget. Inject a tiny same-architecture VideoUNet + TemporalVAE — latentChannels
    // (4) and contextDim (768) stay paper-correct; only base channels / level count / res-blocks shrink.
    protected override IDiffusionModel<float> CreateModel()
        => new TokenFlowModel<float>(
            predictor: new VideoUNetPredictor<float>(
                inputChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2 },
                numResBlocks: 1, numHeads: 8, contextDim: 768,
                inputHeight: 16, inputWidth: 16, numFrames: 16, seed: 42),
            temporalVAE: new TemporalVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numTemporalLayers: 1,
                temporalKernelSize: 3, latentScaleFactor: 0.18215, seed: 42));
}
