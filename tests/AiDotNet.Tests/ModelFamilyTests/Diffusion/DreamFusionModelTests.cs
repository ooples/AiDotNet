using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DreamFusionModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1, 3, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
    {
        // DreamFusion uses a pretrained diffusion model as prior
        var prior = new DDPMModel<double>(
            unet: new UNetNoisePredictor<double>(
                inputChannels: 3, outputChannels: 3,
                baseChannels: 64, channelMultipliers: [1, 2, 4],
                numResBlocks: 1, attentionResolutions: [1, 2],
                contextDim: 0, numHeads: 4, inputHeight: 32, seed: 42),
            seed: 42);

        return new DreamFusionModel<double>(diffusionPrior: prior, seed: 42);
    }
}
