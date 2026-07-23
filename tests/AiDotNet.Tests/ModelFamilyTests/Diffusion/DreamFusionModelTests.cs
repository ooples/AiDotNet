using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class DreamFusionModelTests : DiffusionModelTestBase<float>
{
    // DreamFusion operates in the 4-channel Stable-Diffusion VAE LATENT space
    // (DreamFusionModel.DREAM_LATENT_CHANNELS = 4, matching the SD 1.5 VAE used as
    // its 2D diffusion prior — see the model's documented Predict example, a
    // [1,4,8,8] latent). Its internal U-Net, VAE (3->4 latent) and SDS path all run
    // at 4 latent channels, so Predict/Train consume a 4-channel latent and the
    // noise-prediction tensor is 4-channel by construction. A 3-channel input here
    // was paper-incorrect: the 4-channel U-Net emits [1,4,H,W] while the noise vector
    // is sized to the 3-channel input, tripping "number of values does not match the
    // specified shape" in DiffusionModelBase.Train, and OutputShape (latent in ==
    // latent out) fails 3072 vs 4096. The prior's U-Net must match (4 channels).
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
    {
        // DreamFusion uses a pretrained diffusion model as prior — 4-channel SD latent.
        var prior = new DDPMModel<float>(
            unet: new UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4,
                baseChannels: 64, channelMultipliers: [1, 2, 4],
                numResBlocks: 1, attentionResolutions: [1, 2],
                contextDim: 0, numHeads: 4, inputHeight: 32, seed: 42),
            seed: 42);

        return new DreamFusionModel<float>(diffusionPrior: prior, seed: 42);
    }
}
