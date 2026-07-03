using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class LuminaImage2ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 16, 64, 64];
    protected override int[] OutputShape => [1, 16, 64, 64];

    // Build the model at a REDUCED scale (small Flag-DiT + small VAE) instead of the default
    // foundation-scale config (hiddenSize 4096 x 32 layers ~ tens of GB in float32, which OOMs the 16 GB
    // CI runner). Shape-critical dims preserved to match the exercised latent tensor: inputChannels =
    // LUMINA_LATENT_CHANNELS (16), contextDim 4096, latentSize 64 — so the patchify path is exercised on
    // the same 64x64 latent grid used by InputShape/OutputShape. Only width/depth (and the VAE) shrink so
    // the test stays exact, fast, and in the PR gate.
    protected override IDiffusionModel<float> CreateModel()
        => new LuminaImage2Model<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.FlagDiTPredictor<float>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2,
                numKVHeads: 1, contextDim: 4096, latentSize: 64, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
