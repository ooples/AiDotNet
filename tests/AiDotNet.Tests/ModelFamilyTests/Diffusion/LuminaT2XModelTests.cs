using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class LuminaT2XModelTests : DiffusionModelTestBase<float>
{
    // LuminaT2X uses a 4-channel latent (T2X_LATENT_CHANNELS=4) and its Flag-DiT predictor is built
    // for a 32x32 latent (latentSize: 32 -> 256 patches at patch size 2). Input/output are the
    // 4-channel latent; the predictor patchifies/unpatchifies back to the same shape.
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    // Build the model at a REDUCED scale (small Flag-DiT + small VAE) instead of the default
    // foundation-scale config (hiddenSize 4096 x 32 layers = multi-billion params ~ tens of GB in
    // float32, which OOMs the 16 GB CI runner). The shape-critical dims are preserved — inputChannels =
    // T2X_LATENT_CHANNELS (4), contextDim = T2X_CONTEXT_DIM (2048), latentSize = 32 — so the forward/
    // patchify path is exercised identically; only the width/depth (and the VAE) are shrunk so the test
    // stays exact, fast, and in the default PR gate. Mirrors LatteModelTests/LGMModelTests.
    protected override IDiffusionModel<float> CreateModel()
        => new LuminaT2XModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.FlagDiTPredictor<float>(
                inputChannels: 4, hiddenSize: 64, numLayers: 2, numHeads: 2,
                numKVHeads: 1, contextDim: 2048, latentSize: 32, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
