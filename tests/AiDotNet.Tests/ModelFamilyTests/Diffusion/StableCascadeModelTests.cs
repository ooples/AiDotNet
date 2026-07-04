using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// StableCascade test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. <c>StableCascadeModel&lt;double&gt;</c> OOMs in
/// fresh-process probes (24-channel 64×64 latent through a multi-stage cascade
/// at paper defaults is too large for the 16 GB CI host at FP64).
/// </summary>
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class StableCascadeModelTests : DiffusionModelTestBase<float>
{
    // 16x16 latent (not the paper's 64x64): the reduced spatial keeps the prior U-Net's self-attention
    // at 256 tokens instead of 4096 (~16x cheaper per step), so Training_ShouldReducePredictionError's
    // multi-iteration train loop finishes inside the 120s gate. The forward path is exercised identically
    // (24-channel latent through the two-stage cascade); only the token count shrinks.
    protected override int[] InputShape => [1, 24, 16, 16];
    protected override int[] OutputShape => [1, 24, 16, 16];

    // Build the two cascade U-Nets + VAE at a REDUCED scale instead of the paper defaults
    // (baseChannels 384/320 x [1,2,4,4] with multi-resolution attention, which peaks well past the
    // 16 GB CI runner during Training and times out). Shape-critical dims are preserved — prior
    // inputChannels = CASCADE_LATENT_CHANNELS (24), decoder inputChannels = CASCADE_STAGE_B_LATENT_CHANNELS
    // (4), contextDim = CASCADE_CROSS_ATTENTION_DIM (1280) — so the two-stage forward is exercised
    // identically; only width/depth (and the VAE) shrink so the test stays exact, fast, and in the PR gate.
    protected override IDiffusionModel<float> CreateModel()
        => new StableCascadeModel<float>(
            priorUnet: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 24, outputChannels: 24, baseChannels: 32,
                channelMultipliers: new[] { 1, 2, 4 }, numResBlocks: 1,
                attentionResolutions: new[] { 1, 2 }, contextDim: 1280, seed: 42),
            decoderUnet: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 32,
                channelMultipliers: new[] { 1, 2, 4 }, numResBlocks: 1,
                attentionResolutions: new[] { 1, 2 }, contextDim: 1280, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 0.3611, seed: 42),
            seed: 42);
}
