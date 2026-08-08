using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// Model-family coverage for <see cref="InstantStyleModel{T}"/> (arXiv:2404.02733).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this fixture is smaller than the model's defaults.</b> InstantStyle's production defaults
/// are SDXL-scale — a 320-channel UNet over a 64x64 latent (512x512 images through an 8x VAE) with a
/// 2048-wide cross-attention context. A training step at that size exceeds both the per-test time
/// budget and the 16 GB CI runner's memory (#1706/#1305). The fixture therefore constructs the SAME
/// model with a smaller predictor and VAE over an 8x8 latent: identical topology, identical code
/// path, identical injection mechanism — only the widths and the spatial size are reduced.
/// </para>
/// <para>
/// Nothing about the test's rigour is reduced with it. The base class's full training-iteration
/// count runs unchanged, and the paper's actual mechanisms — block-specific injection and the
/// style/content decoupling subtraction — are asserted separately and at no reduced fidelity in
/// <c>InstantStyleMechanismTests</c>, which do not depend on model scale.
/// </para>
/// </remarks>
public class InstantStyleModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override IDiffusionModel<float> CreateModel()
        => new InstantStyleModel<float>(
            predictor: new UNetNoisePredictor<float>(
                architecture: null, inputChannels: 4, outputChannels: 4,
                baseChannels: 32, channelMultipliers: [1, 2],
                numResBlocks: 1, attentionResolutions: [2], contextDim: 64, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4,
                baseChannels: 16, channelMultipliers: [1, 2],
                numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
