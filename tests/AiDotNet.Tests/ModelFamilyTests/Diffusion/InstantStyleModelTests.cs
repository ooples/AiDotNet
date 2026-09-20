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
    // This fixture used to override CloneOutputRelativeTolerance to 1.5e-5, on the same grounds as
    // its sibling StyDiffModelTests. Removed on the condition both overrides set: that the clone
    // difference holds at zero once the bit-identity check no longer runs BETWEEN the two forwards
    // (see DiffusionModelTestBase).
    //
    // Validation status. Clone_ShouldProduceIdenticalOutput passed in 133 ms on Linux CI with the
    // default tolerance restored -- shard "ModelFamily - Diffusion D-I", 247/247 -- and passes on
    // Windows/x64 and on AVX2 Linux. That is one green CI observation, not a proof the divergence
    // is gone: the ISA of that runner was not recorded, because the numeric-environment reading
    // only printed on failure. It is emitted unconditionally from here on, so the next run will
    // say.
    //
    // Unlike StyDiff's, this override was load-bearing: Linux CI observed 1.86e-5 against a
    // 1.60e-5 allowance on an output of magnitude 4.63 -- about 4e-6 relative, roughly 39 float
    // ulp -- so at that magnitude the widened relative term dominated the bound. If it returns,
    // that is the measurement to reproduce, and the history is in StyDiffModelTests: the "cold
    // versus warm packed-weight path" story is contradicted by Predict_ShouldBeDeterministic,
    // which makes exactly that cold-then-warm comparison on one instance and matches to 12
    // decimals, and the divergence has never reproduced on AVX2 hardware.

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
