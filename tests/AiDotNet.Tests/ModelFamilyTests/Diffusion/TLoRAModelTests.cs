using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// Model-family coverage for <see cref="TLoRAModel{T}"/> (arXiv:2507.05964).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this fixture is smaller than the model's defaults.</b> T-LoRA's production defaults are
/// SD-scale — a 320-channel UNet over a 64x64 latent with a 768-wide cross-attention context. That is
/// far past what a per-test budget can construct and train, which is the same reason
/// InstantStyleModelTests reduces its fixture. This constructs the SAME model with a smaller predictor
/// and VAE over an 8x8 latent: identical topology, identical code path, identical timestep-dependent
/// adaptation — only the widths and the spatial size are reduced. Production defaults are untouched.
/// </para>
/// <para>
/// The paper's actual mechanism — the timestep-dependent rank schedule and its orthogonal
/// parametrization — is asserted at full fidelity and independently of model scale in
/// <c>TimestepDependentLoraTests</c>.
/// </para>
/// <para>
/// <b>This class carried [HeavyTimeout] and [FoundationScaleSerial], and it should not have.</b> Both
/// were added for a "training hang" that did not exist. The real cause was that the #55 rebuild of
/// TLoRAModel dropped its required [ModelDomain]/[ModelCategory]/[ModelTask]/[ModelComplexity]/
/// [ModelInput]/[ResearchPaper] attributes, so AIDN001 failed the BUILD; the run never reached the
/// test, and the reported "timed out after 120000 milliseconds" was diagnosed as a slow test instead
/// of a build that never produced an assembly. A stale testhost process holding the output DLLs
/// compounded it. The mistake that let it survive four rounds of investigation was verifying builds
/// with <c>grep -cE "error CS"</c>, which counts zero for an ANALYZER error.
/// </para>
/// <para>
/// With the attributes restored the whole class passes in about five seconds, so the traits and the
/// iteration cap that existed only to accommodate the phantom are gone. The fixture now differs from
/// InstantStyleModelTests only in the model under test, which is what it should always have been.
/// </para>
/// </remarks>
public class TLoRAModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    protected override IDiffusionModel<float> CreateModel()
        => new TLoRAModel<float>(
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
