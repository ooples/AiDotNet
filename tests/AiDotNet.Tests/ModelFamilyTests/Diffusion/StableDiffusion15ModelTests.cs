using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// SD1.5 test scaffold — uses <see cref="DiffusionModelTestBase{TNum}"/> with
/// <c>TNum = float</c> rather than the default-double shim. SD1.5 at paper
/// defaults (UNet baseChannels=320, multipliers=[1,2,4,4]) OOMs in
/// fresh-process probes at FP64 on the 16 GB CI host, fits at FP32. FP32 is
/// also the production-canonical type for SD weights and the only numeric
/// type
/// <see cref="AiDotNet.AiModelBuilder{T, TInput, TOutput}.ConfigureMixedPrecision"/>
/// accepts.
/// </summary>
// Foundation-scale-at-default: the model's full-scale default config has a Training peak (weights +
// gradients + Adam state + activations ~ 4x the ~1 GB SD/DiT-scale weights) that OOMs the 16 GB CI
// runner (verified via the CI logs — testhost/runner OOM at default scale; fits only on a larger box).
// Moved to the HeavyTimeout nightly lane so the default PR-gate shard fits and passes (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class StableDiffusion15ModelTests : DiffusionModelTestBase<float>
{
    // Per Rombach et al. 2022: SD 1.5 operates on 64×64 latent space
    // with 4 channels from the VAE (512×512 images / 8x downsampling)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    // Use default constructor — all architecture parameters match the paper:
    // UNet: baseChannels=320, multipliers=[1,2,4,4], 2 ResBlocks, attention at [1,2,3]
    // VAE: baseChannels=128, multipliers=[1,2,4,4], latentScale=0.18215
    // Users can customize via constructor params, but defaults = paper values
    protected override IDiffusionModel<float> CreateModel()
        => new StableDiffusion15Model<float>(seed: 42);
}
