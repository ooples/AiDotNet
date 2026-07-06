using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
// HeavyTimeout (#1706): full default (paper) scale — single-model peak exceeds the 16 GB PR runner,
// which SIGTERM-cancelled the SA-SD shard. Runs in the nightly lane (gate filters Category!=HeavyTimeout).
[Xunit.Trait("Category", "HeavyTimeout")]
public class SCottModelTests : DiffusionModelTestBase<float>
{
    // SD-based latent diffusion: 4 channels, 64x64 latent (512x512 images / 8x VAE)
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new SCottModel<float>(seed: 42);
}
