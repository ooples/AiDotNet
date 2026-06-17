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
public class StableCascadeModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 24, 64, 64];
    protected override int[] OutputShape => [1, 24, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new StableCascadeModel<float>(seed: 42);
}
