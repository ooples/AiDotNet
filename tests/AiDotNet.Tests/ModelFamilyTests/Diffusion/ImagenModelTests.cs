using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// Imagen test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/> for
/// the FP32 rationale. Imagen at paper defaults OOMs in fresh-process probes
/// at FP64 on the 16 GB CI host.
/// </summary>
public class ImagenModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new ImagenModel<float>(seed: 42);
}
