using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// DeepFloydIF test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. DeepFloyd IF at paper defaults OOMs in fresh-process
/// probes at FP64 on the 16 GB CI host.
/// </summary>
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class DeepFloydIFModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 64, 64];
    protected override int[] OutputShape => [1, 4, 64, 64];

    protected override IDiffusionModel<float> CreateModel()
        => new DeepFloydIFModel<float>(seed: 42);
}
