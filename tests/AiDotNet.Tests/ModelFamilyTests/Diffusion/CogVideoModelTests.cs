using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class CogVideoModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 16, 16, 16];
    protected override int[] OutputShape => [1, 16, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
        => new CogVideoModel<double>(seed: 42);
}
