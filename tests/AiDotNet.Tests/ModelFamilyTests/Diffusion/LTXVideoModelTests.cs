using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class LTXVideoModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 128, 32, 32];
    protected override int[] OutputShape => [1, 128, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new LTXVideoModel<double>(seed: 42);
}
