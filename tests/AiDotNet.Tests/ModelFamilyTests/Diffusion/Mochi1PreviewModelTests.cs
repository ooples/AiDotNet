using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class Mochi1PreviewModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 12, 32, 32];
    protected override int[] OutputShape => [1, 12, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new Mochi1PreviewModel<double>();
}
