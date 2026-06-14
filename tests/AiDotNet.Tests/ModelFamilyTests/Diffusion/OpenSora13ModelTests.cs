using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class OpenSora13ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new OpenSora13Model<float>(seed: 42);
}
