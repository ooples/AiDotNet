using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.LongVideo;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class StreamingT2VModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<double> CreateModel()
        => new StreamingT2VModel<double>(seed: 42);
}
