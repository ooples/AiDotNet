using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.AudioVisual;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class EmuVideo2ModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    protected override IDiffusionModel<double> CreateModel()
        => new EmuVideo2Model<double>(seed: 42);
}
