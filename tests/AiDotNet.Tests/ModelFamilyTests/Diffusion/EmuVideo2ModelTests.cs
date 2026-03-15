using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.AudioVisual;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class EmuVideo2ModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new EmuVideo2Model<double>();
}
