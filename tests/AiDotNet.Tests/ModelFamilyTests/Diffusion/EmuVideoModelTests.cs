using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.AudioVisual;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class EmuVideoModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new EmuVideoModel<double>();
}
