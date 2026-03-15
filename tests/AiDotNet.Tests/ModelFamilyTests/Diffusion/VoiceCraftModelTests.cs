using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class VoiceCraftModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new VoiceCraftModel<double>();
}
