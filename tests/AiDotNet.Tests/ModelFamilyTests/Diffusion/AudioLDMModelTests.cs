using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class AudioLDMModelTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new AudioLDMModel<double>();
}
