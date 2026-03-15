using AiDotNet.Interfaces;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class TrainingEfficientLCMTests : DiffusionModelTestBase
{
    protected override IDiffusionModel<double> CreateModel()
        => new TrainingEfficientLCM<double>();
}
