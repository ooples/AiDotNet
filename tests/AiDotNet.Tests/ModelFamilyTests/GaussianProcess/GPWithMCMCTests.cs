using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

public class GPWithMCMCTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new GPWithMCMC<double>(new GaussianKernel<double>());
}
