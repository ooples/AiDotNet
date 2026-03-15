using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

public class StandardGaussianProcessTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new StandardGaussianProcess<double>(new GaussianKernel<double>());
}
