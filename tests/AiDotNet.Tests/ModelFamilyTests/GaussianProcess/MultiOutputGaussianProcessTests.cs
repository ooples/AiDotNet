using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

public class MultiOutputGaussianProcessTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new MultiOutputGaussianProcess<double>(new GaussianKernel<double>());
}
