using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

public class DeepGaussianProcessTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new DeepGaussianProcess<double>(
            [new GaussianKernel<double>(), new GaussianKernel<double>()],
            [4]);
}
