using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

public class SparseGaussianProcessTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new SparseGaussianProcess<double>(new GaussianKernel<double>());
}
