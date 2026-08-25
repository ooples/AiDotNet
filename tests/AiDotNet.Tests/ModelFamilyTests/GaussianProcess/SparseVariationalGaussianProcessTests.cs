using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.Fixtures;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

[Collection(ConvergenceSensitiveCollection.Name)]
public class SparseVariationalGaussianProcessTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new SparseVariationalGaussianProcess<double>(new GaussianKernel<double>());
}
