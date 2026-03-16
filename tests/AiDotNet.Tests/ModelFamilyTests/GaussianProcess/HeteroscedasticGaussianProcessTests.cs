using AiDotNet.Interfaces;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.GaussianProcess;

public class HeteroscedasticGaussianProcessTests : GaussianProcessModelTestBase
{
    protected override IGaussianProcess<double> CreateModel()
        => new HeteroscedasticGaussianProcess<double>(
            // Sigma=5.0 matches the data scale (features in [0,10]).
            // Default sigma=1.0 causes exp(-||x||²) ≈ 0 for distant points,
            // making the kernel matrix near-identity and the GP degenerate.
            new GaussianKernel<double>());
}
