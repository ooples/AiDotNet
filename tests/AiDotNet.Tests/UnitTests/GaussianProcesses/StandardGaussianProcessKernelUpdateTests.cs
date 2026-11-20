using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.GaussianProcesses;

public sealed class StandardGaussianProcessKernelUpdateTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void KernelUpdate_BeforeOrAfterFit_MatchesFreshUpdatedModel(bool fitFirst)
    {
        var features = new Matrix<double>(3, 1);
        features[0, 0] = 0.0;
        features[1, 0] = 1.0;
        features[2, 0] = 2.0;
        var targets = new Vector<double>(new[] { 0.5, -1.0, 2.0 });
        var model = new StandardGaussianProcess<double>(new GaussianKernel<double>(sigma: 0.5));
        if (fitFirst) model.Fit(features, targets);

        // Before fitting, a kernel update must not try to factor a zero-sized covariance.
        // After fitting, changing sigma changes the off-diagonal covariance and requires a refit.
        var updatedKernel = new GaussianKernel<double>(sigma: 2.0);
        model.UpdateKernel(updatedKernel);
        if (!fitFirst) model.Fit(features, targets);

        var reference = new StandardGaussianProcess<double>(updatedKernel);
        reference.Fit(features, targets);
        var query = new Vector<double>(new[] { 0.75 });

        Assert.Equal(reference.Predict(query), model.Predict(query));
    }
}
