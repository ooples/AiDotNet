using AiDotNet.Enums;
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// Integration tests for Gaussian Process classes.
/// </summary>
public class GaussianProcessesIntegrationTests
{
    #region StandardGaussianProcess Tests

    [Fact]
    public void StandardGaussianProcess_Construction_WithGaussianKernel_Succeeds()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void StandardGaussianProcess_Construction_WithLinearKernel_Succeeds()
    {
        var kernel = new LinearKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void StandardGaussianProcess_Construction_WithPolynomialKernel_Succeeds()
    {
        var kernel = new PolynomialKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void StandardGaussianProcess_Construction_WithLaplacianKernel_Succeeds()
    {
        var kernel = new LaplacianKernel<double>();
        var gp = new StandardGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void StandardGaussianProcess_Construction_DifferentDecompositionTypes_Succeed()
    {
        var kernel = new GaussianKernel<double>();

        var gpCholesky = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Cholesky);
        var gpLu = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Lu);
        var gpSvd = new StandardGaussianProcess<double>(kernel, MatrixDecompositionType.Svd);

        Assert.NotNull(gpCholesky);
        Assert.NotNull(gpLu);
        Assert.NotNull(gpSvd);
    }

    [Fact]
    public void StandardGaussianProcess_Float_Construction_Succeeds()
    {
        var kernel = new GaussianKernel<float>();
        var gp = new StandardGaussianProcess<float>(kernel);

        Assert.NotNull(gp);
    }

    #endregion

    #region SparseGaussianProcess Tests

    [Fact]
    public void SparseGaussianProcess_Construction_WithGaussianKernel_Succeeds()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new SparseGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void SparseGaussianProcess_Construction_WithLinearKernel_Succeeds()
    {
        var kernel = new LinearKernel<double>();
        var gp = new SparseGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void SparseGaussianProcess_Construction_DifferentDecompositionTypes_Succeed()
    {
        var kernel = new GaussianKernel<double>();

        var gpCholesky = new SparseGaussianProcess<double>(kernel, MatrixDecompositionType.Cholesky);
        var gpLu = new SparseGaussianProcess<double>(kernel, MatrixDecompositionType.Lu);
        var gpSvd = new SparseGaussianProcess<double>(kernel, MatrixDecompositionType.Svd);

        Assert.NotNull(gpCholesky);
        Assert.NotNull(gpLu);
        Assert.NotNull(gpSvd);
    }

    [Fact]
    public void SparseGaussianProcess_Float_Construction_Succeeds()
    {
        var kernel = new GaussianKernel<float>();
        var gp = new SparseGaussianProcess<float>(kernel);

        Assert.NotNull(gp);
    }

    #endregion

    #region MultiOutputGaussianProcess Tests

    [Fact]
    public void MultiOutputGaussianProcess_Construction_WithGaussianKernel_Succeeds()
    {
        var kernel = new GaussianKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void MultiOutputGaussianProcess_Construction_WithLinearKernel_Succeeds()
    {
        var kernel = new LinearKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void MultiOutputGaussianProcess_Construction_WithPolynomialKernel_Succeeds()
    {
        var kernel = new PolynomialKernel<double>();
        var gp = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void MultiOutputGaussianProcess_Float_Construction_Succeeds()
    {
        var kernel = new GaussianKernel<float>();
        var gp = new MultiOutputGaussianProcess<float>(kernel);

        Assert.NotNull(gp);
    }

    #endregion

    #region Cross-GP Tests with Different Kernels

    [Fact]
    public void AllGaussianProcessTypes_WithExponentialKernel_Succeed()
    {
        var kernel = new ExponentialKernel<double>();

        var standard = new StandardGaussianProcess<double>(kernel);
        var sparse = new SparseGaussianProcess<double>(kernel);
        var multiOutput = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(standard);
        Assert.NotNull(sparse);
        Assert.NotNull(multiOutput);
    }

    [Fact]
    public void AllGaussianProcessTypes_WithMaternKernel_Succeed()
    {
        var kernel = new MaternKernel<double>();

        var standard = new StandardGaussianProcess<double>(kernel);
        var sparse = new SparseGaussianProcess<double>(kernel);
        var multiOutput = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(standard);
        Assert.NotNull(sparse);
        Assert.NotNull(multiOutput);
    }

    [Fact]
    public void AllGaussianProcessTypes_WithRationalQuadraticKernel_Succeed()
    {
        var kernel = new RationalQuadraticKernel<double>();

        var standard = new StandardGaussianProcess<double>(kernel);
        var sparse = new SparseGaussianProcess<double>(kernel);
        var multiOutput = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(standard);
        Assert.NotNull(sparse);
        Assert.NotNull(multiOutput);
    }

    [Fact]
    public void AllGaussianProcessTypes_WithCauchyKernel_Succeed()
    {
        var kernel = new CauchyKernel<double>();

        var standard = new StandardGaussianProcess<double>(kernel);
        var sparse = new SparseGaussianProcess<double>(kernel);
        var multiOutput = new MultiOutputGaussianProcess<double>(kernel);

        Assert.NotNull(standard);
        Assert.NotNull(sparse);
        Assert.NotNull(multiOutput);
    }

    #endregion
}
