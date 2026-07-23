using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.AnomalyDetection.TreeBased;
using AiDotNet.Data.Loaders;
using AiDotNet.GaussianProcesses;
using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Pins that Gaussian processes and anomaly detectors, now that IGaussianProcess and IAnomalyDetector
/// derive from IFullModel&lt;T, Matrix&lt;T&gt;, Vector&lt;T&gt;&gt;, flow through ConfigureModel like any
/// other model — so their dedicated (and previously inert) Configure methods were correctly removed.
/// </summary>
/// <remarks>
/// The migration is a formalization: both bases already implement the full IFullModel contract via
/// ModelBase, exactly as ClusteringBase does. These tests prove the type system and the build pipeline
/// accept them through the single ConfigureModel door.
/// </remarks>
public class FullModelMigrationBuildTests
{
    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 40, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            y[i] = Math.Cos(i * 0.2);
        }

        return (x, y);
    }

    [Fact]
    public void GaussianProcess_IsAnIFullModel()
    {
        var gp = new StandardGaussianProcess<double>(new GaussianKernel<double>(1.0));
        Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, Vector<double>>>(gp);
    }

    [Fact]
    public void AnomalyDetector_IsAnIFullModel()
    {
        var detector = new IsolationForest<double>();
        Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, Vector<double>>>(detector);
    }

    [Fact(Timeout = 120000)]
    public async Task GaussianProcess_BuildsThroughConfigureModel()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new StandardGaussianProcess<double>(new GaussianKernel<double>(1.0)))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task AnomalyDetector_BuildsThroughConfigureModel()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new IsolationForest<double>(numTrees: 20, maxSamples: 16))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result);
    }
}
