using AiDotNet.Interfaces;
using AiDotNet.Clustering.Subspace;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class CLIQUETests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new CLIQUE<double>();

    // For single-cluster data (90 points, std=0.01), the default 10 intervals
    // creates borderline dense cells. Use fewer intervals so all points fall
    // into 1-2 cells per dimension, meeting the density threshold easily.
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => new CLIQUE<double>(new CLIQUEOptions<double>
        {
            NumIntervals = 3,
            DensityThreshold = 0.05
        });
}
