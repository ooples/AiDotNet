using AiDotNet.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class CLARANSTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new CLARANS<double>(new AiDotNet.Clustering.Options.CLARANSOptions<double>
        {
            NumClusters = NumClusters,
            NumLocal = 5, // More restarts for better convergence
            Seed = 42
        });
}
