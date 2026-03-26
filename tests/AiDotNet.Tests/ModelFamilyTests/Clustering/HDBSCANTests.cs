using AiDotNet.Interfaces;
using AiDotNet.Clustering.Density;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class HDBSCANTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new HDBSCAN<double>(new AiDotNet.Clustering.Options.HDBSCANOptions<double>
        {
            MinClusterSize = 3,
            MinSamples = 3,
            ClusterSelection = AiDotNet.Clustering.Options.HDBSCANClusterSelection.Leaf
        });

    // HDBSCAN is density-based — doesn't have centroid parameters
    protected override bool HasFlatParameters => false;
}
