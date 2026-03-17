using AiDotNet.Interfaces;
using AiDotNet.Clustering.Ensemble;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class ConsensusClusteringTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new ConsensusClustering<double>(new AiDotNet.Clustering.Ensemble.ConsensusClusteringOptions<double>
        {
            NumClusters = NumClusters,
            Seed = 42
        });
}
