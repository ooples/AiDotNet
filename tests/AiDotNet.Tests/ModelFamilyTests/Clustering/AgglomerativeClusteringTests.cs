using AiDotNet.Interfaces;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class AgglomerativeClusteringTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new AgglomerativeClustering<double>(new HierarchicalOptions<double>
        {
            NumClusters = NumClusters
        });

    // Hierarchical clustering doesn't have flat centroid parameters
    protected override bool HasFlatParameters => false;
}
