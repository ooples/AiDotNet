using AiDotNet.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class KMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new KMeans<double>(new KMeansOptions<double> { NumClusters = NumClusters });
}
