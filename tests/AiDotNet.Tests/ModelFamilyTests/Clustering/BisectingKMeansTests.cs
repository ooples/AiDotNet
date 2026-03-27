using AiDotNet.Interfaces;
using AiDotNet.Clustering.Hierarchical;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class BisectingKMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new BisectingKMeans<double>(new AiDotNet.Clustering.Options.BisectingKMeansOptions<double> { NumClusters = NumClusters });
}
