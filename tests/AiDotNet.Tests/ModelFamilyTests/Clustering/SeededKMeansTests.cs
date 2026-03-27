using AiDotNet.Interfaces;
using AiDotNet.Clustering.SemiSupervised;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class SeededKMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new SeededKMeans<double>(new AiDotNet.Clustering.Options.SeededKMeansOptions<double>
        {
            NumClusters = NumClusters,
            Seed = 42
        });
}
