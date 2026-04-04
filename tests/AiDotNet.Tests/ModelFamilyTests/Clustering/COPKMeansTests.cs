using AiDotNet.Interfaces;
using AiDotNet.Clustering.SemiSupervised;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class COPKMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new COPKMeans<double>();

    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => new COPKMeans<double>(new AiDotNet.Clustering.Options.COPKMeansOptions<double>
        {
            NumClusters = 1
        });
}
