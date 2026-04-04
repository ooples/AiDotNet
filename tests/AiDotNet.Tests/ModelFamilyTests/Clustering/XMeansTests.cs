using AiDotNet.Interfaces;
using AiDotNet.Clustering.AutoK;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class XMeansTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new XMeans<double>();

    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateSingleClusterModel()
        => new XMeans<double>(new AiDotNet.Clustering.Options.XMeansOptions<double>
        {
            MaxClusters = 2
        });
}
