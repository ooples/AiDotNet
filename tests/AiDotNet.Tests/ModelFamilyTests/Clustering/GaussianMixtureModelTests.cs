using AiDotNet.Interfaces;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class GaussianMixtureModelTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new GaussianMixtureModel<double>(new GMMOptions<double>
        {
            NumComponents = NumClusters
        });
}
