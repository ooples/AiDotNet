using AiDotNet.Interfaces;
using AiDotNet.Clustering.Neural;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class SelfOrganizingMapTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new SelfOrganizingMap<double>(new AiDotNet.Clustering.Options.SOMOptions<double>
        {
            GridWidth = 5,
            GridHeight = 5,
            MaxIterations = 2000, // More iterations for better convergence
            Seed = 42
        });
}
